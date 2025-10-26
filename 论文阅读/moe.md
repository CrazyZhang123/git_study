---
title: "混合专家模型（MoE）详解"
thumbnail: /blog/assets/moe/thumbnail.png
authors:
- user: osanseviero
- user: lewtun
- user: philschmid
- user: smangrul
- user: ybelkada
- user: pcuenq
translators:
- user: xinyu66
- user: zhongdongy
  proofreader: true
---

# 混合专家模型 (MoE) 详解

随着 Mixtral 8x7B ([announcement](https://mistral.ai/news/mixtral-of-experts/), [model card](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)) 的推出，一种称为混合专家模型 (Mixed Expert Models，简称 MoEs) 的 Transformer 模型在开源人工智能社区引起了广泛关注。在本篇博文中，我们将深入探讨 MoEs 的核心组件、训练方法，以及在推理过程中需要考量的各种因素。

让我们开始吧！

## 目录

- [什么是混合专家模型？](#什么是混合专家模型)
- [混合专家模型简史](#混合专家模型简史)
- [什么是稀疏性？](#什么是稀疏性)
- [混合专家模型中令牌的负载均衡](#混合专家模型中令牌的负载均衡)
- [MoEs and Transformers](#moes-and-transformers)
- [Switch Transformers](#switch-transformers)
- [用Router z-loss稳定模型训练](#用router-z-loss稳定模型训练)
- [专家如何学习？](#专家如何学习)
- [专家的数量对预训练有何影响？](#专家的数量对预训练有何影响)
- [微调混合专家模型](#微调混合专家模型)
- [稀疏 VS 稠密，如何选择?](#稀疏-VS-稠密如何选择)
- [让MoE起飞](#让moe起飞)
  - [并行计算](#并行计算)
  - [容量因子和通信开销](#容量因子和通信开销)
  - [部署技术](#部署技术)
  - [高效训练](#高效训练)
- [开源混合专家模型](#开源混合专家模型)
- [一些有趣的研究方向](#一些有趣的研究方向)
- [相关资源](#相关资源)
- [简单代码实现](#简单代码实现)

## 简短总结

混合专家模型 (MoEs):

- 与稠密模型相比， **预训练速度更快**
- 与具有相同参数数量的模型相比，具有更快的 **推理速度**
- 需要 **大量显存**，因为所有专家系统都需要加载到内存中
- 在 **微调方面存在诸多挑战**，但 [近期的研究](https://arxiv.org/pdf/2305.14705.pdf) 表明，对混合专家模型进行 **指令调优具有很大的潜力**。

让我们开始吧！

## 什么是混合专家模型？

模型规模是提升模型性能的关键因素之一。在有限的计算资源预算下，用更少的训练步数训练一个更大的模型，往往比用更多的步数训练一个较小的模型效果更佳。

混合专家模型 (MoE) 的一个显著优势是它们能够在远少于稠密模型所需的计算资源下进行有效的预训练。这意味着在相同的计算预算条件下，您可以显著扩大模型或数据集的规模。特别是在预训练阶段，与稠密模型相比，混合专家模型通常能够更快地达到相同的质量水平。

那么，究竟什么是一个混合专家模型 (MoE) 呢？作为一种基于 Transformer 架构的模型，混合专家模型主要由两个关键部分组成:

- **稀疏 MoE 层**: 这些层代替了传统 Transformer 模型中的前馈网络 (FFN) 层。MoE 层包含若干“专家”(例如 8 个)，每个专家本身是一个独立的神经网络。在实际应用中，这些专家通常是前馈网络 (FFN)，但它们也可以是更复杂的网络结构，甚至可以是 MoE 层本身，从而形成层级式的 MoE 结构。
- **门控网络或路由**: 这个部分用于决定哪些令牌 (token) 被发送到哪个专家。例如，在下图中，“More”这个令牌可能被发送到第二个专家，而“Parameters”这个令牌被发送到第一个专家。有时，一个令牌甚至可以被发送到多个专家。令牌的路由方式是 MoE 使用中的一个关键点，因为路由器由学习的参数组成，并且与网络的其他部分一同进行预训练。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png" alt="Switch Layer">
  <figcaption>[Switch Transformers paper](https://arxiv.org/abs/2101.03961) 论文中的 MoE layer</figcaption>
</figure>

总结来说，在混合专家模型 (MoE) 中，我们将传统 Transformer 模型中的每个前馈网络 (FFN) 层替换为 MoE 层，其中 MoE 层由两个核心部分组成: 一个门控网络和若干数量的专家。

尽管混合专家模型 (MoE) 提供了若干显著优势，例如更高效的预训练和与稠密模型相比更快的推理速度，但它们也伴随着一些挑战:

- **训练挑战**: 虽然 MoE 能够实现更高效的计算预训练，但它们在微调阶段往往面临泛化能力不足的问题，长期以来易于引发过拟合现象。
- **推理挑战**: MoE 模型虽然可能拥有大量参数，但在推理过程中只使用其中的一部分，这使得它们的推理速度快于具有相同数量参数的稠密模型。然而，这种模型需要将所有参数加载到内存中，因此对内存的需求非常高。以 Mixtral 8x7B 这样的 MoE 为例，需要足够的 VRAM 来容纳一个 47B 参数的稠密模型。之所以是 47B 而不是 8 x 7B = 56B，是因为在 MoE 模型中，只有 FFN 层被视为独立的专家，而模型的其他参数是共享的。此外，假设每个令牌只使用两个专家，那么推理速度 (以 FLOPs 计算) 类似于使用 12B 模型 (而不是 14B 模型)，因为虽然它进行了 2x7B 的矩阵乘法计算，但某些层是共享的。

了解了 MoE 的基本概念后，让我们进一步探索推动这类模型发展的研究。

## 混合专家模型简史

混合专家模型 (MoE) 的理念起源于 1991 年的论文 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)。这个概念与集成学习方法相似，旨在为由多个单独网络组成的系统建立一个监管机制。在这种系统中，每个网络 (被称为“专家”) 处理训练样本的不同子集，专注于输入空间的特定区域。那么，如何选择哪个专家来处理特定的输入呢？这就是门控网络发挥作用的地方，它决定了分配给每个专家的权重。在训练过程中，这些专家和门控网络都同时接受训练，以优化它们的性能和决策能力。

在 2010 至 2015 年间，两个独立的研究领域为混合专家模型 (MoE) 的后续发展做出了显著贡献:

1. **组件专家**: 在传统的 MoE 设置中，整个系统由一个门控网络和多个专家组成。在支持向量机 (SVMs) 、高斯过程和其他方法的研究中，MoE 通常被视为整个模型的一部分。然而，[Eigen、Ranzato 和 Ilya 的研究](https://arxiv.org/abs/1312.4314) 探索了将 MoE 作为更深层网络的一个组件。这种方法允许将 MoE 嵌入到多层网络中的某一层，使得模型既大又高效。
2. **条件计算**: 传统的神经网络通过每一层处理所有输入数据。在这一时期，Yoshua Bengio 等研究人员开始探索基于输入令牌动态激活或停用网络组件的方法。

这些研究的融合促进了在自然语言处理 (NLP) 领域对混合专家模型的探索。特别是在 2017 年，[Shazeer 等人](https://arxiv.org/abs/1701.06538) (团队包括 Geoffrey Hinton 和 Jeff Dean，后者有时被戏称为 [“谷歌的 Chuck Norris”](https://www.informatika.bg/jeffdean)) 将这一概念应用于 137B 的 LSTM (当时被广泛应用于 NLP 的架构，由 Schmidhuber 提出)。通过引入稀疏性，这项工作在保持极高规模的同时实现了快速的推理速度。这项工作主要集中在翻译领域，但面临着如高通信成本和训练不稳定性等多种挑战。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png" alt="MoE layer in LSTM">
  <figcaption>Outrageously Large Neural Network 论文中的 MoE layer</figcaption>
</figure>

混合专家模型 (MoE) 的引入使得训练具有数千亿甚至万亿参数的模型成为可能，如开源的 1.6 万亿参数的 Switch Transformers 等。这种技术不仅在自然语言处理 (NLP) 领域得到了广泛应用，也开始在计算机视觉领域进行探索。然而，本篇博客文章将主要聚焦于自然语言处理领域的应用和探讨。

## 什么是稀疏性?

稀疏性的概念采用了条件计算的思想。在传统的稠密模型中，所有的参数都会对所有输入数据进行处理。相比之下，稀疏性允许我们仅针对整个系统的某些特定部分执行计算。这意味着并非所有参数都会在处理每个输入时被激活或使用，而是根据输入的特定特征或需求，只有部分参数集合被调用和运行。

让我们深入分析 Shazeer 对混合专家模型 (MoE) 在翻译应用中的贡献。条件计算的概念 (即仅在每个样本的基础上激活网络的不同部分) 使得在不增加额外计算负担的情况下扩展模型规模成为可能。这一策略在每个 MoE 层中实现了数以千计甚至更多的专家的有效利用。

这种稀疏性设置确实带来了一些挑战。例如，在混合专家模型 (MoE) 中，尽管较大的批量大小通常有利于提高性能，但当数据通过激活的专家时，实际的批量大小可能会减少。比如，假设我们的输入批量包含 10 个令牌， **可能会有五个令牌被路由到同一个专家，而剩下的五个令牌分别被路由到不同的专家。这导致了批量大小的不均匀分配和资源利用效率不高的问题**。在接下来的部分中，将会讨论 [让 MoE 高效运行](#让moe起飞) 的其他挑战以及相应的解决方案。

那我们应该如何解决这个问题呢？一个可学习的门控网络 (G) 决定将输入的哪一部分发送给哪些专家 (E):

$$
y = \sum_{i=1}^{n} G(x)_i E_i(x)
$$

在这种设置下，虽然所有专家都会对所有输入进行运算，但通过门控网络的输出进行加权乘法操作。但是，如果 G (门控网络的输出) 为 0 会发生什么呢？如果是这种情况，就没有必要计算相应的专家操作，因此我们可以节省计算资源。那么一个典型的门控函数是什么呢？一个典型的门控函数通常是一个带有 softmax 函数的简单的网络。这个网络将学习将输入发送给哪个专家。

$$
G_\sigma(x) = \text{Softmax}(x \cdot W_g)
$$

Shazeer 等人的工作还探索了其他的门控机制，其中包括带噪声的 TopK 门控 (Noisy Top-K Gating)。这种门控方法引入了一些可调整的噪声，然后保留前 k 个值。具体来说:

1. 添加一些噪声

$$
H(x)_i = (x \cdot W_{\text{g}})_i + \text{StandardNormal()} \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)
$$

- **A.专家基础得分（确定性部分）**

  (*x*⋅*W*g)*i*

  - **x：** 当前的输入向量（来自 Transformer 等前一层）。
  - **Wg：** 门控网络（Gating Network）中用于计算**专家重要性**的权重矩阵。
  - **(x\⋅Wg)i：** 这是对专家 *i* 的**线性评分**。它度量了根据输入 *x* 和专家 *i* 的专业领域权重 *W*g，该专家有多重要或多适合处理当前输入。这是一个**确定性**的得分。

  

  #### B. 可调控的噪声项（随机性部分）

  StandardNormal()⋅Softplus((*x*⋅*W*noise)*i*)

  - **StandardNormal()：** 这是一个**随机数**，来自标准正态分布（均值为 0，方差为 1）。它提供了门控的**随机性**。
  - **(xWnoise)i：** 这是门控网络中用于计算**噪声强度**的另一个权重矩阵 *W*noise 产生的线性得分。
  - **Softplus(⋅)：** Softplus 是一个平滑且非负的激活函数，其形状类似于 ReLU，但曲线平滑。这里的 Softplus 函数将线性得分 (*x*⋅*W*noise)*i* 转换成一个**非负的、可学习的缩放因子**。
  - **整个噪声项：** 随机数乘以这个缩放因子。这意味着：
    - **噪声强度是可学习的：** 模型可以学习调整 *W*noise，从而控制向每个专家添加的噪声的**大小**。
    - **噪声的作用：** 这种噪声鼓励在激活值相近的专家之间进行**随机选择**，有助于防止模型过度依赖少数几个专家，从而促进所有专家的均匀利用和负载平衡。

2. 选择保留前 K 个值

$$
\text{KeepTopK}(v, k)_i = \begin{cases}
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v, \\
-\infty & \text{otherwise.}
\end{cases}
$$

3. 应用 Softmax 函数

$$
G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))
$$

这种稀疏性引入了一些有趣的特性。通过使用较低的 k 值 (例如 1 或 2)，我们可以比激活多个专家时更快地进行训练和推理。为什么不仅选择最顶尖的专家呢？最初的假设是，需要将输入路由到不止一个专家，以便门控学会如何进行有效的路由选择，因此至少需要选择两个专家。[Switch Transformers](#switch-transformers) 就这点进行了更多的研究。

我们为什么要添加噪声呢？这是为了专家间的负载均衡！





## 混合专家模型中令牌的负载均衡

正如之前讨论的，如果所有的令牌都被发送到只有少数几个受欢迎的专家，那么训练效率将会降低。在通常的混合专家模型 (MoE) 训练中，门控网络往往倾向于主要激活相同的几个专家。这种情况可能会自我加强，因为受欢迎的专家训练得更快，因此它们更容易被选择。为了缓解这个问题，引入了一个 **辅助损失**，==旨在鼓励给予所有专家相同的重要性。==这个损失确保所有专家接收到大致相等数量的训练样本，从而平衡了专家之间的选择。接下来的部分还将探讨**专家容量**的概念，它引入了==一个关于专家可以处理多少令牌的阈值==。在 `transformers` 库中，可以通过 `aux_loss` 参数来控制辅助损失。

## MoEs and Transformers

Transformer 类模型明确表明，增加参数数量可以提高性能，因此谷歌使用 [GShard](https://arxiv.org/abs/2006.16668) 尝试将 Transformer 模型的参数量扩展到超过 6000 亿并不令人惊讶。

GShard 将在编码器和解码器中的每个前馈网络 (FFN) 层中的替换为使用 Top-2 门控的混合专家模型 (MoE) 层。下图展示了编码器部分的结构。这种架构对于大规模计算非常有效: ==当扩展到多个设备时，MoE 层在不同设备间共享，而其他所有层则在每个设备上复制==。我们将在 [“让 MoE 起飞”](#让moe起飞) 部分对这一点进行更详细的讨论。

> - **FFN 的作用回顾：** 在标准的 Transformer 结构中，FFN 层占据了模型参数的**绝大部分**（约三分之二）。它负责对自注意力层提取的特征进行深度的非线性转换。
>
> - | 模块类型                        | 复制方式                  | 并行策略                         | 作用                                                         |
>   | ------------------------------- | ------------------------- | -------------------------------- | ------------------------------------------------------------ |
>   | **MoE 层（专家组）**            | 在**不同设备间共享/分散** | **模型并行** (Model Parallelism) | 将数千个专家分散到数百个 GPU 上，从而将模型的总参数量扩展到万亿级别，突破单个设备的内存限制。 |
>   | **其他层（Attention, 归一化）** | 在**每个设备上复制**      | **数据并行** (Data Parallelism)  | 每个设备都处理一部分训练数据，并独立执行这些参数相同的层，提高训练速度。 |
>   

> **“共享”的含义**： 所有的设备共同（或曰“共享”）存储了整个 MoE 层的全部参数。如果一个设备想访问一个不在它本地的专家，它需要通过网络通信到存储该专家的设备。
>
> **数据流的并行：**
>
> - 假设总批次大小（Batch Size）为 *B*=1000。
> - 这 1000 个数据样本会被**平均分割**成 100 份，每份 10 个样本。
> - **每个设备**接收 10 个样本。
> - **独立计算：** 每个设备独立地使用它本地存储的**模型参数副本**计算这 10 个样本的前向传播和损失（Loss）。
> - **同步更新：** 在反向传播过程中，每个设备计算出自己的梯度（对参数的修正量）。这些梯度随后被**汇总（All-reduce）**、平均，然后用来更新**所有设备上的参数副本**，确保每个设备上的参数始终保持一致。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/02_moe_block.png" alt="MoE Transformer Encoder">
  <figcaption>GShard 论文中的 MoE Transformer Encoder</figcaption>
</figure>

为了保持负载平衡和训练效率，GShard 的作者除了引入了上一节中讨论的类似辅助损失外，还引入了一些关键变化:

- **随机路由**: 在 Top-2 设置中，我们始终选择排名最高的专家，但第二个专家是根据其权重比例随机选择的。

> **随机路由的核心意义**
>
> 1. **提升专家模块的利用率**
>
>    Top-2 路由若完全选择 “权重最高的两个专家”，会导致部分 “次优但有用” 的专家长期闲置。按权重比例随机选择第二个专家，能让更多专家参与不同样本的计算，避免模型过度依赖少数高频激活的专家，充分发挥 MoE “多专家分工” 的优势。
>
> 2. **增强模型的泛化能力与鲁棒性**
>
>    固定选择 Top-2 专家容易让模型对 “高频样本模式” 产生过拟合，面对分布外样本时灵活性不足。随机选择第二个专家会为模型引入可控的 “不确定性”，迫使模型学习更通用的特征表示，而非死记硬背特定样本的最优专家组合，从而在陌生场景（如长尾任务）中表现更稳定。
>
> 3. **平衡计算成本与性能收益**
>
>    MoE 的核心矛盾是 “专家数量（性能潜力）” 与 “推理成本（每次激活的专家数）” 的权衡。Top-2 随机路由在 “仅激活 2 个专家” 的低成本前提下，通过随机化扩展了专家的覆盖范围，既避免了 Top-1 路由的表达能力不足，又无需承担 Top-K（K>2）路由的高额计算开销，实现了成本与性能的最优平衡点。

- **专家容量**: 我们可以设定一个阈值，==定义一个专家能处理多少令牌。如果两个专家的容量都达到上限，令牌就会溢出，并通过残差连接传递到下一层，或在某些情况下被完全丢弃==。专家容量是 MoE 中最重要的概念之一。为什么需要专家容量呢？因为所有张量的形状在编译时是静态确定的，我们无法提前知道多少令牌会分配给每个专家，因此需要一个固定的容量因子。

GShard 的工作对适用于 MoE 的并行计算模式也做出了重要贡献，但这些内容的讨论超出了这篇博客的范围。

**注意**: 在推理过程中，只有部分专家被激活。同时，有些计算过程是共享的，例如自注意力 (self-attention) 机制，它适用于所有令牌。这就解释了为什么我们可以使用相当于 12B 稠密模型的计算资源来运行一个包含 8 个专家的 47B 模型。如果我们采用 Top-2 门控，模型会使用高达 14B 的参数。但是，由于自注意力操作 (专家间共享) 的存在，实际上模型运行时使用的参数数量是 12B。

## Switch Transformers

尽管混合专家模型 (MoE) 显示出了很大的潜力，但它们在训练和微调过程中存在稳定性问题。[Switch Transformers](https://arxiv.org/abs/2101.03961) 是一项非常激动人心的工作，它深入研究了这些话题。作者甚至在 Hugging Face 上发布了一个 [1.6 万亿参数的 MoE](https://huggingface.co/google/switch-c-2048)，拥有 2048 个专家，你可以使用 `transformers` 库来运行它。Switch Transformers 实现了与 T5-XXL 相比 4 倍的预训练速度提升。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png" alt="Switch Transformer Layer">
  <figcaption>Switch Transformer 论文中的 Switch Transformer Layer</figcaption>
</figure>

就像在 GShard 中一样，作者用混合专家模型 (MoE) 层替换了前馈网络 (FFN) 层。**Switch Transformers 提出了一个 Switch Transformer 层，它接收两个输入 (两个不同的令牌) 并拥有四个专家。**

与最初使用至少两个专家的想法相反，Switch Transformers 采用了简化的单专家策略。这种方法的效果包括:

- 减少门控网络 (路由) 计算负担
- 每个专家的批量大小至少可以减半
- 降低通信成本
- 保持模型质量

Switch Transformers 也对 **专家容量** 这个概念进行了研究。

$$
\text{Expert Capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}
$$

上述建议的容量是将批次中的令牌数量均匀分配到各个专家。如果我们使用大于 1 的容量因子，我们为令牌分配不完全平衡时提供了一个缓冲。

> 意思是说，给定m个expert需要处理batch条数据(token),每个expert分到的样本(tokne)数量是括号里面的相除的结果；但现在乘了一个容量因子(>1),说明现在的expert 可以处理的capactity已经超过了这个batch的样本(token)数目，因此具有了缓存token的效果。

==增加容量因子会导致更高的设备间通信成本，因此这是一个需要考虑的权衡==。特别值得注意的是，Switch Transformers 在**低容量因子 (例如 1 至 1.25)** 下表现出色。

Switch Transformer 的作者还重新审视并简化了前面章节中提到的负载均衡损失。在训练期间，对于每个 Switch 层的辅助损失被添加到总模型损失中。这种损失鼓励均匀路由，并可以使用超参数进行加权。

**作者还尝试了混合精度的方法，例如用 `bfloat16` 精度训练专家，同时对其余计算使用全精度进行。较低的精度可以减少处理器间的通信成本、计算成本以及存储张量的内存。**然而，在最初的实验中，当专家和门控网络都使用 `bfloat16` 精度训练时，出现了不稳定的训练现象。==这种不稳定性特别是由路由计算引起的，因为路由涉及指数函数等操作，这些操作对精度要求较高。==因此，为了保持计算的稳定性和精确性，保持更高的精度是重要的。为了减轻不稳定性，路由过程也使用了全精度。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/04_switch_table.png" alt="Table shows that selective precision does not degrade quality.">
  <figcaption>使用混合精度不会降低模型质量并可实现更快的训练</figcaption>
</figure>

这个 [Jupyter Notebook](https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4?usp=sharing) 展示了如何对 Switch Transformers 进行微调以进行摘要生成的详细指南。然而，在开始微调 Switch Transformers 之前，强烈建议您先阅读关于 [微调混合专家模型](#微调混合专家模型) 部分的内容。

Switch Transformers 采用了编码器 - 解码器的架构，实现了与 T5 类似的混合专家模型 (MoE) 版本。[GLaM](https://arxiv.org/abs/2112.06905) 这篇工作探索了如何使用仅为原来 1/3 的计算资源 (因为 MoE 模型在训练时需要的计算量较少，从而能够显著降低碳足迹) 来训练与 GPT-3 质量相匹配的模型来提高这些模型的规模。==作者专注于仅解码器 (decoder-only) 的模型以及少样本和单样本评估，而不是微调。他们使用了 Top-2 路由和更大的容量因子。此外，他们探讨了将容量因子作为一个动态度量，根据训练和评估期间所使用的计算量进行调整。==

## 用 Router z-loss 稳定模型训练

之前讨论的平衡损失可能会导致稳定性问题。我们可以使用许多方法来稳定稀疏模型的训练，但这可能会牺牲模型质量。例如，引入 dropout 可以提高稳定性，但会导致模型质量下降。另一方面，增加更多的乘法分量可以提高质量，但会降低模型稳定性。

==[ST-MoE](https://arxiv.org/abs/2202.08906) 引入的 `Router z-loss` 在保持了模型性能的同时显著提升了训练的稳定性。==这种损失机制**通过惩罚门控网络输入的较大 `logits` 来起作用，目的是促使数值的绝对大小保持较小，这样可以有效减少计算中的舍入误差。**这一点对于那些依赖指数函数进行计算的门控网络尤其重要。为了深入了解这一机制，建议参考原始论文以获得更全面的细节。

> ##### 1、稀疏模型训练中 “稳定性” 的具体含义
>
> 在 MoE 这类稀疏模型中，“不稳定” 不是指训练崩溃，而是指训练过程难以稳定收敛，或最终模型性能波动大，具体体现在三个方面：
>
> - **梯度波动大**：稀疏激活导致每次更新时，只有部分专家的参数被调整，容易出现梯度骤增 / 骤减，使参数更新方向不稳定。
> - **专家激活失衡**：若路由策略偏向少数专家，会导致 “热门专家” 参数更新过度、“冷门专家” 参数几乎不动，模型整体学习节奏混乱。
> - **性能一致性差**：相同训练条件下，多次训练的最终测试精度差异大，无法稳定复现最优结果。
>
> ##### 2、为什么 “平衡损失的替代方案” 会陷入 “稳定性 - 质量” 矛盾？
>
> 两种方案，**本质是**通过 “牺牲某一维度” 来优化另一维度，核心原因是它们没有从根本上解决稀疏模型的 “激活稀疏性” 与 “参数更新均衡性” 的核心冲突。
>
> ###### 方案 1：引入 Dropout 提升稳定性，但降低模型质量
>
> - 提升稳定性的逻辑
>
>   ：Dropout 会随机 “关闭” 部分专家或路由连接，强制模型不依赖固定的专家组合。
>
>   - 避免了少数专家被过度激活，让梯度更新更分散，减少梯度波动；
>   - 防止模型 “死记硬背” 特定样本的专家路由模式，训练过程更平稳。
>
> - 降低模型质量的原因
>
>   ：Dropout 的随机性本质是 “主动破坏模型的有效特征提取路径”。
>
>   - 稀疏模型本身依赖 “专家分工” 提取精细特征，Dropout 会导致关键专家被随机屏蔽，有用信息丢失；
>   - 过多的随机性会让模型难以学习到稳定的映射关系，最终泛化能力下降（比如对长尾样本的适配性变弱）。
>
> ###### 方案 2：增加乘法分量提升质量，但降低稳定性
>
> - 提升模型质量的逻辑
>
>   ：“乘法分量” 通常指路由权重计算中加入更多非线性交互项（如专家特征与输入特征的乘法结合）。
>
>   - 让路由决策更精细，能更精准地匹配 “输入样本 - 专家能力”，充分发挥不同专家的分工优势；
>   - 增强模型对复杂模式的拟合能力，比如处理多模态输入或歧义样本时，表现更优。
>
> - 降低稳定性的原因
>
>   ：乘法分量会放大 “激活稀疏性” 带来的不平衡。
>
>   - 非线性交互会让少数专家的权重被过度放大（比如某专家对特定输入的响应通过乘法项骤增），加剧 “热门专家更热、冷门专家更冷” 的问题；
>   - 权重差异变大后，梯度更新会更偏向热门专家，导致参数更新失衡，训练过程波动加剧（比如损失曲线上下震荡幅度变大）。

## 专家如何学习？

ST-MoE 的研究者们发现，编码器中不同的专家倾向于专注于特定类型的令牌或浅层概念。例如，某些专家可能专门处理标点符号，而其他专家则专注于专有名词等。与此相反，解码器中的专家通常具有较低的专业化程度。此外，研究者们还对这一模型进行了多语言训练。尽管人们可能会预期每个专家处理一种特定语言，但实际上并非如此。**由于令牌路由和负载均衡的机制，没有任何专家被特定配置以专门处理某一特定语言。**

 

![img](https://cas-bridge.xethub.hf.co/xet-bridge-us/621ffdd236468d709f1835cf/98e65371dad06ddb89b3e4a32f10695c6479c76e7fb2278341a3e330c1c4f9e3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251015%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251015T022326Z&X-Amz-Expires=3600&X-Amz-Signature=c948a2b8eae3ba0b962e86d7ab630fcfd27db17f72008d05de2888ea7c9dca58&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=66fa82dbdf4d7ebc64b131d6&response-content-disposition=inline%3B+filename*%3DUTF-8''05_experts_learning.png%3B+filename%3D"05_experts_learning.png"%3B&response-content-type=image%2Fpng&x-id=GetObject&Expires=1760498606&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MDQ5ODYwNn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MjFmZmRkMjM2NDY4ZDcwOWYxODM1Y2YvOThlNjUzNzFkYWQwNmRkYjg5YjNlNGEzMmYxMDY5NWM2NDc5Yzc2ZTdmYjIyNzgzNDFhM2UzMzBjMWM0ZjllMyoifV19&Signature=UeuiORZgin25T5YzuJEYYBJL-XTFb0fjSfsMmDtk6v3kTpH1QxxDbOFi8IB6KvlvVVAXWSPndMyW0bOmTgxLJ6-OP5-2mpRsOmOAM3~l5H3QesRqdy954uTp3ZK8wza3Hkr1sX3gdRCqe2h7trWQ8mzIFGR-RGrUfNMmKszAZjNBEHCcGMKsmOahlvsfqcVlk3rwK~n8PyKmNuHa0WfvElYgMeOomN-eC63lI940OcDBEPOZKoh1VgdtXObzu8hf9-CP9TEjKrWXOTBAjF2KkcwX3SEFTJuQuDMY~PjP-pkGVz3TpVcfGSHcUlPm7LejeCwI25PDb-yPoKFF5gbiZQ__&Key-Pair-Id=K2L8F4GPSG1IFC)  

<figcaption>ST-MoE 论文中显示了哪些令牌组被发送给了哪个专家的表格</figcaption>


## 专家的数量对预训练有何影响？

增加更多专家可以提升处理样本的效率和加速模型的运算速度，**但这些优势随着专家数量的增加而递减 (尤其是当专家数量达到 256 或 512 之后更为明显)。**同时，这也意味着在推理过程中，需要更多的显存来加载整个模型。值得注意的是，Switch Transformers 的研究表明，其在大规模模型中的特性在小规模模型下也同样适用，即便是每层仅包含 2、4 或 8 个专家。

## 微调混合专家模型

> `4.36.0` 版本的 `transformers` 库支持 Mixtral 模型。你可以用以下命令进行安装: `pip install "transformers==4.36.0 --upgrade`

稠密模型和稀疏模型在过拟合的动态表现上存在显著差异**。稀疏模型更易于出现过拟合现象**，因此在处理这些模型时，尝试更强的内部正则化措施是有益的，比如使用更高比例的 dropout。例如，我们可以为稠密层设定一个较低的 dropout 率，而为稀疏层设置一个更高的 dropout 率，以此来优化模型性能。

在微调过程中是否使用辅助损失是一个需要决策的问题。ST-MoE 的作者尝试关闭辅助损失，发现即使高达 11% 的令牌被丢弃，模型的质量也没有显著受到影响。**令牌丢弃可能是一种正则化形式，有助于防止过拟合。**

Switch Transformers 的作者观察到，==在相同的预训练困惑度下，稀疏模型在下游任务中的表现不如对应的稠密模型==，特别是在重理解任务 (如 SuperGLUE) 上。另一方面，对于知识密集型任务 (如 TriviaQA)，稀疏模型的表现异常出色。作者还观察到，在微调过程中，较少的专家的数量有助于改善性能。另一个关于泛化问题确认的发现是，**模型在小型任务上表现较差，但在大型任务上表现良好。**

> ##### 1.“任务大小与 MoE 过拟合” 的关系
>
> - **小任务（左图）过拟合**：小任务的数据量少、场景单一，MoE 层中众多 “专家” 的分工能力无法被充分利用。此时模型容易让少数专家过度适配小任务的有限样本，导致在验证集上泛化能力差（即过拟合）。
> - **大任务（右图）表现好**：大任务的数据量大、场景多样，能充分激活 MoE 层的 “专家分工” 优势 —— 不同专家可分别适配任务中的不同子场景（如文本分类任务中，有的专家处理情感类文本，有的处理事实类文本），避免单一专家过度拟合，自然泛化性更优。
>
> ##### 2. 为什么 “冻结非专家层性能暴跌，冻结 MoE 层却几乎等效全量微调”？
>
> 这背后的核心原因是：**MoE 模型的 “能力核心” 集中在 MoE 层，而非普通层（如 Transformer 的注意力层、Feed-Forward 普通层）**。
>
> #### 情况 1：冻结所有非专家层 → 性能大幅下降
>
> - MoE 模型的普通层（非专家层），本质是 “基础特征处理器”，负责将输入转化为适合专家层处理的通用特征（如文本的词嵌入、图像的低级视觉特征）。
> - 微调的核心目标是让模型适配 “新任务的特定模式”，若冻结非专家层，通用特征无法根据新任务调整，会导致 MoE 层的专家们 “拿到的原始特征不匹配新任务”—— 即便专家层能学习，也难以适配新任务需求，最终性能暴跌。
> - 文中提到 “这符合预期”，正是因为 MoE 层虽占网络主要部分，但它依赖非专家层提供的 “任务适配型特征”，二者是 “基础 - 分工” 的依赖关系，缺一不可。
>
> #### 情况 2：仅冻结 MoE 层 → 效果接近全量微调
>
> - MoE 层是模型的 “核心能力载体”：大任务预训练后，MoE 层的专家们已具备多样化的 “子任务处理能力”（如有的专家擅长逻辑推理，有的擅长情感分析），这些能力具有一定通用性，可迁移到相似新任务中。
> - 微调时，仅更新非专家层，本质是让 “基础特征处理器” 针对新任务优化 —— 将输入转化为 “能更好匹配 MoE 层现有专家能力” 的特征。此时 MoE 层无需调整，就能通过适配后的特征发挥原有分工优势，自然效果接近全量微调。
> - 更关键的是，MoE 层参数占比极高（通常是模型参数的主要部分），冻结后无需更新大量专家参数，能大幅减少计算量（加速微调）和参数存储（降低显存需求），完美平衡效果与效率。
>
> ##### 3. 这一微调策略的本质：抓住 MoE 的 “能力迁移特性”
>
> MoE 模型的微调逻辑和普通 dense 模型完全不同：
>
> - 普通 dense 模型微调，需要更新所有层以让整体网络适配新任务；
> - MoE 模型微调，==因 “专家层已具备通用子任务能力”，只需调整 “特征输入端”（非专家层），就能让现有专家能力适配新任务== —— 这是一种 “借力现有核心能力，优化输入适配” 的高效策略。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/06_superglue_curves.png" alt="Fine-tuning learning curves">
  <figcaption>在小任务 (左图) 中，我们可以看到明显的过拟合，因为稀疏模型在验证集中的表现要差得多。在较大的任务 (右图) 中，MoE 则表现良好。该图来自 ST-MoE 论文</figcaption>
</figure>


**一种可行的微调策略是尝试冻结所有非专家层的权重。**实践中，这会导致性能大幅下降，但这符合我们的预期，因为混合专家模型 (MoE) 层占据了网络的主要部分。我们可以尝试相反的方法: 仅冻结 MoE 层的参数。实验结果显示，这种方法几乎与更新所有参数的效果相当。这种做法可以加速微调过程，并降低显存需求。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/07_superglue_bars.png" alt="Only updating the non MoE layers works well in fine-tuning">
  <figcaption>通过仅冻结 MoE 层，我们可以在保持质量的同时加快训练速度。该图来自 ST-MoE 论文</figcaption>
</figure>

在微调稀疏混合专家模型 (MoE) 时需要考虑的最后一个问题是，它们有特别的微调超参数设置——例如，==稀疏模型往往更适合使用较小的批量大小和较高的学习率，这样可以获得更好的训练效果。==

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/08_superglue_dense_vs_sparse.png" alt="Table comparing fine-tuning batch size and learning rate between dense and sparse models.">
  <figcaption>提高学习率和调小批量可以提升稀疏模型微调质量。该图来自 ST-MoE 论文</figcaption>
</figure>
**提高学习率和调小批量可以提升稀疏模型微调质量。**

此时，您可能会对人们微调 MoE 中遇到的这些挑战而感到沮丧，但最近的一篇论文 [《MoEs Meets Instruction Tuning》](https://arxiv.org/pdf/2305.14705.pdf) (2023 年 7 月) 带来了令人兴奋的发现。这篇论文进行了以下实验:

- 单任务微调
- 多任务指令微调
- 多任务指令微调后接单任务微调

当研究者们对 MoE 和对应性能相当的 T5 模型进行微调时，他们发现 T5 的对应模型表现更为出色。然而，当研究者们对 Flan T5 (一种 T5 的指令优化版本) 的 MoE 版本进行微调时，MoE 的性能显著提升。更值得注意的是，Flan-MoE 相比原始 MoE 的性能提升幅度超过了 Flan T5 相对于原始 T5 的提升，这意味着 MoE 模型可能从指令式微调中获益更多，甚至超过了稠密模型。此外，MoE 在多任务学习中表现更佳。与之前关闭 **辅助损失** 函数的做法相反，实际上这种损失函数可以帮助防止过拟合。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/09_fine_tune_evals.png" alt="MoEs benefit even more from instruct tuning than dense models">
  <figcaption>与稠密模型相比，稀疏模型从指令微调中受益更多。该图来自 MoEs Meets instructions Tuning 论文</figcaption>
</figure>

## 稀疏 VS 稠密，如何选择?

==稀疏混合专家模型 (MoE) 适用于拥有多台机器且要求高吞吐量的场景。在固定的预训练计算资源下，稀疏模型往往能够实现更优的效果。相反，在显存较少且吞吐量要求不高的场景，稠密模型则是更合适的选择。==

**注意**: 直接比较稀疏模型和稠密模型的参数数量是不恰当的，因为这两类模型基于的概念和参数量的计算方法完全不同。

## 让 MoE 起飞

最初的混合专家模型 (MoE) 设计采用了分支结构，这导致了计算效率低下。这种低效主要是因为 GPU 并不是为处理这种结构而设计的，而且由于设备间需要传递数据，网络带宽常常成为性能瓶颈。在接下来的讨论中，我们会讨论一些现有的研究成果，旨在使这些模型在预训练和推理阶段更加高效和实用。我们来看看如何优化 MoE 模型，让 MoE 起飞。

### 并行计算

让我们简要回顾一下并行计算的几种形式:

- **数据并行**: 相同的权重在所有节点上复制，数据在节点之间分割。
- **模型并行**: 模型在节点之间分割，相同的数据在所有节点上复制。
- **模型和数据并行**: 我们可以在节点之间同时分割模型和数据。注意，不同的节点处理不同批次的数据。
- **专家并行**: 专家被放置在不同的节点上。如果与数据并行结合，每个节点拥有不同的专家，数据在所有节点之间分割。

在专家并行中，专家被放置在不同的节点上，每个节点处理不同批次的训练样本。对于非 MoE 层，专家并行的行为与数据并行相同。对于 MoE 层，序列中的令牌被发送到拥有所需专家的节点。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/10_parallelism.png" alt="Image illustrating model, expert, and data prallelism">
  <figcaption>Switch Transformers 论文中展示如何使用不同的并行技术在节点上分割数据和模型的插图</figcaption>
</figure>

### 容量因子和通信开销

==提高容量因子 (Capacity Factor, CF) 可以增强模型的性能，但这也意味着更高的通信成本和对保存激活值的显存的需求。==在设备通信带宽有限的情况下，选择较小的容量因子可能是更佳的策略。**一个合理的初始设置是采用 Top-2 路由、1.25 的容量因子，同时每个节点配置一个专家。**在评估性能时，应根据需要调整容量因子，以在设备间的通信成本和计算成本之间找到一个平衡点。

### 部署技术

> 您可以在 `Inference Endpoints` 部署 [mistralai/Mixtral-8x7B-Instruct-v0.1](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMixtral-8x7B-Instruct-v0.1&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000)。

部署混合专家模型 (MoE) 的一个关键挑战是其庞大的参数规模。对于本地使用情况，我们可能希望使用更小的模型。为了使模型更适合部署，下面是几种有用的技术:

- **预先蒸馏实验**: Switch Transformers 的研究者们进行了预先蒸馏的实验。**他们通过将 MoE 模型蒸馏回其对应的稠密模型，成功保留了 30-40%的由稀疏性带来的性能提升。**预先蒸馏不仅加快了预训练速度，还使得在推理中使用更小型的模型成为可能。
- 任务级别路由: **最新的方法中，路由器被修改为将整个句子或任务直接路由到一个专家。**这样做可以提取出一个用于服务的子网络，有助于简化模型的结构。
- 专家网络聚合: 这项技术通过合并各个专家的权重，在推理时减少了所需的参数数量。这样可以在不显著牺牲性能的情况下降低模型的复杂度。

### 高效训练

FasterMoE (2022 年 3 月) 深入分析了 MoE 在不同并行策略下的理论性能极限，并且探索了一系列创新技术，**包括用于专家权重调整的方法、减少延迟的细粒度通信调度技术，以及一个基于最低延迟进行专家选择的拓扑感知门控机制。**这些技术的结合使得 MoE 运行速度提升高达 17 倍。

Megablocks (2022 年 11 月) 则专注于通过开发新的 GPU kernel 来处理 MoE 模型中的动态性，以实现更高效的稀疏预训练。其核心优势在于，它不会丢弃任何令牌，并能高效地适应现代硬件架构 (支持块稀疏矩阵乘)，从而达到显著的加速效果。Megablocks 的创新之处在于，它不像传统 MoE 那样使用批量矩阵乘法 (这通常假设所有专家形状相同且处理相同数量的令牌)，而是将 MoE 层表示为块稀疏操作，可以灵活适应不均衡的令牌分配。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png" alt="Matrix multiplication optimized for block-sparse operations.">
  <figcaption>针对不同规模的专家和令牌数量的块稀疏矩阵乘法。该图来自 [MegaBlocks](https://arxiv.org/abs/2211.15841) 论文</figcaption>
</figure>

## 开源混合专家模型

目前，下面这些开源项目可以用于训练混合专家模型 (MoE):

- Megablocks: <https://github.com/stanford-futuredata/megablocks>
- Fairseq: <https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm>
- OpenMoE: <https://github.com/XueFuzhao/OpenMoE>

对于开源的混合专家模型 (MoE)，你可以关注下面这些:

- [Switch Transformers (Google)](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f): 基于 T5 的 MoE 集合，专家数量从 8 名到 2048 名。最大的模型有 1.6 万亿个参数。
- [NLLB MoE (Meta)](https://huggingface.co/facebook/nllb-moe-54b): NLLB 翻译模型的一个 MoE 变体。
- [OpenMoE](https://huggingface.co/fuzhao): 社区对基于 Llama 的模型的 MoE 尝试。
- [Mixtral 8x7B (Mistral)](https://huggingface.co/mistralai): 一个性能超越了 Llama 2 70B 的高质量混合专家模型，并且具有更快的推理速度。此外，还发布了一个经过指令微调的模型。有关更多信息，可以在 Mistral 的 [公告博客文章](https://mistral.ai/news/mixtral-of-experts/) 中了解。

## 一些有趣的研究方向

首先是==尝试将稀疏混合专家模型 (SMoE) **蒸馏** 回到具有更少实际参数但相似等价参数量的稠密模型。==

MoE 的 **量化** 也是一个有趣的研究领域。例如，[QMoE](https://arxiv.org/abs/2310.16795) (2023 年 10 月) 通过将 MoE 量化到每个参数不到 1 位，将 1.6 万亿参数的 Switch Transformer 所需的存储从 3.2TB 压缩到仅 160GB。

简而言之，一些值得探索的有趣领域包括:

- 将 Mixtral 蒸馏成一个稠密模型。
- 探索合并专家模型的技术及其对推理时间的影响。
- 尝试对 Mixtral 进行极端量化的实验。

## 相关资源

- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- [Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Jun 2020)](https://arxiv.org/abs/2006.16668)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Dec 2021)](https://arxiv.org/abs/2112.06905)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Feb 2022)](https://arxiv.org/abs/2202.08906)
- [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models(April 2022)](https://dl.acm.org/doi/10.1145/3503221.3508418)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Nov 2022)](https://arxiv.org/abs/2211.15841)
- [Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (May 2023)](https://arxiv.org/abs/2305.14705)
- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

# 简单代码实现

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module): 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__() 
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x): 
        return self.fc(x) 
    
class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features): 
        super(MoELayer, self).__init__() 
        self.num_experts = num_experts 
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)]) 
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x): 
        # 步骤 1：门控权重计算
        gate_score = F.softmax(self.gate(x), dim=-1) # shape: (batch_size, num_experts)
        # 步骤 2：所有专家分别处理输入，得到各自的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # shape: (batch_size, num_experts, out_features)
        # 步骤3：将门控权重与专家输出加权融合
        # gate_score.unsqueeze(1)：给权重增加一个维度，shape 变为 (batch_size, 1, num_experts)（方便矩阵乘法）。
# torch.bmm：批量矩阵乘法，用门控权重（1×num_experts）与专家输出（num_experts×out_features）相乘，得到每个样本的融合结果（1×out_features）。
# squeeze(1)：去除多余的维度，最终输出 shape 为 (batch_size, out_features)。
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1) 
        return output 

input_size = 5
output_size = 3
num_experts = 4
batch_size = 10

model = MoELayer(num_experts, input_size, output_size)

demo = torch.randn(batch_size, input_size)

output = model(demo)

print(output.shape)  # 输出: torch.Size([10, 3])
```



## Citation

```bibtex
@misc {sanseviero2023moe,
    author       = { Omar Sanseviero and
                     Lewis Tunstall and
                     Philipp Schmid and
                     Sourab Mangrulkar and
                     Younes Belkada and
                     Pedro Cuenca
                   },
    title        = { Mixture of Experts Explained },
    year         = 2023,
    url          = { https://huggingface.co/blog/moe },
    publisher    = { Hugging Face Blog }
}
```

```
Sanseviero, et al., "Mixture of Experts Explained", Hugging Face Blog, 2023.
```
