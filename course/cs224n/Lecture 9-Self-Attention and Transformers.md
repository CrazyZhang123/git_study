## Lecture Plan

![image-20251030235024263](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235024263.png)

- 课程计划
  1. 从循环网络（RNN）到基于注意力的自然语言处理模型
  2. 介绍 Transformer 模型
  3. Transformer 模型的出色成果
  4. Transformer 模型的缺陷与变体
- 提醒事项：
  - 作业 4 周四截止！
  - 期中反馈调查于 2 月 16 日（周二）太平洋标准时间晚上 11:59 截止！
  - 期末项目提案于 2 月 16 日（周二）太平洋标准时间下午 4:30 截止！
  - 请尽量按时提交项目提案；我们希望尽快给你反馈！

![image-20251030235221980](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235221980.png)



- **As of last week: recurrent models for (most) NLP!**
  - Circa 2016, the de facto strategy in NLP is to **encode** sentences with a bidirectional LSTM: (for example, the source sentence in a translation)
  - Define your output (parse, sentence, summary) as a sequence, and use an LSTM to generate it.
  - Use attention to allow flexible access to memory



- **截至上周：循环模型主导（大部分）自然语言处理任务！**
  - 大约在2016年，自然语言处理领域的实际策略是用**双向LSTM对句子进行编码**：（例如，翻译任务中的源语句）
  - 将输出（句法分析、句子、摘要等）定义为序列，并用LSTM生成它。
  - 利用注意力机制实现对“记忆”的灵活访问



这张幻灯片回顾了**2016年前后循环神经网络（RNN/LSTM）在NLP领域的主导地位**，核心是三个技术环节：
1. **双向LSTM编码**：通过双向LSTM捕捉句子的上下文信息，为后续任务提供语义表示（如翻译任务中的源句编码）。
2. **LSTM序列生成**：将输出（如句法分析结果、生成的句子或摘要）建模为序列，用LSTM逐一生成。
3. **注意力机制辅助**：引入注意力机制，让模型能灵活“关注”输入序列的关键部分（即“灵活访问记忆”），提升长文本任务的表现（如翻译、摘要生成）。

这一套技术组合是Transformer出现前NLP的主流范式，为后续模型的发展奠定了基础。

![image-20251030235241884](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235241884.png)



- **Today: Same goals, different building blocks**
  - Last week, we learned about sequence-to-sequence problems and encoder-decoder models.
  - Today, we’re not trying to motivate entirely new ways of looking at problems (like Machine Translation)
  - Instead, we’re trying to find the best **building blocks** to plug into our models and enable broad progress.
  - 图示说明：2014-2017ish Recurrence → Lots of trial and error → 2021 ??????



- **今日主题：目标不变，构建模块换新**
  - 上周，我们学习了序列到序列问题和编码器-解码器模型。
  - 今天，我们并非要提出看待问题（如机器翻译）的全新视角
  - 而是试图找到最适合的**构建模块**，将其融入我们的模型以推动广泛进步。
  - 图示说明：2014-2017年左右的循环结构 → 大量试错 → 2021年的未知新结构



这张幻灯片是**自然语言处理模型技术演进的过渡说明**：
- 核心逻辑是“任务目标（如序列到序列的翻译、生成）不变，但实现这些目标的‘基础组件’在迭代升级”。
- 2014-2017年以循环神经网络（RNN/LSTM）为核心构建模块，经过大量试错后，2021年前后逐渐被Transformer等新构建模块取代。
- 课程通过这张幻灯片引出后续重点——介绍Transformer模型，它作为新一代“构建模块”，彻底改变了NLP的技术格局。

![image-20251030235427813](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235427813.png)



- **Issues with recurrent models: Linear interaction distance**
  - RNNs are unrolled “left-to-right”.
  - This encodes linear locality: a useful heuristic!
    - Nearby words often affect each other’s meanings
  - Problem: RNNs take O(sequence length) steps for distant word pairs to interact.



- **循环模型的问题：线性交互距离**
  - 循环神经网络（RNN）是“从左到右”展开计算的。
  - 这编码了线性局部性：是一种有用的启发式逻辑！
    - 相邻的单词通常会相互影响语义
  - 问题：对于距离较远的单词对，RNN需要花费与序列长度成正比（O(sequence length)）的步骤才能实现交互。



这张幻灯片剖析了**循环神经网络（RNN）的核心缺陷——长程依赖建模能力弱**：
- RNN按“从左到右”的顺序逐词计算，这种结构对“相邻单词”的语义交互很友好（比如“tasty”和“pizza”能快速交互）；
- 但对于“距离较远”的单词对（比如句子开头的“chef”和结尾的“was”），RNN需要一步步传递信息，耗时与序列长度成正比（O(sequence length)），导致长程语义依赖的捕捉效率极低，甚至丢失关键信息。

这一缺陷是后续Transformer模型（通过自注意力机制实现“全局交互”）崛起的重要动因。

![image-20251030235519490](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235519490.png)

- **Issues with recurrent models: Linear interaction distance**
  - O(sequence length) steps for distant word pairs to interact means:
    - Hard to learn long-distance dependencies (because gradient problems!)
    - Linear order of words is “baked in”; we already know linear order isn’t the right way to think about sentences…
  - 图示说明：Info of *chef* has gone through O(sequence length) many layers!



- **循环模型的问题：线性交互距离**
  - 距离较远的单词对需要花费与序列长度成正比的步骤才能交互，这意味着：
    - 难以学习长程依赖关系（因为梯度问题！）
    - 单词的线性顺序被“固化”了；但我们已经知道线性顺序并非理解句子的正确方式……
  - 图示说明：“chef”的信息要经过与序列长度成正比的多层传递！

这张幻灯片进一步阐述了循环神经网络（RNN）“线性交互距离”缺陷的**具体危害**：
1. **长程依赖学习困难**：由于远距离单词交互需要O(序列长度)的步骤，梯度在反向传播时会快速衰减（梯度消失/爆炸问题），导致模型难以学习句子中的长程语义依赖（比如“chef”和远处的“was”之间的语法、语义关联）。
2. **固化线性顺序偏见**：RNN的“从左到右”计算方式强行把单词的线性顺序作为语义交互的唯一逻辑，但实际上句子的语义依赖是网状的（比如“主语-谓语”可能跨越多词），线性顺序并非理解句子的最优方式。

这些缺陷直接推动了后续Transformer模型的诞生——它通过自注意力机制实现“全局并行交互”，彻底解决了RNN的长程依赖和线性顺序局限。

![image-20251030235608847](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235608847.png)



- **Issues with recurrent models: Lack of parallelizability**
  - Forward and backward passes have O(sequence length) unparallelizable operations
    - GPUs can perform a bunch of independent computations at once!
    - But future RNN hidden states can’t be computed in full before past RNN hidden states have been computed
    - Inhibits training on very large datasets!
  - 图示说明：Numbers indicate min # of steps before a state can be computed



- **循环模型的问题：缺乏并行性**
  - 前向和后向传播存在与序列长度成正比（O(sequence length)）的不可并行操作
    - GPU可以同时执行大量独立计算！
    - 但RNN的后续隐藏状态必须等前面的隐藏状态计算完成后才能完全计算
    - 阻碍了在超大数据集上的训练！
  - 图示说明：数字表示某个状态可计算前的最少步骤数



这张幻灯片聚焦循环神经网络（RNN）的**并行性缺陷**：
RNN的计算是“顺序依赖”的——第 \( t \) 步的隐藏状态必须等第 \( t-1 \) 步计算完成后才能进行（如图示中 \( h_2 \) 要等 \( h_1 \) 计算完，\( h_T \) 要等 \( h_{T-1} \) 计算完）。这种特性导致其前向、后向传播中存在大量“不可并行”的操作，而GPU的优势是并行处理独立计算，因此RNN无法充分利用GPU算力，训练超大数据集时效率极低。

这一缺陷也是Transformer模型的重要改进点——Transformer基于自注意力机制，所有位置的计算是**并行独立**的，能充分发挥GPU的并行优势，大幅提升训练效率。

![image-20251030235644625](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235644625.png)



- **If not recurrence, then what? How about word windows?**
  - Word window models aggregate local contexts
    - (Also known as 1D convolution; we’ll go over this in depth later!)
    - Number of unparallelizable operations does not increase sequence length!
  - 图示说明：Numbers indicate min # of steps before a state can be computed



- **不用循环结构的话，用什么呢？试试词窗口模型如何？**
  - 词窗口模型聚合局部上下文
    - （也称为一维卷积；我们之后会深入讲解！）
    - 不可并行操作的数量不会随序列长度增加而增加！
  - 图示说明：数字表示某个状态可计算前的最少步骤数



这张幻灯片提出了**循环结构的替代方案——词窗口模型（一维卷积）**，核心是解决RNN的并行性缺陷：
词窗口模型通过“固定大小的窗口”来聚合局部上下文（比如每次只关注相邻几个词的语义），这种结构的计算是**局部并行**的——不同窗口的计算可以同时进行，不会像RNN那样必须顺序依赖前一个状态。因此，不可并行操作的数量不会随序列长度增长，能更好地利用GPU的并行算力。

不过，词窗口模型的局限在于**只能捕捉局部依赖**，长程语义关联仍需多层叠加才能实现，这也为后续Transformer模型（通过自注意力实现“全局并行关联”）的出现埋下了伏笔。

![image-20251030235733407](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235733407.png)



- **If not recurrence, then what? How about word windows?**
  - Word window models aggregate local contexts
  - What about long-distance dependencies?
    - Stacking word window layers allows interaction between farther words
  - Maximum Interaction distance = sequence length / window size
    - (But if your sequences are too long, you’ll just ignore long-distance context)
  - 图示说明：Red states indicate those “visible” to \( h_k \)；Too far from \( h_k \) to be considered



- **不用循环结构的话，用什么呢？试试词窗口模型如何？**
  - 词窗口模型聚合局部上下文
  - 那长程依赖怎么办？
    - 堆叠词窗口层可以让距离较远的单词实现交互
  - 最大交互距离 = 序列长度 / 窗口大小
    - （但如果序列太长，你就会忽略长程上下文）
  - 图示说明：红色状态表示对 \( h_k \)“可见”；距离 \( h_k \) 太远，无法被考虑



这张幻灯片剖析了**词窗口模型的长程依赖缺陷**：
词窗口模型通过“局部窗口”聚合语义，虽然解决了并行性问题，但天然只能捕捉局部依赖。为了处理长程依赖，它需要**堆叠多层窗口**——每堆叠一层，交互距离扩大一个窗口大小。但这种方式的“最大交互距离”是有限的（等于“序列长度/窗口大小”），如果序列过长，仍会丢失远距离的语义关联。

这一局限进一步说明：仅靠“局部聚合+层堆叠”无法彻底解决长程依赖问题，从而为**Transformer模型（通过自注意力实现“全局无距离限制的语义交互”）**的必要性提供了论证。

> 好的，我们来举一个完整的例子，使用一个**虚构的、基于词窗口堆叠的深度模型**（即在 $\text{Transformer}$ 出现之前的深度 $\text{RNN}$ 或 $\text{CNN}$ 时代，试图用局部窗口解决长程依赖的方法），逐步分析随着层数堆叠，每个单词能够**间接**“看到”的**最大距离**。
>
> 
>
> ### 假设条件
>
> 
>
> 1. **模型类型：** 假设这是一个由 4 层堆叠而成、每层都使用局部窗口聚合信息的模型。
> 2. **序列长度 ($\text{N}$):** 假设序列有 15 个单词。
> 3. **窗口大小 ($\text{K}$):** 假设每一层的**上下文窗口半径 $K=2$**。
>    - 这意味着在某一层，一个词 $w_i$ 的输出 $h_i$ 是由**其自身**和其左右 $K=2$ 个词 $h_{i-2}, h_{i-1}, h_i, h_{i+1}, h_{i+2}$ 的输入聚合计算而来的。
> 4. **目标：** 我们关注**最中心词 $w_8$** 能够“看到”的单词范围。
>
> 
>
> ### 序列示例
>
> 
>
> 假设我们的输入序列是：
>
> 
>
> $$\underbrace{w_1, w_2, w_3, w_4}_{\text{远端}}, w_5, w_6, w_7, \mathbf{w_8}, w_9, w_{10}, w_{11}, \underbrace{w_{12}, w_{13}, w_{14}, w_{15}}_{\text{远端}}$$
>
> 
>
> ### 逐步分析 (堆叠 4 层)
>
> 
>
> 我们用 $R_L$ 表示在第 $L$ 层，中心词 $w_8$ **直接或间接**能覆盖到的输入词的索引范围。
>
> 
>
> #### 第 1 层 (Layer 1)
>
> 
>
> - **输入：** 原始词 $w_i$。
> - **计算：** $w_8$ 的新表示 $h_8^{(1)}$ 是由其左右 $K=2$ 范围内的原始词计算而来的。
> - **最大交互距离：** $K=2$。
> - **可见范围 $R_1$：** $[8-2, 8+2] = [6, 10]$。
> - **看到的单词：** $w_6, w_7, \mathbf{w_8}, w_9, w_{10}$。
> - **结论：** 仍然是纯粹的局部信息。
>
> 
>
> #### 第 2 层 (Layer 2)
>
> 
>
> - **输入：** 第 1 层的表示 $h_i^{(1)}$。
> - **计算：** $w_8$ 的新表示 $h_8^{(2)}$ 是由 $h_6^{(1)}$ 到 $h_{10}^{(1)}$ 计算而来的。
>   - 由于 $h_6^{(1)}$ 包含了 $w_4$ 到 $w_8$ 的信息（$R_1$ 向左延伸 $K=2$）。
>   - 由于 $h_{10}^{(1)}$ 包含了 $w_8$ 到 $w_{12}$ 的信息（$R_1$ 向右延伸 $K=2$）。
> - **最大交互距离：** $2 \times K = 4$。
> - **可见范围 $R_2$：** $[8 - (2 \times 2), 8 + (2 \times 2)] = [4, 12]$。
> - **看到的单词：** $w_4, w_5, w_6, w_7, \mathbf{w_8}, w_9, w_{10}, w_{11}, w_{12}$。
> - **结论：** 交互距离扩大了，但信息是间接传递的。
>
> 
>
> #### 第 3 层 (Layer 3)
>
> 
>
> - **输入：** 第 2 层的表示 $h_i^{(2)}$。
> - **计算：** $w_8$ 的新表示 $h_8^{(3)}$ 是由 $h_6^{(2)}$ 到 $h_{10}^{(2)}$ 计算而来的。
> - **最大交互距离：** $3 \times K = 6$。
> - **可见范围 $R_3$：** $[8 - (3 \times 2), 8 + (3 \times 2)] = [2, 14]$。
> - **看到的单词：** $w_2, \dots, w_{14}$。
> - **结论：** 已经覆盖了大部分序列，但仍有 $w_1$ 和 $w_{15}$ 无法触及。
>
> 
>
> #### 第 4 层 (Layer 4)
>
> 
>
> - **输入：** 第 3 层的表示 $h_i^{(3)}$。
> - **计算：** $w_8$ 的新表示 $h_8^{(4)}$ 是由 $h_6^{(3)}$ 到 $h_{10}^{(3)}$ 计算而来的。
> - **最大交互距离：** $4 \times K = 8$。
> - **可见范围 $R_4$：** $[8 - (4 \times 2), 8 + (4 \times 2)] = [0, 16]$。
>   - **实际范围：** 由于序列只有 $w_1$ 到 $w_{15}$，实际范围是 $[1, 15]$。
> - **看到的单词：** $\mathbf{w_1, \dots, w_{15}}$。
> - **结论：** 在第 4 层，中心词 $w_8$ **终于**间接地“看到”了序列中的所有单词。
>
> 
>
> ### 总结与 $\text{Transformer}$ 对比
>
> 
>
> | **特性**         | **词窗口堆叠模型 (K=2, L=4)**                         | **Transformer (自注意力)**                                   |
> | ---------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
> | **覆盖整个序列** | 需要 **4 层**。                                       | 仅需 **1 层** $\text{Attention}$ 即可。                      |
> | **信息路径**     | $w_1 \to h_3 \to h_5 \to h_7 \to h_8$（**间接且长**） | $w_1 \to \text{Attention}(w_8)$（**直接**）                  |
> | **层数需求**     | **深度依赖**。最大交互距离受限于深度 $L$。            | **深度独立**。所有交互在第 1 层就完成，堆叠层数只用于特征提炼。 |
>
> 这个例子清晰地展示了：要让词窗口模型覆盖 $N=15$ 的序列，**必须**堆叠到 $L=4$ 层，而如果序列长度 $N=1000$，这种方法将是灾难性的，因为信息必须通过上百个计算步骤才能从远端传到中心，这正是 $\text{Transformer}$ 旨在解决的**“长程依赖缺陷”**。

![image-20251030235759254](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235759254.png)



- **If not recurrence, then what? How about attention?**
  - Attention treats each word’s representation as a **query** to access and incorporate information from a set of **values**.
    - We saw attention from the **decoder** to the **encoder**; today we’ll think about attention **within a single sentence**.
  - Number of unparallelizable operations does not increase sequence length.
  - Maximum interaction distance: O(1), since all words interact at every layer!
  - 图示说明：All words attend to all words in previous layer; most arrows here are omitted



- **不用循环结构的话，用什么呢？试试注意力机制如何？**
  - 注意力机制将每个单词的表示视为**查询（query）**，用于访问并整合一组**值（values）**中的信息。
    - 我们之前见过解码器到编码器的注意力；今天我们来思考单句内的注意力。
  - 不可并行操作的数量不会随序列长度增加。
  - 最大交互距离：O(1)，因为所有单词在每一层都会相互交互！
  - 图示说明：所有单词都会关注前一层的所有单词；此处省略了大部分箭头



这张幻灯片引出了**注意力机制作为循环结构替代方案的核心优势**：
注意力机制将每个单词的表示作为“查询”，可以直接访问序列中所有其他单词的“值”，实现**全局语义交互**。这种特性带来两个关键改进：
1. **并行性友好**：不可并行操作的数量不随序列长度增长，能充分利用GPU并行算力；
2. **长程依赖无限制**：所有单词在每一层都能直接交互，最大交互距离是O(1)（即无距离限制），彻底解决了RNN和词窗口模型的长程依赖缺陷。

这为后续Transformer模型的“自注意力（Self-Attention）”核心模块奠定了理论基础，是NLP从“循环时代”迈向“注意力时代”的关键过渡。

![image-20251030235843549](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235843549.png)



- **Self-Attention**
  - Recall: Attention operates on **queries**, **keys**, and **values**.
    - We have some queries \( q_1, q_2, ..., q_T \). Each query is \( q_i \in \mathbb{R}^d \)
    - We have some keys \( k_1, k_2, ..., k_T \). Each key is \( k_i \in \mathbb{R}^d \)
    - We have some values \( v_1, v_2, ..., v_T \). Each value is \( v_i \in \mathbb{R}^d \)
  - In **self-attention**, the queries, keys, and values are drawn from the same source.
    - For example, if the output of the previous layer is \( x_1, ..., x_T \) (one vec per word), we could let \( v_i = k_i = q_i = x_i \) (that is, use the same vectors for all of them!)
  - The (dot product) self-attention operation is as follows:
    - \( e_{ij} = q_i^\top k_j \) （Compute key-query affinities）
    - \( \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})} \) （Compute attention weights from affinities (softmax)）
    - \( \text{output}_i = \sum_j \alpha_{ij} v_j \) （Compute outputs as weighted sum of values）
  - 图示说明：The number of queries can differ from the number of keys and values in practice.



- **自注意力**
  - 回顾：注意力机制基于**查询（queries）**、**键（keys）**和**值（values）**运作。
    - 我们有若干查询 \( q_1, q_2, ..., q_T \)，每个查询 \( q_i \in \mathbb{R}^d \)
    - 我们有若干键 \( k_1, k_2, ..., k_T \)，每个键 \( k_i \in \mathbb{R}^d \)
    - 我们有若干值 \( v_1, v_2, ..., v_T \)，每个值 \( v_i \in \mathbb{R}^d \)
  - 在**自注意力**中，查询、键和值来自同一数据源。
    - 例如，若前一层的输出是 \( x_1, ..., x_T \)（每个单词对应一个向量），我们可以令 \( v_i = k_i = q_i = x_i \)（即对三者使用相同的向量！）
  - 点积自注意力的操作流程如下：
    - \( e_{ij} = q_i^\top k_j \) （计算键-查询的相似度）
    - \( \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})} \) （通过相似度计算注意力权重（softmax））
    - \( \text{output}_i = \sum_j \alpha_{ij} v_j \) （通过值的加权和计算输出）
  - 图示说明：实际应用中，查询的数量可以与键和值的数量不同。



这张幻灯片详细定义了**自注意力（Self-Attention）**的核心逻辑，它是Transformer模型的灵魂组件：
- 自注意力的本质是“**同一序列内部的注意力交互**”——查询、键、值都来自同一个输入序列（比如一个句子的单词表示）。
- 计算流程分为三步：先通过点积计算“查询-键”的相似度（\( e_{ij} \)），再用softmax将相似度转化为注意力权重（\( \alpha_{ij} \)），最后通过权重对“值”进行加权求和得到输出。
- 这种机制让序列中**任意两个单词都能直接交互**（无需像RNN那样顺序传递），既解决了长程依赖问题，又能充分利用GPU并行计算，是Transformer超越循环模型的关键突破。

![image-20251030235935170](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251030235935170.png)



- **Self-attention as an NLP building block**
  - In the diagram at the right, we have stacked self-attention blocks, like we might stack LSTM layers.
  - Can self-attention be a drop-in replacement for recurrence?
  - No. It has a few issues, which we’ll go through.
  - First, self-attention is an operation on **sets**. It has no inherent notion of order.
  - 图示说明：Self-attention doesn’t know the order of its inputs.



- **作为自然语言处理构建模块的自注意力**
  - 在右侧的图示中，我们堆叠了自注意力块，就像我们可能堆叠LSTM层一样。
  - 自注意力能直接替代循环结构吗？
  - 不能。它存在一些问题，我们将逐一讲解。
  - 首先，自注意力是对**集合**的操作。它本身没有对顺序的固有认知。
  - 图示说明：自注意力不知道其输入的顺序。



这张幻灯片指出了**自注意力的核心缺陷——缺乏对输入顺序的感知**：
自注意力将输入视为“无顺序的集合”，只关注元素间的语义关联，却丢失了自然语言中至关重要的“顺序信息”（比如“厨师做食物”和“食物做厨师”语义完全不同，依赖于单词顺序）。这意味着如果直接用自注意力替代循环结构，模型会无法区分语序不同的句子，从而影响语义理解。

为了解决这个问题，Transformer后续引入了**位置编码（Positional Encoding）**，为每个位置的单词注入顺序信息，从而让自注意力既能捕捉全局语义关联，又能保留语序的重要性。