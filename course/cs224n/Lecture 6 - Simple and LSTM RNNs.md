## Lecture plan

1. RNN Language Models (25 mins)  **RNN 语言模型**（25 分钟 ）
2. Other uses of RNNs (8 mins)  **RNN 的其他应用**（8 分钟 ）
3. Exploding and vanishing gradients (15 mins) **梯度爆炸与梯度消失**（15 分钟 ）
4. LSTMs (20 mins)  **LSTM**（20 分钟 ）
5. Bidirectional and multi-layer RNNs (12 mins)  **双向与多层 RNN**（12 分钟 ）

Projects

- Next Thursday: a lecture about choosing final projects  下周四：讲解 “如何选择期末项目” 的讲座
- It’s fine to delay thinking about projects until next week 你可以推迟到下周再思考项目，没问题
- But if you’re already thinking about projects, you can view some info/inspiration on the website. It’s still last year’s information at present! 但如果你已经开始想项目了，可在课程网站查看灵感 / 信息（目前还是去年的内容 ）
- It’s great if you can line up your own mentor; we also lining up some mentors 如果你能自己联系导师更好；我们也会安排一些导师 ==line up ” 是**找到、安排、组织**的意思==

## Overview 课程概览
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250811175358599.png)
- Last lecture we learned:
    上节课内容：
    - Language models, n-gram language models, and Recurrent Neural Networks (RNNs)
    - **语言模型、n 元语法语言模型、循环神经网络（RNN ）**
- Today we’ll learn how to get RNNs to work for you
    本节课目标：学会让 RNN 为你所用
    - Training RNNs  **RNN 的训练方法**
    - Uses of RNNs  **RNN 的应用场景**
    - Problems with RNNs (exploding and vanishing gradients) and how to fix them **RNN 的问题（梯度爆炸与梯度消失 ）及解决方案**
    - These problems motivate a more sophisticated RNN architecture: LSTMs **这些问题催生更复杂的 RNN 架构：LSTM（长短期记忆网络 ）**
    - And other more complex RNN options: **bidirectional RNNs and multi-layer RNNs**
    - 其他**复杂 RNN 变体**：双向 RNN、多层 RNN
- Next lecture we’ll learn:
    下节课内容
    - How we can do Neural Machine Translation (NMT) using an RNN-based architecture called sequence-to-sequence with attention
    - ==如何用 “基于 RNN 的序列到序列（sequence-to-sequence ）+ 注意力机制” 架构，实现神经机器翻译（NMT ）==

## 1、The Simple RNN Language Model
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250811175948528.png)
每一个时间步都可以得到输出y

## Training an RNN Language Model RNN 语言模型的训练流程
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250811180152263.png)
- $J(\theta) = \frac{1}{T} \sum_{t=1}^T J^{(t)}(\theta) = \frac{1}{T} \sum_{t=1}^T -\log \hat{y}_{x_{t+1}}^{(t)}$
  
- Get a big corpus of text which is a sequence of words $x^{(1)}, ..., x^{(T)}$
    获取一个**大规模文本语料库**，形式为词序列 $x^{(1)}, ..., x^{(T)}$
- **Feed into** RNN-LM; compute output distribution $\hat{y}^{(t)}$ for every step t.
    将序列输入 RNN 语言模型（RNN-LM ），**在每个时间步 t 计算输出分布 $\hat{y}^{(t)}$**
    - i.e., predict probability dist of every word, given words so far
    - 即：==根据历史词，预测 “下一个词” 的概率分布（所有词的概率 ）==
- Loss function on step t is cross-entropy between predicted probability distribution $\hat{y}^{(t)}$, and the true next word $y^{(t)}$ (one-hot for $x^{(t+1)}$):
- 时间步 t 的**损失函数**：用 “预测分布 $\hat{y}^{(t)}$” 与 “真实下一个词 $y^{(t)}$（对 $x^{(t+1)}$ 做==独热编码==）” 的**交叉熵**计算：
- $J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w \in V} y_w^{(t)} \log \hat{y}_w^{(t)} = -\log \hat{y}_{x_{t+1}}^{(t)}$
  
- Average this to get overall loss for entire training set:
- **整体损失**：对所有时间步的损失取平均，得到整个训练集的损失
- $J(\theta) = \frac{1}{T} \sum_{t=1}^T J^{(t)}(\theta) = \frac{1}{T} \sum_{t=1}^T -\log \hat{y}_{x_{t+1}}^{(t)}$

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024005744584.png" alt="image-20251024005744584" style="zoom: 67%;" />

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024010250422.png" alt="image-20251024010250422" style="zoom:67%;" />

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024010316721.png" alt="image-20251024010316721" style="zoom:67%;" />

这里相当于滑动窗口来训练和预测，输入the后，应该输出students；输入students后期望输出opened，依次类推，通过teacher force确保模型训练的时候可以看到正确答案，然后计算预测概率和标准答案的交叉熵损失求和后，平均，作为总的loss fuction



<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024010954760.png" alt="image-20251024010954760" style="zoom:67%;" />

- **训练RNN语言模型**
- 然而：在整个语料库 $ x^{(1)},\ldots,x^{(T)}$ 上计算损失和梯度**太昂贵了**！
  $$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)$$
- 实际上，将 $ x^{(1)},\ldots,x^{(T)} $ 视为一个**句子（或一个文档）**。
- 回顾：**随机梯度下降**允许我们对小批量数据计算损失和梯度，然后进行更新。
- 对一个句子（实际上是一个batch的句子）计算损失 $J(\theta) $，计算梯度并更新权重。重复此过程。

这段内容围绕**RNN语言模型的训练优化**展开，核心是解决“全量数据计算成本过高”的问题，可从以下几点理解：
- **损失函数的定义**：公式 $ J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta) $ 表示整个语料库的损失是每个时间步（或每个样本片段）损失的平均值。若直接在“整个语料库”上计算，计算量会大到难以承受。
- **数据粒度的选择**：实际训练时，把语料拆分成“句子”或“文档”级别的单元，降低单次计算的规模。
- **随机梯度下降（SGD）的应用**：SGD的核心思想是**用小批量数据近似全量数据的梯度**，从而在保证收敛性的前提下，大幅降低计算成本。这里的“小批量”就是“一批句子”，通过“计算小批量损失→求梯度→更新权重→重复”的流程，实现模型的迭代优化。

简单来说，这段内容是在讲：**因为全量语料计算梯度太耗资源，所以RNN语言模型训练时会把语料拆成句子/文档的小批量，用随机梯度下降的方式分批更新模型权重。**





- **训练 RNN 的参数：RNN 的反向传播**
- 图中符号：$h^{(0)}$、$h^{(t-3)}$、$h^{(t-2)}$、$h^{(t-1)}$、$h^{(t)}$ 是不同时间步的隐藏状态；$W_h$ 是重复使用的权重矩阵；$J^{(t)}(\theta)$ 是第 t 步的损失函数。
- **问题**：损失函数 $J^{(t)}(\theta)$ 对重复的权重矩阵 $W_h$ 的导数是什么？ derivative：导数
- **答案**：$\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}$

- **梯度的累加性**：损失函数 $J^{(t)}(\theta)$ 对 $W_h$ 的梯度，是**该权重在每个时间步上产生的梯度的总和**。公式 $\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}$ 就体现了这一点 —— 把第 1 步到第 t 步中 $W_h$ 对损失的梯度逐个累加，得到最终的梯度用于权重更新。

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024170722573.png" alt="image-20251024170722573" style="zoom: 33%;" />

**“对重复权重的梯度，是该权重每次出现时对应的梯度的总和”**。

- 损失函数对这个共享权重的梯度，本质上是**该权重在每个时间步对损失的 “贡献梯度” 的累加**。根据链式法则，当一个参数在多个位置被使用时，它的总梯度就是各个位置上梯度的总和。

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024180637959.png)

- **RNN的反向传播：证明概要**
  - 给定多变量函数 $ f(x, y) $，以及两个单变量函数 $ x(t) $ 和 $ y(t) $，多变量链式法则表述为：
    $
    \underbrace{\frac{d}{dt} f(x(t), y(t))}_{\text{复合函数的导数}} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
    $
  - **在我们的例子中**：通过流程图展示损失函数 $ J^{(t)}(\theta) $ 与重复权重 $ W_h $ 的依赖关系（$ W_h $ 在多个时间步以 $ W_h|_{(1)}、W_h|_{(2)}、\dots、W_h|_{(t)} $ 形式出现，且这些形式都等于 $ W_h $）。
  - **应用多变量链式法则**：
    $
    \frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)} \cdot \underbrace{\frac{\partial W_h|_{(i)}}{\partial W_h}}_{=1} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}
    $
- **解读**
  这段内容通过**多变量链式法则**严格证明了RNN中“重复权重的梯度是各时间步梯度之和”的结论。**核心逻辑是**：==RNN的共享权重 $ W_h $ 在每个时间步都可视为一个“中间变量”，损失函数对 $ W_h $ 的总梯度，就是它在每个时间步上梯度的累加（因为 $ \frac{\partial W_h|_{(i)}}{\partial W_h} = 1 $，即每个时间步的 $ W_h|_{(i)} $ 与 $ W_h $ 本身完全等价）==。这一证明把RNN反向传播的梯度累加规则建立在微积分链式法则的理论基础上，让其合理性更具说服力。

### Backpropagation for RNNs

![image-20251024181619578](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024181619578.png)

- **RNN的反向传播**

  - 图中展示了RNN隐藏状态的时间序列（$ h^{(0)}, h^{(t-3)}, h^{(t-2)}, h^{(t-1)}, h^{(t)} $），以及共享权重 $ W_h $ 在各时间步的传递关系，损失函数 $ J^{(t)}(\theta) $ 依赖于 $ h^{(t)} $。
  - 公式 $ \frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)} $ 表示损失对共享权重的梯度是各时间步梯度的总和。
  - 问题“如何计算这个梯度？”的答案是：**沿时间步 $ i=t, \dots, 0 $ 反向传播，同时累加梯度**。这种算法被称为**“随时间反向传播（BPTT）”**，由Werbos于1988年在《Neural Networks》期刊中提出。

- **解读**
  随时间反向传播（BPTT）是RNN特有的反向传播算法，核心是利用“权重共享”的特性，将每个时间步的梯度累加，从而计算出损失对共享权重的总梯度。它通过“正向计算各时间步隐藏状态→反向沿时间步传递梯度并累加”的流程，实现了RNN的参数更新，是训练RNN类模型（如LSTM、GRU）的理论基础。

- ==在 RNN 的随时间反向传播（BPTT）中，共享权重 $W_h$ 是**一次性更新**的。==

  因为 $W_h$ 在所有时间步共享，我们需要先计算它在**所有时间步上的梯度总和**（即 $\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}$），然后基于这个**总梯度**对 $W_h$ 进行一次更新，而不是在每个时间步单独更新。

### Generating text with a RNN Language Model

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251024182515895.png)

- Just like a n-gram Language Model, you can use an RNN Language Model to generate text by repeated sampling. Sampled output becomes next step’s input.
  - 就像 **n 元语法语言模型**一样，**你可以通过重复采样的方式**，使用 RNN 语言模型生成文本。采样得到的输出会成为下一步的输入。

#### train in different text

![image-20251024182701376](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024182701376.png)

![image-20251024182830980](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024182830980.png)

### Evaluating Language Models

![image-20251024183243518](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024183243518.png)



- 语言模型的标准评估指标是困惑度。

- 困惑度公式：对$t=1$到T，计算语言模型预测下一个词概率的倒数的乘积，再开T次根

- Normalized by number of words 按词数归一化
- Inverse probability of corpus, according to Language Model 根据语言模型得到的语料逆概率
- This is equal to the exponential of the cross-entropy loss $J(\theta)$: 这等于交叉熵损失$J(\theta)$的指数：
- **Lower perplexity is better!**

#### RNNs have greatly improved perplexity

![image-20251024183534795](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024183534795.png)

![image-20251024183923440](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024183923440.png)

**我们为何要关注语言建模？**

- 语言建模是一项基准任务，有助于我们衡量在理解语言方面的进展。
- 语言建模是许多自然语言处理（NLP）任务的子组件，尤其是那些涉及文本生成或文本概率估计的任务：
  - 预测输入（打字预测）
  - 语音识别
  - 手写识别
  - 拼写 / 语法纠错
  - 作者身份识别
  - 机器翻译
  - 文本摘要
  - 对话系统
  - 等等

### Recap

回顾

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024184030530.png" alt="image-20251024184030530" style="zoom:60%;" />

- 语言模型：一个预测下一个单词的系统。
- 循环神经网络：一类神经网络，具有以下特点：
  - 接收任意长度的序列输入。
  - 每一步应用相同的权重。
  - 可以选择在每一步产生输出。
- **循环神经网络≠语言模型。**
- 我们已经说明，循环神经网络是构建语言模型的绝佳方式。
- 但循环神经网络的用途远不止于此！



### 2.Other RNN uses: RNNs can be used for sequence tagging

e.g. **part-of-speech tagging**, named entity recognitiom

词性标注、命名实体识别

![**image-20251024184237963**](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024184237963.png)

（下方示例中的单词与标签：）the → DT（限定词）startled → JJ（形容词）cat → NN（名词）knocked → VBN（过去分词）over → IN（介词）the → DT（限定词）vase → NN（名词）

- 这段内容展示了**循环神经网络（RNN）在序列标注任务中的应用**。序列标注是自然语言处理的基础任务之一，核心是给序列中的每个元素打上类别标签。

以词性标注为例，RNN 通过 “逐个处理单词、利用隐藏状态传递上下文信息” 的方式，为每个单词预测其词性（如 “the” 是限定词 DT、“cat” 是名词 NN 等）。这种能力也延伸到命名实体识别（识别文本中的人名、地名、机构名等）等任务中，体现了 RNN 处理序列数据、捕捉上下文依赖的优势。

#### RNNs can be used for sentence classification

循环神经网络可用于句子分类

- 例如：情感分类

![image-20251024184751570](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024184751570.png)

- **核心内容**：
  - 展示了用 RNN 进行句子分类的流程：对输入句子（如 “overall I enjoyed the movie a lot”）逐词处理，通过 RNN 的隐藏状态捕捉句子的整体语义，最终用**最终的隐藏状态**作为 “句子编码”，进而判断句子类别（如 “positive” 表示积极情感）。
  - 还提出了 “如何计算句子编码？” 的问题，给出**基础方法** “使用最终隐藏状态”。

![image-20251024184736784](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024184736784.png)

- 获取句子编码——**通常的办法**：逐元素的获取所有hidden states，求他们的最大值或者均值

  - 这里的 “element-wise max or mean of all hidden states” 指的是对所有时间步的隐藏状态 $h_t$，在**每个元素维度上**分别求最大值或均值。

  #### language encoder module

  ![image-20251024185351747](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024185351747.png)

- 标题

  ：循环神经网络可作为语言编码模块

  - 例如：问答系统、机器翻译及许多其他任务！

- 核心内容

  ：

  - 以 “贝多芬的国籍是什么？” 这一问答任务为例，展示 RNN 作为语言编码器的作用：RNN 先对问题（“what nationality was Beethoven?”）逐词处理，生成包含问题语义的编码；同时对上下文文本（介绍贝多芬的内容）进行编码，再结合两者的编码，通过神经网络架构输出答案（“German”）。

- 解读:

  在问答、机器翻译等复杂 NLP 任务中，RNN 的核心作用是 **“语言编码”**—— ==将输入的文本序列（问题、原文等）转换为蕴含语义的向量表示(类似上面的获取句子表示的方法，可以获取最后一个ht，也可以对所有的ht使用函数进行聚合)==。这种编码是后续任务（如生成答案、生成目标语言）的基础，体现了 RNN 在捕捉文本语义、支撑复杂语言任务方面的价值。



#### generate text

RNN 语言模型可用于文本生成

- 例如：语音识别、机器翻译、文本摘要

![image-20251024185713088](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024185713088.png)

- 核心内容

  - 以语音识别为例，展示 RNN 语言模型作为 “条件语言模型” 的应用：输入音频信号后，通过 “条件控制” 引导 RNN 语言模型生成文本（如 “what’s the weather”）。生成过程从`<START>`标记开始，每一步的输出作为下一步的输入，逐步生成完整文本。
  - 说明这是 “条件语言模型” 的示例，并提示下节课会详细讲解机器翻译。

- 解读

  RNN 语言模型不仅能自主生成文本，还能在 “外部条件（如音频、源语言文本）” 的引导下完成

  条件生成任务

  （如语音转文字、机器翻译、文本摘要）。这种 “条件生成” 能力让 RNN 语言模型成为诸多复杂 NLP 和跨模态任务的核心组件，体现了其在 “受控文本生成” 场景下的价值。



### 3、Problems with Vanishing and Exploding Gradients

#### Vanishing gradient intuition

梯度消失的直观理解

![image-20251024190257302](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024190257302.png)

- **标题**：梯度消失的直观理解
- **核心内容**：
  - 展示了RNN在反向传播时的梯度计算过程，以4个时间步的隐藏状态 $ h^{(1)}、h^{(2)}、h^{(3)}、h^{(4)} $ 为例，梯度 $ \frac{\partial J^{(4)}}{\partial h^{(1)}} $ 是多个梯度项的乘积（$ \frac{\partial h^{(2)}}{\partial h^{(1)}} \times \frac{\partial h^{(3)}}{\partial h^{(2)}} \times \frac{\partial h^{(4)}}{\partial h^{(3)}} \times \frac{\partial J^{(4)}}{\partial h^{(4)}} $）。
  - 指出**梯度消失问题**：当这些梯度项较小时，梯度信号会在反向传播过程中越来越小，导致距离输出远的早期时间步参数难以更新。
- **解读**：
  梯度消失是RNN的经典问题之一，==根源是反向传播时梯度的连乘效应。若激活函数（如tanh）的导数绝对值小于1，多次连乘后梯度会趋近于0，使得RNN难以学习长距离依赖关系（即早期输入对后期输出的影响无法有效传递）==。这一问题推动了LSTM、GRU等改进型RNN结构的出现，它们通过门控机制缓解了梯度消失，增强了长序列建模能力。

#### Vanishing gradient proof sketch

梯度消失证明概要（线性情形）

![image-20251024191112856](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024191112856.png)

### 翻译与解释

- **标题**：梯度消失证明概要（线性情形）
- **核心内容**：
  - **步骤1：简化假设**
    回顾RNN隐藏状态公式 $ h^{(t)} = \sigma \left( W_h h^{(t-1)} + W_x x^{(t)} + b_1 \right) $，假设激活函数 $ \sigma $ 为恒等函数（$ \sigma(x) = x $），则隐藏状态对前一隐藏状态的梯度为 $ \frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W_h $。
  - **步骤2：梯度的连乘效应**
    考虑损失 $ J^{(i)}(\theta) $ 对早期隐藏状态 $ h^{(j)} $ 的梯度（$ \ell = i - j $ 为时间步差），由链式法则得 $ \frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j < t \leq i} W_h = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^\ell $。若 $ W_h $ 是“小矩阵”，则当 $ \ell $ 增大时，该梯度会指数级衰减。
  - **步骤3：特征值分析**
    若 $ W_h $ 的所有特征值 $ \lambda_1, \lambda_2, \dots, \lambda_n < 1 $，则 $ W_h^\ell $ 可通过特征向量分解为 $ \sum_{i=1}^n c_i \lambda_i^\ell q_i $（$ q_i $ 为特征向量）。当 $ \ell $ 增大时，$ \lambda_i^\ell $ 趋近于0，导致梯度消失。
  - **步骤4：非线性激活的推广**
    对于tanh、sigmoid等非线性激活函数，本质问题与线性情形一致：激活函数的导数与权重矩阵的连乘效应，会使长距离的梯度信号指数级衰减。

- **解读**：
  这组内容从**线性情形**严格证明了RNN梯度消失的根源：权重矩阵的幂次（或特征值幂次）随时间步差 $ \ell $ 增长而指数级衰减，导致反向传播时早期时间步的梯度趋近于0。这一理论解释了RNN难以学习长距离依赖的本质原因，也为LSTM、GRU等通过门控机制“缓解梯度消失”的改进结构提供了理论支撑。

![image-20251024191145056](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024191145056.png)

### 翻译与解释

- **标题**：梯度消失证明概要（线性情形）
- **核心内容**：
  - **权重矩阵的幂次问题**：==探讨 $ W_h^\ell $（$ W_h $ 的 $ \ell $ 次幂）的缺陷，指出“特征值全小于1”是梯度消失的充分非必要条件==。
  - **特征值分析**：若 $ W_h $ 的特征值 $ \lambda_1, \lambda_2, \dots, \lambda_n < 1 $，将 $ \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^\ell $ 按特征向量分解后，得到 $ \sum_{i=1}^n c_i \lambda_i^\ell q_i $（$ q_i $ 为特征向量）。当 $ \ell $ 增大时，$ \lambda_i^\ell $ 趋近于0，导致梯度消失。
  - **非线性激活的推广**：对于tanh、sigmoid等非线性激活函数，问题本质与线性情形一致——激活函数的导数与权重矩阵的连乘效应，会使长距离梯度信号指数级衰减，仅需将条件调整为“特征值小于与激活函数和维度相关的 $ \gamma $”即可。

- **解读**：
  这段内容从**线性代数的特征值视角**，严格论证了RNN梯度消失的数学根源：当权重矩阵的特征值小于1时，其幂次会随时间步差 $ \ell $ 指数级衰减，导致反向传播时早期时间步的梯度趋近于0。这一理论不仅解释了RNN难以学习长距离依赖的本质，也为LSTM、GRU等通过门控机制“稳定梯度传播”的改进结构提供了理论依据。

#### Why is vanishing gradient problem?

![image-20251024191935443](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024191935443.png)

- 展示了 RNN 在反向传播时，远距离时间步的梯度信号（如 $h^{(1)}$ 到 $h^{(4)}$ 的梯度）会因衰减变得远小于近距离时间步的梯度信号（如 $h^{(2)}$ 到 $h^{(2)}$ 的梯度）。
- 说明梯度消失的后果：==远距离的梯度信号丢失，模型权重仅能根据近距离的影响更新，无法学习长时依赖关系。==
- 梯度消失会导致 **RNN “短视”**—— 只能捕捉相邻时间步的依赖，而对长距离的上下文关系（如一段文本中前后文的语义关联）无法有效学习。这会严重影响 RNN 在机器翻译、长文本理解等任务中的性能，也是 LSTM、GRU 等改进模型被提出的核心动因之一。

#### Effect of vanishing gradient on RNN-LM

梯度消失对 RNN 语言模型的影响

![image-20251024192343562](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024192343562.png)

- **语言模型任务示例**：在文本 “当她尝试打印门票时，发现打印机没墨了…… 最终打印了她的______” 中，模型需要学习第 7 步的 “tickets” 与结尾目标词 “tickets” 的长距离依赖。

- **梯度消失的后果**：由于梯度过小，RNN 语言模型无法学习这种长距离依赖，导致在测试时也无法预测类似的长距离语义关联。

- 解读

  这段内容通过具体的文本生成任务，直观展示了梯度消失对 RNN 语言模型的危害 ——

  **丧失长距离语义关联的学习能力。**在需要捕捉上下文长时依赖的场景（如长文本生成、连贯对话）中，这种缺陷会导致模型生成的文本逻辑断裂、语义不连贯，这也是后续 LSTM、Transformer 等模型在长序列建模任务中替代传统 RNN 的重要原因。



#### Why is exploding gradient a problem?

为何梯度爆炸是个问题？

![image-20251024192730243](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024192730243.png)

- 核心内容

  - **梯度爆炸对参数更新的影响**：==若梯度过大，随机梯度下降（SGD）的更新步长会变得过大==。参数更新公式为 $\theta^{new} = \theta^{old} - \alpha \nabla_\theta J(\theta)$（其中 $\alpha$ 是学习率，$\nabla_\theta J(\theta)$ 是梯度），大梯度会导致参数更新幅度过大。
  - **坏更新的后果**：参数更新步长过大会使模型进入损失很高的异常参数配置。**形象地说，就像 “本以为在爬一座小山，结果突然到了爱荷华州（完全偏离预期）”。**
  - **最严重的情况**：会导致网络中出现无穷大（Inf）或非数值（NaN）的情况，此时必须从之前的检查点重启训练，前功尽弃。

- 解读

  **梯度爆炸会破坏模型的训练稳定性** —— 要么使参数更新偏离优化方向，导致模型性能急剧下降；要么直接让训练崩溃。这也是深度学习中需要通过梯度裁剪、合理初始化、选择合适激活函数等方式来预防梯度爆炸的原因。

  

#### Gradient clipping : solution for exploding gradient

梯度裁剪：梯度爆炸的解决方案

![image-20251024193016275](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024193016275.png)

- 核心内容

  - **梯度裁剪定义**：若梯度的范数超过某个阈值，在应用随机梯度下降（SGD）更新前，将其按比例缩小。
  - **伪代码逻辑**：先计算梯度 $\hat{\mathbf{g}} = \frac{\partial \mathcal{E}}{\partial \theta}$；若其范数 $\|\hat{\mathbf{g}}\| \geq \text{threshold}$，则将梯度缩放为 $\hat{\mathbf{g}} \leftarrow \frac{\text{threshold}}{\|\hat{\mathbf{g}}\|} \hat{\mathbf{g}}$。
  - **直观理解**：在梯度方向不变的前提下，减小更新步长。
  - **实践价值**：梯度裁剪是解决梯度爆炸的简单有效方法，在实际训练中需注意应用。

- 解读

  **梯度裁剪通过 “限制梯度的最大范数”，避免了梯度爆炸导致的参数更新幅度过大甚至训练崩溃的问题。**它是深度学习中应对梯度爆炸的经典手段，能有效提升模型训练的稳定性，与梯度消失的解决方案（如 LSTM 的门控机制）形成互补，共同保障深度序列模型的训练效果。

#### How to fix the vanishing gradient problem?

如何解决梯度消失问题？

![image-20251024193531802](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024193531802.png)

- 核心内容

  - **梯度消失的核心问题：RNN 难以在多个时间步上学习保留信息。**
  - vanilla RNN 的缺陷：隐藏状态会被持续重写，公式为 $h^{(t)} = \sigma \left( W_h h^{(t-1)} + W_x x^{(t)} + b \right)$。
  - 解决思路：==提出 “带独立记忆的 RNN” 这一方向，为后续 LSTM、GRU 等具有门控记忆机制的模型做铺垫。==

- 解读

  这段内容指出了 **vanilla RNN 梯度消失的根源**是==“隐藏状态的持续重写导致信息难以长期保留”，并引出了 “独立记忆模块” 的解决方案思路==。这一思路直接推动了 LSTM（长短期记忆网络）和 GRU（门控循环单元）的诞生 —— 它们通过 “遗忘门、输入门、输出门” 或 “更新门、重置门” 的设计，实现了对信息的选择性保留与更新，从而有效缓解了梯度消失问题，大幅提升了 RNN 在长序列建模任务中的性能。

### Long Short-Term Memory RNNs(LSTMs)

长短期记忆循环神经网络（LSTM）

![image-20251024194131910](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024194131910.png)

- 核心内容

  - **提出背景与发展**：由 Hochreiter 和 Schmidhuber 于 1997 年提出，用于解决梯度消失问题；现代 LSTM 的关键部分源于 Gers 等人 2000 年的研究。
  - **核心结构**：在时间步t，包含隐藏状态$h^{(t)}$和细胞状态$c^{(t)}$（均为n维向量）。**细胞状态存储长期信息**，LSTM 可对其进行读取、擦除、写入操作，概念上类似计算机的 RAM。
  - **门控机制**：通过三个门（遗忘门、输入门、输出门）控制信息的擦除、写入、读取。==门是n维向量，每个元素可在 0（关闭）到 1（打开）之间动态变化，其值由当前上下文计算得出。==

- 解读

  LSTM 是针对传统 RNN 梯度消失问题的经典改进模型。它通过 “细胞状态 + 门控机制” 的设计，实现了对信息的长期存储与选择性更新 —— 遗忘门决定丢弃哪些历史信息，输入门决定纳入哪些新信息，输出门决定输出哪些信息。这种机制让 LSTM 能有效捕捉长距离依赖关系，在机器翻译、文本生成、语音识别等序列任务中表现出色，是深度学习序列建模领域的里程碑成果之一。



![image-20251024194746811](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024194746811.png)

### 翻译与解释

- **核心内容**：
  对于输入序列 $ x^{(t)} $，LSTM在时间步$ t $会计算隐藏状态 $ h^{(t)} $ 和细胞状态 $ c^{(t)} $，其核心流程由**三个门控和细胞状态更新**组成：
  - **遗忘门（$ f^{(t)} $）**：控制从之前的细胞状态中保留或遗忘的信息，通过sigmoid函数计算（$ f^{(t)} = \sigma \left( W_f h^{(t-1)} + U_f x^{(t)} + b_f \right) $），输出在0到1之间。
  - **输入门（$ i^{(t)} $）**：控制新细胞内容写入细胞的部分，计算方式为 $ i^{(t)} = \sigma \left( W_i h^{(t-1)} + U_i x^{(t)} + b_i \right) $。
  - **输出门（$ o^{(t)} $）**：控制细胞中输出到隐藏状态的部分，计算方式为 $ o^{(t)} = \sigma \left( W_o h^{(t-1)} + U_o x^{(t)} + b_o \right) $。
  - **新细胞内容（$ \tilde{c}^{(t)} $）**：待写入细胞的新内容，由tanh函数计算（$ \tilde{c}^{(t)} = \tanh \left( W_c h^{(t-1)} + U_c x^{(t)} + b_c \right) $）。
  - **细胞状态更新（$ c^{(t)} $）**：结合遗忘门和输入门，对细胞状态进行更新，公式为 $ c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)} $（$ \circ $ 表示按元素乘积）。
  - **隐藏状态更新（$ h^{(t)} $）**：由输出门控制从细胞状态中读取的内容，公式为 $ h^{(t)} = o^{(t)} \circ \tanh c^{(t)} $。

- **解读**：
  LSTM通过“遗忘-输入-输出”三个门的动态控制，实现了对细胞状态的**选择性更新与读取**，从而解决了传统RNN的梯度消失问题。遗忘门决定历史信息的留存，输入门决定新信息的纳入，输出门决定最终输出的内容，这种机制让LSTM能有效捕捉长距离依赖，在机器翻译、文本生成等序列任务中表现优异，是深度学习序列建模的核心工具之一。

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024195457369.png)

核心是更新ct的加号，可以让模型保存较远的信息

- LSTM 最核心的设计是 **“细胞状态 + 门控机制 + 加法操作”**。细胞状态是信息的 “长期存储器”，三个门（遗忘门、输入门、输出门）是信息的 “控制器”，==而加法操作则是保障梯度稳定传播的 “关键秘钥”。==这种组合让 LSTM 既能选择性地保留长时信息，又能避免梯度消失，从而高效处理长序列的依赖关系，成为序列建模任务中的经典解决方案。

#### How does LSTM solve vanishing gradients?

LSTM 如何解决梯度消失问题？

![image-20251024200049048](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251024200049048.png)

- **信息保留机制**：LSTM 的架构使 RNN 更容易在多个时间步上保留信息。例如，若某细胞维度的遗忘门设为 1、输入门设为 0，该细胞的信息可被无限期保留；而传统 RNN 难以学习能在隐藏状态中保留信息的循环权重矩阵 $W_h$。
- **实际效果**：在实践中，LSTM 能有效建模约 100 个时间步的依赖，远优于传统 RNN 的约 7 个时间步。
- **局限性说明**：LSTM 不保证完全消除梯度消失 / 爆炸，但为模型学习长距离依赖提供了更简便的方式。

### LSTMs: real-world success 

LSTM 的真实世界成功应用

![image-20251025005741216](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025005741216.png)

- **2013-2015 年的统治地位**：LSTM 在手写识别、语音识别、机器翻译、句法分析、图像描述以及语言模型等任务中取得了当时的最优结果，成为大多数自然语言处理（NLP）任务的主流方法。
- **2021 年的技术迭代**：Transformer 等新方法在许多任务中取代 LSTM 成为主流。以机器翻译领域的 WMT 会议为例，2014 年尚无神经机器翻译系统；2016 年 RNN（以 LSTM 为代表）主导并夺冠；2019 年 Transformer 的提及次数（105 次）远超 RNN（7 次）。

#### is vanishing/exploding gradient just a RNN problem?

梯度消失 / 爆炸只是 RNN 的问题吗？

![image-20251025010205031](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025010205031.png)

- - **问题的普遍性**：不是。它是所有神经网络架构（包括前馈网络、卷积网络）的问题，尤其是极深的网络。原因是链式法则和非线性函数的选择，导致梯度在反向传播时变得极小，使得底层网络学习非常缓慢（难以训练）。
  - **解决方案**：许多新的深度前馈 / 卷积架构通过增加更直接的连接（如残差连接）来让梯度更顺畅地流动。
  - **示例（残差网络 ResNet）**：残差连接（也叫跳跃连接）通过恒等映射默认保留信息，使深度网络更容易训练。其核心结构是 $\mathcal{F}(x) + x$，==让梯度能直接通过恒等路径传播，缓解了梯度消失问题==。
- **解读**：梯度消失 / 爆炸是深度学习的共性问题，并非 RNN 独有。在深度前馈或卷积网络中，随着层数增加，梯度的连乘效应同样会导致底层网络难以更新。残差网络通过 “shortcut 连接” 的设计，为梯度提供了直接的传播路径，成为解决深度网络梯度问题的经典方案，也为后续各种带残差结构的网络（如 DenseNet）提供了设计思路。

#### Bidirectional and Multi-layer RNNs: motivation

双向与多层 RNN：动机

![image-20251025011848535](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025011848535.png)

- **任务背景**：情感分类（判断文本情感是积极还是消极）
- **核心内容**：
  - **单向 RNN 的局限**：以句子 “the movie was terribly exciting!” 为例，单向 RNN 对 “terribly” 的上下文表示仅包含左侧语境（如 “the movie was”），但右侧语境（如 “exciting”）会改变 “terribly” 的语义（从消极转向积极），单向 RNN 无法捕捉这种右侧语境信息。
  - **双向 / 多层 RNN 的动机**：为了让模型同时捕捉**左右两侧的上下文信息**，从而更准确地理解单词在语境中的含义，提升情感分类等任务的性能。
- **解读**：单向 RNN 只能按序列顺序（如从左到右）建模上下文，导致对具有 “双向语义依赖” 的词汇（如本例中受右侧 “exciting” 影响的 “terribly”）理解不足。双向 RNN 通过同时运行 “从左到右” 和 “从右到左” 两个 RNN，融合两侧的语境信息，解决了这一局限；多层 RNN 则通过堆叠多个 RNN 层，让模型学习更抽象的语义表示。这种设计让 RNN 在情感分析、机器翻译等需要全局语境理解的任务中表现更优。

#### Bidirectional RNNs

![image-20251025012009808](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025012009808.png)

![image-20251025012337829](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025012337829.png)

- **核心内容**：
  在时间步$ t $，双向RNN的计算由**前向RNN**和**后向RNN**两部分组成：
  - 前向RNN：$ \overrightarrow{\boldsymbol{h}}^{(t)} = \text{RNN}_{\text{FW}}(\overrightarrow{\boldsymbol{h}}^{(t-1)}, \boldsymbol{x}^{(t)}) $，按序列顺序（如从左到右）计算隐藏状态。
  - 后向RNN：$ \overleftarrow{\boldsymbol{h}}^{(t)} = \text{RNN}_{\text{BW}}(\overleftarrow{\boldsymbol{h}}^{(t+1)}, \boldsymbol{x}^{(t)}) $，按序列逆序（如从右到左）计算隐藏状态。
  - 拼接隐藏状态：将前向和后向的隐藏状态拼接，得到双向RNN的隐藏状态 $ \boldsymbol{h}^{(t)} = [\overrightarrow{\boldsymbol{h}}^{(t)}; \overleftarrow{\boldsymbol{h}}^{(t)}] $，作为后续网络的输入。
  - 通用说明：这里的RNN可以是简单RNN、LSTM或GRU等，且前向和后向RNN的权重是独立的。

- **解读**：
  双向RNN通过“同时从两个方向建模序列”，让每个时间步的隐藏状态都能融合**左右两侧的上下文信息**，从而更准确地捕捉词汇的语境含义（如解决“terribly”因右侧“exciting”而语义反转的问题）。这种设计在情感分析、命名实体识别等需要全局语境理解的NLP任务中表现出色，是对单向RNN的重要升级。

对于时间步 $ t=6 $（对应单词“*!*”），后向RNN的计算过程为： $$ \overleftarrow{\boldsymbol{h}}^{(6)} = \text{RNN}_{\text{BW}}(\overleftarrow{\boldsymbol{h}}^{(7)}, \boldsymbol{x}^{(6)}) $$ 其中： 

- ==$ \overleftarrow{\boldsymbol{h}}^{(7)} $ 是时间步 $ t=7 $ 的后向隐藏状态，由于序列已到末尾，通常初始化为**全0向量**（或可学习的初始状态）；==
- $ \boldsymbol{x}^{(6)} $ 是单词“*!*”的词嵌入。 这一步是后向RNN的**起始计算**，为后续时间步（如 $ t=5 $ 的“*exciting*”）提供右侧语境的依赖信息。

![image-20251025013050491](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251025013050491.png)

- **双向RNN的适用条件**：
  - 仅当能获取**完整输入序列**时适用。
  - 不适用于语言建模任务，因为语言建模仅能利用左侧语境。
- **双向RNN的优势场景**：
  ==若能获取完整输入序列（如各类编码任务），双向结构非常强大，应默认使用。==
- **典型示例（BERT）**：
  BERT（基于Transformer的双向编码器表示）是一个强大的预训练上下文表示系统，其核心设计依赖双向性。后续会深入学习Transformer（包括BERT）的相关知识。

- **解读**：
  双向RNN的价值在于**融合序列的全局语境**，但受限于“需完整输入序列”的前提。==在语言建模这类“逐词生成、仅知左侧语境”的任务中无法应用，但在情感分析、命名实体识别等“需全局理解序列”的编码任务中表现出色==。BERT作为双向预训练模型的代表，进一步证明了双向结构在捕捉深层语境信息上的优势，成为NLP领域的里程碑技术之一。



