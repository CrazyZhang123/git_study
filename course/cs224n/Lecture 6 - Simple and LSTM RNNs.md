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