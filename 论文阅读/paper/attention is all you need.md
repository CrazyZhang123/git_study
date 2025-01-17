**Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time**, they generate a sequence of hidden states h t , as a function of the previous hidden state h t−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit **batching across examples**. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

**递归模型通常会将计算过程因子化为输入和输出序列的符号位置。将位置与计算时间步骤对齐**，它们会根据**前一个隐状态 $h_{t−1}$ 和位置 t的输入生成隐状态序列$h_{t}$。这种隐含的顺序性使得在训练示例中**无法实现并行化**，而在较长的序列长度下这一点变得至关重要，因为内存限制限制了跨示例的批处理(batching across examples)。最近的研究通过因子化技巧(factorization tricks)[21]和条件计算(conditional computation )[32]在计算效率方面取得了显著进展，同时在后者的情况下也改进了模型性能。**然而，顺序计算的基本限制仍然存在。

单词：

- in case of万一；如果发生……；若在……情况下
- preclude 排除；妨碍；阻止；



**Attention mechanisms** have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such **attention mechanisms** are used in conjunction with **a recurrent network.**

**注意机制**已经成为引人注目的序列建模和转换模型的重要组成部分，适用于各种任务，可以模拟输入或输出序列中的依赖关系，而不考虑它们的距离[2, 19]。然而，在除了一些特殊情况[27]之外，这种**注意机制**通常与**递归网络**一起使用。



- integral 完整的；不可或缺的；必需的；
- compelling   引人注目的；扣人心弦的；不可抗拒的；非常强烈的
- regard v.将…认为；看待；(尤指以某种方式)注视，凝视
  - n. 注意；关注；尊重；关心；
- In all but a few cases  在除了一些特殊情况之外
- in conjunction with   连同…;与…一起
  - conjunction 连词 (如and、but、or)；(引起某种结果的事物等的)结合；

In this work we propose the **Transformer,a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.** The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

在这项工作中，我们提出了**Transformer**，**这是一种模型架构，完全避免使用循环，并且完全依赖于注意机制来在输入和输出之间建立全局依赖关系。** Transformer可以**实现更高程度的并行化**，并且在仅在8个P100 GPU上训练12小时后，能够达到最新的翻译质量水平。

- architecture 结构；体系结构；建筑设计；
- eschew  避免；(有意地)避开；回避
- draw  提取；牵引；抽出;获取
- significantly  显著地；明显地；
- state of the art 最先进的



### **2 Background**

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing **hidden representations** in parallel for all input and output positions. In these models, **the number of operations** required to relate signals from two arbitrary input or output positions **grows** in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced **effective resolution** due to **averaging attention-weighted positions**, an effect we counteract with **Multi-Head Attention** as described in section 3.2.

减少顺序计算的目标也是Extended Neural GPU [16]、ByteNet [18]和ConvS2S [9]的基础，它们都使用卷积神经网络作为基本构建模块，对所有输入和输出位置并行计算**隐状态表示（hidden representations）**。*在这些模型中，将两个任意输入或输出位置之间的信号相关联所需的**操作次数**，随着位置之间的距离增加而**增加***，对于ConvS2S是线性增加，对于ByteNet是对数增加。这使得学习远距离位置之间的依赖关系更加困难[12]。在Transformer中，这种增长被减少为一定数量的操作，尽管通过**平均注意力加权位置**来降低了**有效分辨率**的代价，我们通过3.2节中描述的**多头注意力**来抵消这种影响。

- arbitrary 任意的；武断的；随心所欲的
- distant 遥远的；远处的；
- albeit 尽管；虽然
- effective resolution  有效分辨率
- counteract 抵消；抵抗；

> **the number of operations** required to relate signals from two arbitrary input or output positions **grows** in the distance between positions,
> 在这些模型中，将两个任意输入或输出位置之间的信号相关联所需的**操作次数**，随着位置之间的距离增加而**增加**

> we counteract with **Multi-Head Attention** as described in section
> 3.2. 我们通过3.2节中描述的**多头注意力**来抵消这种影响。



**Self-attention**, sometimes called **intra-attention** is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, **abstractive summarization, textual entailment** and learning task-independent sentence representations [4, 27, 28, 22].

**自注意力**，有时也称为**内部注意力**，是一种注意机制，用于关联单个序列的不同位置，以便计算序列的表示。自注意力已成功应用于各种任务，包括阅读理解、摘要生成、文本推理和学习与任务无关的句子表示[4, 27, 28, 22]。

- abstractive summarization, textual entailment 摘要总结，文本推理



End-to-end memory networks are based on a **recurrent attention mechanism** instead of **sequence aligned recurrence** and have been shown to perform well on simple-language question answering and language modeling tasks [34].

端到端记忆网络是基于**循环注意机制（recurrent attention mechanism）而不是序列对齐的循环（sequence aligned recurrence）**，并且已经显示出在简单语言问答和语言建模任务上表现良好。[34]

To the best of our knowledge, however, the Transformer is the first transduction model **relying entirely on self-attention** to compute representations of its input and output without using sequence aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

然而，据我们所知，Transformer 是**第一个完全依赖自注意力**来**计算其输入和输出表示（compute representations）**的转换模型，而不使用序列对齐的循环神经网络或卷积。在接下来的几节中，我们将描述Transformer，解释自注意力的动机，并讨论它相对于诸如[17，18]和[9]模型的优势。

### **3 Model Architecture**

Most competitive **neural sequence transduction models** have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x 1 , …, x n ) to a sequence of continuous representations z = ( z1 , …, z n ). Given z, the decoder then generates an output sequence (y 1 , …, y m ) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

大多数具有竞争力的**神经序列转导模型（neural sequence transduction models）**都采用编码器-解码器结构。在这个结构中，编码器将输入的表示序列 $(x_1，...，x_n)$映射到一系列连续的表示$z = (z_1，...，z_n)$。给定z，解码器则逐步生成一个输出符号序列$ (y1，...，ym)$。**在每一步中，模型是自回归的，即在生成下一个符号(symbols)时，它使用先前生成的符号作为额外的输入。**



- continuous representations  连续的表示

The Transformer follows this **overall architecture** using **stacked self-attention** and **point-wise fully connected layers** for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

Transformer模型的总体架构如下，它在编码器和解码器中都采用了**堆叠(overall architecture)的自注意力机制(stacked self-attention)和逐点全连接层(point-wise fully connected layers)**，如图1的左右两半所示。

- overall 总体

![img](https://i-blog.csdnimg.cn/blog_migrate/697264da844754bbda6a71f8ef182bad.png#pic_center)

Figure 1: The Transformer - model architecture.

### **3.1 Encoder and Decoder Stacks**

**Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, **position-wise** fully connected **feed-forward network**. We employ **a residual connection** [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function **implemented** by the sub-layer itself. To **facilitate** these residual connections, all sub-layers in the model, as well as **the embedding layers**, produce outputs of dimension dmodel = 512.

**编码器:** 编码器由N ( N = 6 ) 个相同的层组成。每个层包含两个子层。第一个子层是一个多头自注意机制，第二个子层是一个简单的**逐位置**全连接**前馈网络**。我们采用**残差连接**[11]将这两个子层围绕起来，然后进行层归一化[1]。也就是说，每个子层的输出是 LayerNorm(x + Sublayer(x))，其中 Sublayer(x) 是子层本身**实现**的函数。**为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出都有一个维度为dmodel=512。**



**LayerNorm(x + Sublayer(x)) 残差连接+样本归一化layernorm**

- identical layers 完全相同的；相同的

**Decoder:** The decoder is also composed of a stack of N = 6 identical layers. **In addition to** the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. **Similar to** the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to **prevent positions from attending to subsequent positions.** This masking, **combined with fact that the output embeddings are offset by one position**, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

**解码器：** 解码器也由N(N=6)个相同层次的堆叠组成。除了每个编码器层中的两个子层外，解码器还插入了第三个子层，它在编码器堆叠的输出上执行多头注意力机制。与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。我们还修改了解码器堆栈中的自注意力子层，**以防止位置关注后续的位置**。**这种掩码机制与输出嵌入偏移一个位置相结合**，确保位置i 的预测仅依赖于小于i 位置已知的输出。



**masked multi-head self-attention mechanism**: 只关注自己位置及以前的输入

### **3.2 Attention**

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight **assigned** to each value is computed by a **compatibility function** of the query with the **corresponding key.**

**注意力函数**可以被描述为**将一个查询和一组键值对映射到一个输出的函数**，其中**查询、键、值( query, keys, values)和输出都是向量。输出是通过对值的加权求和来计算的，其中对每个值赋予的权重是通过将查询与相应的键(corresponding key.)** 进行 **兼容性函数(compatibility function)** 计算得到的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/09c51c777076bbedf7e6be9b9315d8c0.png#pic_center)Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
图2：（左侧）缩放点积注意力。 （右侧)多头注意力由多个并行运行的注意力层组成。