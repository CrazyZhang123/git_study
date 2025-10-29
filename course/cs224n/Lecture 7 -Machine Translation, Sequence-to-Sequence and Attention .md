# Lecture Plan

![image-20251028202910985](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028202910985.png)

根据图片内容，翻译如下：

讲座计划

今天我们将：

1. 介绍一个新任务：机器翻译 [15分钟]，这是…的主要应用场景
2. 介绍一个新的神经网络架构：序列到序列 [45分钟]，通过…得到改进
3. 介绍一个新的神经网络技术：注意力机制 [20分钟]

通知

 • 作业3今天截止——希望你们的依存解析器正在解析文本！

 • 作业4今天发布——本讲座将涵盖相关内容，你们有9天时间完成（！），截止日期为周四

 • 尽早开始！它比之前的作业更大更难 😬

 • 周四的讲座关于选择最终项目



## section1: Pre-Neural Machine Translation

![image-20251028212626857](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028212626857.png)



**1990s-2010s：统计机器翻译**
• 核心思想：从数据中学习概率模型
• 假设我们正在将法语翻译成英语
• 我们想要找到给定法语句子x的最佳英语句子y
  $\text{argmax}_y P(y|x)$
• 使用贝叶斯规则将其分解为两个可以单独学习的组件：
  $= \text{argmax}_y P(x|y)P(y)$
**[翻译模型框]**
翻译模型
建模单词和短语应该如何翻译（忠实度）。从平行数据中学习。
**[语言模型框]**
语言模型

建模如何写出好的英语（流畅度）。从单语数据中学习。

----

##### **为什么使用贝叶斯会变成这样？**

这里的关键在于**工程实践和模型可学习性**的考虑：

**1. 原始问题的困难**

直接建模 $P(y|x)$（给定法语，生成英语的概率）非常困难，因为：
- 需要大量平行的法英对照句子
- 模型需要同时理解语言转换和目标语言的语法
- 数据需求量巨大，参数空间复杂

**2. 贝叶斯规则的优势**

使用贝叶斯规则 $P(y|x) = \frac{P(x|y)P(y)}{P(x)}$ 后：
- **分母 $P(x)$ 是常数**：对于固定的法语句子x，$P(x)$ 不变，所以在argmax中可以忽略
- **得到分解形式**：$\text{argmax}_y P(x|y)P(y)$

**3. 两个组件的工程优势**

**翻译模型 $P(x|y)$：**

- 建模"如果英语是y，法语是x的概率"
- 可以从相对较少的平行数据中学习词汇对应关系
- 专注于翻译的**准确性**（fidelity）
**语言模型 $P(y)$：**
- 建模"英语句子y本身有多自然"
- 可以从大量纯英语数据中学习（不需要翻译对照）
- 专注于语言的**流畅性**（fluency）

**4. 实际效果**

最终，系统会寻找既忠实于原文（高$P(x|y)$）又流畅自然（高$P(y)$）的翻译结果，这比直接学习 $P(y|x)$ 更加实用和有效。
==这种分解是统计机器翻译的核心创新，让机器翻译在1990s-2010s取得了巨大突破==。

![image-20251028213441947](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028213441947.png)

**问题：如何学习翻译模型 P(x∣y)\*P\*(\*x\*∣\*y\*)？**

• 首先，需要大量的**平行数据**（例如，人工翻译的法语/英语句子对）

**(图片下方标注)**
**罗塞塔石碑**
*古埃及文*
*世俗体*
*古希腊文*

### Learning alignment for SMT

![image-20251028213650985](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028213650985.png)

**问题：如何从平行语料库中学习翻译模型 P(x|y)？**

**进一步分解：在模型中引入潜变量 a：P(x,a|y)**

**其中 a 是对齐，即源句子 x 和目标句子 y 之间的词级对应关系**

**Morgen fliege ich nach Kanada zur Konferenz**
（明天我飞往加拿大参加会议）

**Tomorrow I will fly to the conference in Canada**
（明天我将飞往加拿大参加会议）

---

**实例说明**：

- 德语句子：“Morgen fliege ich nach Kanada zur Konferenz”
- 英语句子：“Tomorrow I will fly to the conference in Canada”

通过对齐，模型可以学习到：

- “Morgen” ↔ “Tomorrow”
- “fliege” ↔ “fly”
- “ich” ↔ “I”
- “Kanada” ↔ “Canada”
- “Konferenz” ↔ “conference”

这种词对齐是统计机器翻译的基础，它帮助模型理解不同语言之间的词汇对应关系，从而进行准确的翻译。

### What is alignment? 

![image-20251028214304639](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028214304639.png)

Alignment is the **correspondence between particular word**s in the translated sentence pair. 

**Typological differences** between languages lead to complicated alignments! 

Note: Some words have no counterpart

对齐是翻译句子对中特定词语之间的对应关系。

语言类型学差异导致复杂的对齐！

**注意：有些词没有对应词**

### Alignment is complex

对齐是复杂的

![image-20251028215006163](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028215006163.png)

![image-20251028215102975](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028215102975.png)

![image-20251028215113755](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028215113755.png)

对齐可以是**多对一、一对多、多对多**

### Learning alignment for SMT

**为统计机器翻译学习对齐**

![image-20251028215144145](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251028215144145.png)

我们将 P(x, a|y) 学习为多个因素的组合，包括：

- 特定词汇对齐的概率（也依赖于在句子中的位置）
- 特定词汇具有特定生育率的概率（对应词汇的数量）
- 等等

对齐 a 是潜在变量：它们在数据中没有被明确指定！

- 需要使用特殊的学习算法（如期望最大化算法）来学习包含潜在变量的分布参数
- 以前，我们过去在 CS 224N 中做很多这方面的内容，但现在请参见 CS 228！

### Decode for SMT

统计机器翻译的解码

![image-20251029102605960](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029102605960.png)



#### (1)翻译概率模型

统计机器翻译系统需要为每个可能的翻译分配概率：

- 有些翻译更常见（概率更高）
- 有些翻译只在特定上下文中出现（概率较低）
- 系统根据上下文选择最合适的翻译

#### (2)实际应用意义

这个例子说明了：

- **机器翻译的挑战**：需要处理大量歧义
- **统计方法的优势**：通过大数据学习翻译概率
- **上下文的重要性**：正确翻译需要考虑整个句子的语境
  - ==通过探索不同的翻译可能，然后修剪prune，最终翻译了整个输入句子，计算出了相当可能(fairly likely)概率的翻译结果==

这种基于概率的词汇翻译表是统计机器翻译系统的核心组件之一，系统会根据这些概率和上下文信息来选择最佳的翻译结果。



![image-20251029103848482](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029103848482.png)

**1990s-2010s：统计机器翻译**

统计机器翻译曾是一个庞大的研究领域

最佳系统极其复杂
• 我们这里没有提及数百个重要细节
• 系统包含许多独立设计的子组件
• 大量特征工程

  - 需要设计特征来捕捉特定语言现象
• 需要编译和维护额外资源
  - 例如等效短语表
• 需要大量人力维护
  - 每种语言对都需要重复努力！

## Section 2: Neural Machine Translation

![image-20251029103954253](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029103954253.png)

神经机器翻译对机器翻译研究产生了巨大的冲击！

### What is Neural Machine Transaltion?

什么是神经机器翻译

![image-20251029104107287](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029104107287.png)

- **神经机器翻译（NMT）**是一种使用单一端到端神经网络进行机器翻译的方法 
  - 但实际上我们需要构建巨大的神经网络来处理整个句子。
- 这种神经网络架构被称为序列到序列模型（又名seq2seq），它包含两个循环神经网络（RNN）
  - 实际不一定是RNN

### Neural Machine Translation (NMT)

神经机器翻译

![image-20251029111308197](/Users/zjj/Library/Application Support/typora-user-images/image-20251029111308197.png)

- The sequence-to-sequence model → 序列到序列模型
- 模块与流程说明
  - Encoding of the source sentence. Provides initial hidden state for Decoder RNN. → 源语句的编码。为解码器 RNN 提供初始隐藏状态。
  - Encoder RNN → 编码器 RNN
  - Source sentence (input) → 源语句（输入）
  - Encoder RNN produces an encoding of the source sentence. → 编码器 RNN 生成源语句的编码。
  - Target sentence (output) → 目标语句（输出）
  - Decoder RNN → 解码器 RNN
  - Decoder RNN is a Language Model that generates target sentence, conditioned on encoding. → 解码器 RNN 是一个语言模型，基于编码生成目标语句。
  - Note: This diagram shows test time behavior: decoder output is fed in as next step’s input → 注：此图展示测试时的行为：解码器输出被作为下一步的输入。
- 具体元素
  - 源语句单词：*il, a, m’ entarté*
  - 目标语句单词：*<START>, he, hit, me, with, a, pie, <END>*
  - 决策函数：*argmax*

  这张图展示了**神经机器翻译（NMT）中 “序列到序列（Seq2Seq）” 模型的核心架构**，由**编码器 RNN**和**解码器 RNN**两部分组成，是机器翻译从 “基于规则 / 统计” 转向 “基于深度学习” 的里程碑式方法。

  #### 1. 编码器 RNN（Encoder RNN）

  - 作用：对源语句（输入）进行编码，捕获其语义信息。

    示例中，法语源语句 *“il a m’ entarté”* 被逐个单词输入编码器 RNN，**最终生成一个固定长度的编码向量（图中橙色框内的隐藏状态），该向量包含了源语句的整体语义。**

  - **核心功能**：==将 “变长的源语句” 转换为 “定长的语义编码”，为解码器提供上下文==。

  #### 2. 解码器 RNN（Decoder RNN）

  - 作用：基于编码器的语义编码，生成目标语句（输出），本质上是一个 “条件语言模型”（生成目标语言时依赖源语句的编码）。

    示例中，以特殊标记<START>为起始，结合编码器的语义编码，逐词生成英语目标语句

    “he hit me with a pie <END>”。

  - **生成机制（测试时）**：采用 “自回归” 方式 —— 每一步生成的单词会作为**下一步的输入**（如图中粉色虚线所示，前一步输出的 *“he”* 是下一步的输入），直到生成 *<END>* 标记为止。

  - **决策过程**：每一步通过 *argmax* 函数选择概率最高的单词作为输出（实际训练时常用 “teacher forcing” 或波束搜索，测试时多为贪心搜索或波束搜索）。

#### Sequence-to-sequence is versatile! 

序列到序列模型用途广泛！

![image-20251029113224759](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029113224759.png)

- Sequence-to-sequence is useful for more than just MT → 序列到序列模型的用途不止于机器翻译（MT）

- Many NLP tasks can be phrased as sequence-to-sequence: → 许多自然语言处理（NLP）任务可被表述为序列到序列问题：

  - **Summarization** (long text → short text) → 文本摘要（长文本→短文本）
  - **Dialogue** (previous utterances → next utterance) → 对话生成（历史 utterance→下一个 utterance）
  - **Parsing** (input text → output parse as sequence) → 句法分析（输入文本→序列形式的句法分析结果）
  - **Code generation** (natural language → Python code) → 代码生成（自然语言→Python 代码）

  ---

  这张幻灯片旨在说明**序列到序列（Seq2Seq）模型的通用性**—— 它并非仅用于 “机器翻译（MT）” 这一单一任务，而是能适配多种自然语言处理场景，核心逻辑是 “将‘输入序列→输出序列’的映射关系抽象为统一框架”。

  #### 各任务的适配逻辑

  - **文本摘要**：输入是 “长文本序列”，输出是 “短摘要序列”，Seq2Seq 模型可学习 “信息压缩与提炼” 的映射。
  - **对话生成**：输入是 “历史对话序列（如用户的多轮提问）”，输出是 “下一轮回复序列”，Seq2Seq 可捕捉对话的上下文连贯性。
  - **句法分析**：输入是 “自然语言句子序列”，输出是 “句法树的序列化表示（如依存弧序列、短语结构序列）”，Seq2Seq 可学习语法结构的生成规则。
  - **代码生成**：输入是 “自然语言需求描述序列”，输出是 “Python 代码序列”，Seq2Seq 可桥接自然语言与编程语言的语义映射。

### Neural Machine Translation(NMT)

![image-20251029113807738](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029113807738.png)

- **标题**：Neural Machine Translation (NMT) → 神经机器翻译（NMT）
- **要点1**：The sequence-to-sequence model is an example of a Conditional Language Model → 序列到序列模型是**条件语言模型**的一个示例
  - Language Model because the decoder is predicting the next word of the target sentence *y* → **属于语言模型，因为解码器在预测目标语句 *y* 的下一个单词**
  - Conditional because its predictions are also conditioned on the source sentence *x* → **属于“条件”模型，因为其预测同时依赖于源语句 *x***
- **要点2**：**NMT directly calculates** $P(y|x)$:
  $$P(y|x) = P(y_1|x)\ P(y_2|y_1, x)\ P(y_3|y_1, y_2, x)\ \dots \underbrace{P(y_T|y_1, \dots, y_{T-1}, x)}_{\text{给定已生成的目标词和源语句 } x \text{ 时，下一个目标词的概率}}$$
- **问题与回答**：
  - **Question: How to train a NMT system?** → 问题：如何训练一个NMT系统？
  - Answer: Get a big parallel corpus... → 回答：获取一个大规模的平行语料库...


#### 解释
这张幻灯片从**概率建模和训练逻辑**角度，进一步阐释神经机器翻译（NMT）的本质：

#### 1. 条件语言模型的双重属性
序列到序列模型（Seq2Seq）之所以属于“条件语言模型”，是因为它同时具备两种特性：
- **语言模型属性**：解码器的核心任务是“逐词生成目标语句”，每一步都在预测“下一个单词的概率”（和传统语言模型一致）；
- **条件属性**：生成目标语句时，并非“无中生有”，而是**依赖于源语句的语义编码**（即条件是“源语句 *x*”）。

这种“条件+语言模型”的结合，让NMT能学习“源语句→目标语句”的概率分布 $P(y|x)$。


#### 2. 概率分解与生成逻辑
公式 $P(y|x) = P(y_1|x) \cdot P(y_2|y_1, x) \cdot \dots \cdot P(y_T|y_1, \dots, y_{T-1}, x)$ 展示了NMT的**概率分解方式**：
- 目标语句 *y* 被拆分为单词序列 $y_1, y_2, \dots, y_T$；
- 每个单词的生成概率都“依赖于前面生成的所有单词”和“源语句 *x*”。

这体现了NMT的**自回归生成特性**——==生成下一个单词时，会考虑“历史生成结果”和“源语句语义”，保证了翻译的连贯性。==


#### 3. 训练的核心依赖：平行语料库
要训练NMT系统，关键是要有**大规模平行语料库**——即“源语言-目标语言”成对的句子集合（如大量英法对照的句子对）。模型通过学习这些“成对样本”，才能掌握“源语句→目标语句”的映射规律，进而准确计算 $P(y|x)$。


简言之，这张幻灯片从“概率本质”和“训练基础”两个维度，深化了对NMT的理解：==它是一个“基于源语句条件、逐词生成目标语句”的概率模型，训练依赖大规模平行语料。==

### Training a Neural Machine Translation system

训练神经机器翻译系统

![image-20251029114348420](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029114348420.png)

- **标题**：Training a Neural Machine Translation system → 训练神经机器翻译系统
- **损失函数公式**：$J = \frac{1}{T}\sum_{t=1}^T J_t$
  - 解释：整体损失是每个时间步损失的平均值，其中 $J_t$ 是第 $t$ 步的损失。
- **损失项标注**：
  - $J_1$ = negative log prob of “he” → $J_1$ = “he”的负对数概率
  - $J_4$ = negative log prob of “with” → $J_4$ = “with”的负对数概率
  - $J_7$ = negative log prob of <END> → $J_7$ = <END>的负对数概率
- **模型组件**：
  - Encoder RNN → 编码器RNN
  - Decoder RNN → 解码器RNN
  - Source sentence (from corpus) → 源语句（来自语料库）：*il, a, m’ entarté*
  - Target sentence (from corpus) → 目标语句（来自语料库）：*<START>, he, hit, me, with, a, pie*
- **底部说明**：Seq2seq is optimized as a single system. Backpropagation operates “end-to-end”. → 序列到序列模型作为一个单一系统被优化，反向传播以“端到端”的方式进行。


#### 解释
这张图展示了**神经机器翻译（NMT）的训练过程**，核心是通过“端到端的损失优化”让模型学习“源语句→目标语句”的映射：

#### 1. 损失函数的本质
NMT的训练目标是**最大化“目标语句在源语句条件下的概率”**，即最大化 $P(y|x)$。为了用梯度下降优化，通常将其转换为**最小化负对数概率**（即损失函数 $J$）。

公式中 $J_t$ 是“第 $t$ 步生成目标词的负对数概率”，整体损失 $J$ 是所有时间步损失的平均值。例如：
- 第1步生成“he”，$J_1$ 是“模型预测‘he’的概率的负对数”；
- 第4步生成“with”，$J_4$ 是“模型预测‘with’的概率的负对数”；
- 最后一步生成<END>，$J_7$ 是“模型预测<END>的概率的负对数”。

损失越小，说明模型在该步生成目标词的概率越高，翻译越准确。


#### 2. 端到端训练的实现
Seq2seq模型是**作为一个整体进行端到端训练**的：
- 编码器RNN处理源语句，生成语义编码；
- 解码器RNN基于该编码和目标语句的历史词（如<START>、he、hit等），逐词生成目标词；
- 每一步的预测误差（损失 $J_t$）会通过反向传播，同时更新编码器和解码器的所有参数，让模型整体朝着“准确翻译”的方向优化。


#### 3. 训练数据的依赖
训练依赖**平行语料库**（即“源语句-目标语句”成对的样本）。模型通过学习这些成对样本，逐步调整参数，使得“输入源语句时，生成对应目标语句的概率最大”。


简言之，这张图直观呈现了NMT的训练逻辑：以“最小化目标语句的负对数概率”为目标，通过端到端的反向传播，同时优化编码器和解码器的参数，最终让模型学会准确的翻译映射。

#### 4、end-to-end

在人工智能和机器学习领域，“**端到端（end-to-end）**” 是一种核心的设计与训练理念，指的是**将一个任务的 “输入” 到 “输出” 直接构建为一个完整的系统，通过端到端的优化（通常是反向传播）来学习从输入到输出的直接映射，而不需要人工设计中间步骤或子模块**。

##### 具体理解：以神经机器翻译（NMT）为例

在传统机器翻译中，流程是 “分词→词性标注→短语提取→翻译规则生成→目标语言生成”，每个步骤都需要人工设计规则或子模型，属于 **“分步骤” 的 Pipeline 模式 **。

而 NMT 的 “端到端” 训练则是：

- **输入**：源语言句子（如法语 “il a m’ entarté”）；
- **输出**：目标语言句子（如英语 “he hit me with a pie”）；
- **模型**：一个由 “编码器 RNN + 解码器 RNN” 组成的 Seq2Seq 系统；
- **优化**：直接通过 “预测目标词的概率损失” 来反向传播，同时更新编码器和解码器的所有参数，让模型**直接学习 “源句→目标句” 的端到端映射**，无需人工干预中间过程。

##### 核心特点

1. **无需人工设计中间模块**：传统方法需要人为拆分任务、设计子步骤（如特征提取、规则定义），而端到端模型将任务视为 “输入→输出” 的黑箱，中间过程由模型自动学习。
2. **全局优化**：损失函数直接针对最终任务目标（如翻译的准确性），反向传播时会同时调整所有模块的参数，确保整个系统的协同优化。
3. **简化流程**：降低了对领域知识的依赖，让模型能从大规模数据中自主学习模式，尤其适合复杂的序列任务（如翻译、摘要、对话）。

##### 其他领域的端到端示例

- **图像识别**：输入是原始图像，输出是类别标签（如 “猫”“狗”），模型（如 CNN）直接学习 “像素→类别” 的映射；
- **语音识别**：输入是音频波形，输出是文字转录，模型（如端到端 ASR 系统）直接学习 “声音→文字” 的映射；
- **自动驾驶**：输入是传感器数据（图像、雷达），输出是控制指令（转向、加速），模型直接学习 “环境→动作” 的映射。

简言之，“端到端” 的本质是**让模型直接学习 “原始输入到最终输出” 的完整映射，通过全局优化实现任务目标**，是现代深度学习中 “自动化、数据驱动” 理念的典型体现。

#### Multi-layer RNNs

多层RNNs

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029115313288.png)



- **标题**：Multi-layer RNNs → 多层循环神经网络（多层RNN）
- **要点1**：RNNs are already “deep” on one dimension (they unroll over many timesteps) → 循环神经网络（RNN）在一个维度上已经是“深度”的（它们在多个时间步上展开）
- **要点2**：We can also make them “deep” in another dimension by applying multiple RNNs – this is a multi-layer RNN. → 我们还可以通过应用多个RNN，在另一个维度上让它们变得“深度”——这就是多层RNN。
- **要点3**：This allows the network to compute more complex representations → 这使得网络能够计算更复杂的表示
  - The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features. → 下层RNN应计算低层级特征，上层RNN应计算高层级特征。
- **要点4**：Multi-layer RNNs are also called stacked RNNs. → 多层RNN也被称为堆叠RNN。


#### 解释
这张幻灯片介绍了**多层循环神经网络（Multi-layer RNN，也叫堆叠RNN）**的核心设计逻辑，本质是通过“在垂直维度堆叠多个RNN层”，让模型学习更复杂的特征表示：

##### 1. RNN的“双重深度”
RNN本身有两种“深度”维度：
- **时间维度的深度**：RNN在处理序列时，会在“时间步”上展开（如处理一个句子的每个单词时，时间步从1到n），这是RNN天然的“深度”；
- **层级维度的深度**：通过堆叠多个RNN层（多层RNN），在“垂直方向”上增加深度，这是人工设计的“深度”。


##### 2. 多层RNN的特征学习分工
多层RNN的每一层有明确的“特征抽象层级”：
- **下层RNN**：学习“低层级特征”，如文本中的单词、短语的局部语义，或语音中的基本音素；
- **上层RNN**：基于下层的低层级特征，学习“高层级特征”，如文本中的句子语义、篇章逻辑，或语音中的完整语句含义。

这种分工让模型能从“局部到全局”逐步抽象信息，最终生成更复杂、更具语义价值的表示。


简言之，**多层RNN（堆叠RNN）是通过“垂直堆叠多个RNN层”，让模型在“时间深度”之外，增加“层级深度”，从而实现从“低层级特征”到“高层级语义”的逐步抽象，提升对复杂序列任务的处理能力。**

![image-20251029115747045](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029115747045.png)

- **标题**：Multi-layer deep encoder-decoder machine translation net → 多层深度编码器 - 解码器机器翻译网络

- **参考文献**：[Sutskever et al. 2014; Luong et al. 2015]

- **关键说明**：The hidden states from RNN layer *i* are the inputs to RNN layer *i+1* → RNN 第*i*层的隐藏状态是第*i+1*层的输入

- 模块说明

  ：

  - Encoder: Builds up sentence meaning → 编码器：构建语句语义
  - Source sentence → 源语句：*Die, Proteste, waren, am, Wochenende, eskaliert, <EOS>*
  - Decoder → 解码器
  - Translation generated → 生成的翻译：*The, protests, escalated, over, the, weekend, <EOS>*
  - Feeding in last word → 输入上一个单词

- **底部说明**：Conditioning = Bottleneck → 条件限制 = 瓶颈



这张图展示了**多层深度编码器 - 解码器架构在机器翻译中的应用**，核心是通过 “堆叠多层 RNN” 提升模型对复杂语义的建模能力，同时也揭示了早期 Seq2Seq 模型的一个关键瓶颈：

##### 1. 多层 RNN 的堆叠逻辑

- **编码器侧**：源语句（如德语 “Die Proteste waren am Wochenende eskaliert <EOS>”）被逐层输入 RNN—— 下层 RNN 的隐藏状态作为上层 RNN 的输入，实现 “从低层级特征到高层级语义” 的逐步抽象（如从单词语法到句子整体含义）。
- **解码器侧**：生成目标语句（如英语 “The protests escalated over the weekend <EOS>”）时，同样通过多层 RNN，上层依赖下层的隐藏状态，逐词生成翻译结果。

##### 2. “条件限制即瓶颈” 的含义

在早期 Seq2Seq 模型中，编码器的最终语义编码是**固定长度的向量**，它需要 “压缩” 整个源语句的语义信息，然后传递给解码器。这种 “将变长序列压缩为定长向量” 的过程，就是所谓的 “条件限制（Conditioning）”—— ==它会导致**信息丢失**（尤其是长句的语义细节），成为模型性能的 “瓶颈（Bottleneck）”==。

这也是后续 “注意力机制（Attention）” 被提出的重要动机 —— 让解码器能动态关注编码器的不同位置，避免单一固定向量的信息瓶颈。

### multi-layer RNNs in practice

实践中的多层RNNs

![image-20251029120254646](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029120254646.png)

- **标题**：Multi-layer RNNs in practice → 多层循环神经网络的实践应用
- **要点 1**：High-performing RNNs are usually multi-layer (but aren’t as deep as convolutional or feed-forward networks) → **高性能的循环神经网络（RNN）通常是多层的（但不像卷积网络或前馈网络那样深）**
- 要点 2：For example: In a 2017 paper, Britz et al. find that for Neural Machine Translation, 2 to 4 layers is best for the encoder RNN, and 4 layers is best for the decoder RNN → 例如：在 2017 年的一篇论文中，Britz 等人发现，对于神经机器翻译，**编码器 RNN 最好有 2 到 4 层，解码器 RNN 最好有 4 层**
  - Often 2 layers is a lot better than 1, and 3 might be a little better than 2 → 通常 2 层比 1 层好很多，3 层可能比 2 层稍好一些
  - Usually, skip-connections/dense-connections are needed to train deeper RNNs (e.g., 8 layers) → ==通常，训练更深的 RNN（例如 8 层）需要跳跃连接(也就是残差连接) / 密集连接==
- **要点 3**：Transformer-based networks (e.g., BERT) are usually deeper, like 12 or 24 layers. → 基于 Transformer 的网络**（例如 BERT）通常更深，比如 12 层或 24 层**
  - You will learn about Transformers later; they have a lot of skipping-like connections → 你之后会学习 Transformer；它们有很多类似跳跃的连接
- **参考文献**：“Massive Exploration of Neural Machine Translation Architectures”, Britz et al, 2017. https://arxiv.org/pdf/1703.03906.pdf

#### 解释

这张幻灯片聚焦**多层 RNN 在实际应用中的设计规律**，同时对比了 RNN 与 Transformer 在 “深度” 和 “结构” 上的差异，帮助理解不同序列模型的工程化选择：

##### 1. 多层 RNN 的性能与层数关系

在神经机器翻译等任务中，多层 RNN 的性能呈现 “边际效益递减” 的规律：

- 从 1 层到 2 层，性能提升显著（因为 2 层能学习更复杂的特征抽象）；
- 从 2 层到 3 层，性能有小幅提升；
- 超过 4 层后，性能提升有限，甚至可能因 “梯度消失 / 爆炸” 而下降，因此实际中编码器 RNN 常用 2-4 层，解码器 RNN 常用 4 层。

##### 2. 深 RNN 的训练技巧：跳跃连接 / 密集连接

当 RNN 层数超过 4 层（如 8 层）时，单纯堆叠会导致**梯度传播困难**（训练不稳定）。此时需要引入 “跳跃连接” 或 “密集连接”：

- 跳跃连接：让某一层的输出直接连接到后面隔层的输入，**缓解梯度消失**；
- ==密集连接：每一层都与前面所有层的输出连接，增强信息流动。==

### Greedy decoding 

贪心解码

![image-20251029121108602](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029121108602.png)

#### 提取文字与翻译
- **标题**：Greedy decoding → 贪心解码
- **要点1**：We saw how to generate (or “decode”) the target sentence by taking argmax on each step of the decoder → 我们了解了如何通过在解码器的每一步取argmax（最大概率）来生成（或“解码”）目标语句
- **要点2**：This is greedy decoding (take most probable word on each step) → 这就是贪心解码（每一步选择最可能的单词）
- **要点3**：Problems with this method? → 这种方法的问题是什么？
- **图示元素**：
  - 生成的目标语句：*he, hit, me, with, a, pie, <END>*
  - 解码器输入（每一步的历史单词）：*<START>, he, hit, me, with, a, pie*
  - **决策函数：*argmax***


#### 解释
这张图展示了**贪心解码的核心逻辑**：在序列生成任务（如机器翻译、文本生成）中，解码器每一步都选择“当前概率最大的单词”，直到生成终止符（如<END>）。

##### 1. 贪心解码的流程
以机器翻译为例：
- 第一步：输入<START>，解码器选择概率最大的单词“he”；
- 第二步：输入“he”，解码器选择概率最大的单词“hit”；
- 后续步骤以此类推，最终生成完整语句“he hit me with a pie <END>”。

这种“逐词贪心选择”的策略简单高效，计算成本低。


##### 2. 贪心解码的缺陷（为后续内容铺垫）
尽管高效，但贪心解码存在**“局部最优导致全局次优”**的问题：
- 某一步选择了“当前最优”的单词，但可能导致后续步骤的选择空间被严重限制，最终生成的整体序列并非“概率最大的全局最优序列”。
- 例如，在翻译中，某一步贪心选择了单词A（当前概率最高），但后续可能需要搭配概率很低的单词B才能组成通顺的语句，**而如果第一步选择单词C（当前概率次高），后续可能有更优的组合**。

简言之，这张图直观呈现了贪心解码的“逐词最大化”逻辑，同时抛出了其核心缺陷的问题，为理解后续“波束搜索”等更优解码策略奠定了基础。

### Problems with greedy decoding 

贪心解码的问题

![image-20251029121126833](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029121126833.png)

- **标题**：Problems with greedy decoding → 贪心解码的问题
- **要点1**：Greedy decoding has no way to undo decisions! → **贪心解码无法撤销已做的决策！**
  - Input: *il a m’entarté* （对应正确翻译 *he hit me with a pie*）
  - 生成步骤示例：
    - → *he ______*
    - → *he hit ______*
    - → *he hit a ______* （标注：*whoops! no going back now...* → 哎呀！现在没法回头了……）
- **要点2**：How to fix this? → 如何解决这个问题？


#### 解释
这张幻灯片聚焦**贪心解码的核心缺陷**——“决策不可逆”，导致其容易陷入“局部最优但全局次优”的困境：

以机器翻译为例，源句是法语 *“il a m’entarté”*（正确翻译是 *“he hit me with a pie”*）。贪心解码的生成过程可能出现以下问题：
- 第一步选了 *“he”*（合理）；
- 第二步选了 *“hit”*（合理）；
- 第三步错误地选了 *“a”*（而非正确的 *“me”*）；
- 此时，后续的决策只能基于 *“he hit a”* 继续生成，即便这个开头是错误的，也**无法回溯修正**，最终导致整体翻译错误。

**这种“一步错步步错”的问题，本质是贪心解码“只看当前步最优，不考虑全局”的策略缺陷。**为解决这一问题，后续发展出了**波束搜索（Beam Search）**等方法——每一步保留多个候选序列（而非仅一个），从而在一定程度上兼顾“局部选择”和“全局最优”，同时控制计算复杂度。


简言之，这张幻灯片通过实例直观展现了贪心解码的不可逆缺陷，为理解更优的解码策略（如波束搜索）提供了必要性的逻辑支撑。

### Exhaustive search decoding 

穷举搜索解码

![image-20251029121146304](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029121146304.png)



- **要点1**：Ideally, we want to find a (length *T*) translation *y* that maximizes
  $$
  \begin{align*}
  P(y|x) &= P(y_1|x)\ P(y_2|y_1, x)\ P(y_3|y_1, y_2, x)\ \dots, P(y_T|y_1, \dots, y_{T-1}, x) \\
  &= \prod_{t=1}^T P(y_t|y_1, \dots, y_{t-1}, x)
  \end{align*}
  $$
  → 理想情况下，我们希望找到一个长度为*T*的翻译*y*，使其最大化上述概率（即目标语句在源语句条件下的联合概率）。
- **要点2**：We could try computing all possible sequences *y* → ==我们可以尝试计算所有可能的序列*y*==

  - This means that on each step *t* of the decoder, we’re tracking *V<sup>t</sup>* possible partial translations, where *V* is vocab size → **这意味着在解码器的每一步*t*，我们要跟踪*V<sup>t</sup>*种可能的部分翻译（*V*是词汇表大小）**。
  - This *O(V<sup>T</sup>)* complexity is far too expensive! → 这种*O(V<sup>T</sup>)*的复杂度**开销极大**！


#### 解释
这张幻灯片介绍了“穷举搜索解码”的思路与缺陷，是理解“为何需要更高效解码策略（如波束搜索）”的关键：

##### 1. 穷举搜索的目标
穷举搜索的理想目标是**找到“概率最大的目标序列”**，即最大化联合概率 $P(y|x) = \prod_{t=1}^T P(y_t|y_1, \dots, y_{t-1}, x)$。从理论上，这是最“完美”的解码方式——能保证生成最优翻译。


##### 2. 穷举搜索的不可行性
但在实际中，这种方法完全不可行，原因是**复杂度爆炸**：
- 假设词汇表大小*V*是10,000，目标语句长度*T*是20，那么需要遍历的序列数量是 $10,000^{20}$（这个数字远超宇宙原子数量）；
- 每一步*t*的候选数是*V<sup>t</sup>*，时间复杂度是*O(V<sup>T</sup>)*，在计算上完全无法承受。


##### 3. 与贪心解码的对比与联系
- 贪心解码是“每一步选概率最大的单词，不回溯”，虽然高效但可能陷入“局部最优”；
- 穷举搜索是“尝试所有可能，选全局最优”，虽然理论最优但完全不可行；
- 两者的矛盾催生了**波束搜索（Beam Search）**——一种“折中的高效策略”：每一步保留*k*个最可能的候选序列（*k*为波束宽度），既避免了贪心的局部最优，又控制了复杂度（时间复杂度*O(kVT)*）。


简言之，这张幻灯片揭示了“穷举搜索”在理论上的最优性和实践中的不可行性，为理解后续“波束搜索”等实用解码算法提供了逻辑铺垫。

### Beam search decoding 

 波束搜索解码

![image-20251029123117384](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123117384.png)



- **标题**：Beam search decoding → 波束搜索解码
- **核心思想**：Core idea: On each step of decoder, keep track of the *k* most probable partial translations (which we call *hypotheses*) → **核心思想**：==在解码器的每一步，跟踪*k*个最可能的部分翻译（我们称之为“假设”）==
  - *k* is the beam size (in practice around 5 to 10) → ==*k*是波束宽度（实际中约为5到10）==
- **假设的得分计算**：A hypothesis $y_1, \dots, y_t$ has a score which is its log probability:
  $$\text{score}(y_1, \dots, y_t) = \log P_{\text{LM}}(y_1, \dots, y_t|x) = \sum_{i=1}^t \log P_{\text{LM}}(y_i|y_1, \dots, y_{i-1}, x)$$
  - Scores are all negative, and higher score is better → ==得分都是负数，得分越高越好==
  - We search for high-scoring hypotheses, tracking top *k* on each step → ==我们搜索高分假设，每一步跟踪前*k*个==
- **性能特点**：
  - Beam search is not guaranteed to find optimal solution → **波束搜索不保证找到最优解**
  - But much more efficient than exhaustive search! → 但比穷举搜索高效得多！


#### 解释
这张幻灯片介绍了**波束搜索解码**的核心逻辑，它是贪心解码和穷举搜索的“折中方案”，在“搜索质量”和“计算效率”之间取得平衡：

##### 1. 波束搜索的核心机制
- **跟踪*k*个候选假设**：每一步生成单词时，不只是选“一个最优”（贪心），而是保留*k*个“当前最可能的部分翻译序列”（称为假设）。例如，波束宽度*k*=2时，每一步都有2个候选序列在竞争，直到生成终止符。
- **得分是对数概率和**：每个假设的得分是“生成该序列的联合概率的对数和”（因概率是小数，取对数后为负数，得分越高表示概率越大）。


##### 2. 与贪心、穷举的对比
- 对比贪心解码：波束搜索通过保留*k*个候选，避免了“一步错步步错”的局部最优问题；
- 对比穷举搜索：波束搜索的时间复杂度是*O(kVT)*（*V*是词汇表大小，*T*是序列长度），远低于穷举的*O(V<sup>T</sup>)*，在实际中可计算；
- 缺陷：不保证找到“全局概率最大的序列”，但在*k*足够大时（如5-10），能生成非常接近最优的结果。


##### 3. 实际应用价值
波束搜索是**序列生成任务（如机器翻译、文本摘要、对话生成）中最常用的解码策略**，例如在神经机器翻译中，通过设置合适的波束宽度（如5-10），能在可控的计算成本下，大幅提升翻译质量（相比贪心解码）。


简言之，**波束搜索通过“每步保留*k*个候选序列”的策略，既缓解了贪心解码的局部最优问题，又避免了穷举搜索的计算爆炸，是序列生成任务中“性价比最高”的解码方法之一。**

### Beam search decoding: example

波束搜索解码：示例

![image-20251029123431395](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123431395.png)



- **标题**：Beam search decoding: example → 波束搜索解码：示例
- **说明**：Beam size = k = 2. Blue numbers = $\text{score}(y_1, \dots, y_t) = \sum_{i=1}^t \log P_{\text{LM}}(y_i|y_1, \dots, y_{i-1}, x)$ → 波束宽度=k=2。蓝色数字=得分（即生成序列$y_1, \dots, y_t$的对数概率和）
- **初始输入**：`<START>`
- **步骤说明**：Calculate prob dist of next word → **计算下一个单词的概率分布**

![image-20251029123551274](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123551274.png)

![image-20251029123620825](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123620825.png)

![image-20251029123644700](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123644700.png)

![image-20251029123654697](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029123654697.png)

![image-20251029124016769](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029124016769.png)

![image-20251029124035072](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029124035072.png)



**解释**：beam search每次只保留beam size个候选项，计算前beam size个最大的概率结果，每个备选也是计算这些多，然后新的序列排序后再选择前beam size个最大的概率结果，其他项丢弃，以此类推来得到最终的结果。

### Beam search decoding: stopping criterion 

 波束搜索解码：停止准则

![image-20251029124205601](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029124205601.png)



- **标题**：Beam search decoding: stopping criterion → 波束搜索解码：停止准则
- **贪心解码的停止逻辑**：In greedy decoding, usually we decode until the model produces an <END> token → **在贪心解码中，通常解码到模型生成<END>标记为止**
  - 示例：<START> he hit me with a pie <END>
- **波束搜索的停止逻辑**：In beam search decoding, different hypotheses may produce <END> tokens on different timesteps → **在波束搜索解码中，不同假设可能在不同时间步生成<END>标记**
  - When a hypothesis produces <END>, that hypothesis is complete. → 当一个假设生成<END>时，该假设即完成。
  - Place it aside and continue exploring other hypotheses via beam search. → ==将其搁置，继续通过波束搜索探索其他假设。==
- **波束搜索的停止条件**：Usually we continue beam search until: → 通**常我们会继续波束搜索，直到满足以下条件之一：**
  - We reach timestep *T* (where *T* is some pre-defined cutoff), or → ==达到时间步*T*（*T*是预先定义的截断值）==，或
  - We have at least *n* completed hypotheses (where *n* is pre-defined cutoff) → ==至少有*n*个完成的假设（*n*是预先定义的截断值）==


### 解释
这张幻灯片聚焦**波束搜索的“停止准则”**，解决“何时停止生成序列”的问题，这是波束搜索工程化的关键细节：

#### 1. 与贪心解码的停止逻辑对比
- 贪心解码：生成过程是“线性的”，直到生成<END>标记就立即停止，整个过程只有一条序列；
- 波束搜索：生成过程是“并行的”，多个假设（候选序列）同时推进，不同假设可能在不同时间步生成<END>。例如，波束宽度k=2时，可能一个假设在第5步生成<END>，另一个在第7步生成<END>。


#### 2. 波束搜索的停止处理
当某一假设生成<END>时，它会被“搁置”（视为完成的候选），但其他未完成的假设会继续生成，直到满足以下两个停止条件之一：
- **时间步截断（T）**：如果生成到预先设定的最大时间步*T*（如50步）仍有假设未完成，强制停止，避免无限生成；
- **完成假设数截断（n）**：如果已经生成了至少*n*个完成的假设（如3个），则停止，确保有足够的候选序列供最终选择。


#### 3. 工程意义
这种停止准则的设计，既保证了波束搜索能探索到足够多的候选序列（提升找到优质翻译的概率），又避免了无限制的计算开销。例如，在机器翻译中，通常会设置*T*=100（避免过长序列）和*n*=5（确保有足够多的完成候选），从而在“搜索充分性”和“计算效率”之间取得平衡。

简言之，波束搜索的停止准则是其工程化落地的关键环节，通过“多假设并行生成+双截断条件”，确保在可控的成本下生成高质量的序列。

### Beam search decoding: finishing up 

 波束搜索解码：收尾阶段

![image-20251029124154252](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029124154252.png)



- **标题**：Beam search decoding: finishing up → 波束搜索解码：收尾阶段
- **步骤1**：We have our list of completed hypotheses. → 我们有了已完成的假设列表。
- **步骤2**：How to select top one with highest score? → **如何选择得分最高的那个？**
- **得分计算**：Each hypothesis $y_1, \dots, y_t$ on our list has a score
  $$\text{score}(y_1, \dots, y_t) = \log P_{\text{LM}}(y_1, \dots, y_t|x) = \sum_{i=1}^t \log P_{\text{LM}}(y_i|y_1, \dots, y_{i-1}, x)$$
  → 列表中的每个假设$y_1, \dots, y_t$都有一个得分，即生成该序列的联合对数概率和。
- **问题**：Problem with this: longer hypotheses have lower scores → 问题：更长的假设得分更低
- **解决方法**：Fix: Normalize by length. Use this to select top one instead:
  $$\frac{1}{t} \sum_{i=1}^t \log P_{\text{LM}}(y_i|y_1, \dots, y_{i-1}, x)$$
  → 解决：按长度归一化。改用上述公式选择最优序列（即平均每步的对数概率）。


#### 解释
这张幻灯片聚焦波束搜索**最终选优的“长度偏置问题”及解决方法**，是波束搜索工程化的关键细节：

##### 1. 原始得分的缺陷
原始得分是“生成序列的联合对数概率和”，但存在**“长度偏置”**——更长的序列（单词数更多）得分会更低（因为是多个对数概率的累加，而每个对数概率是负数）。例如，一个短序列“he hit”的得分可能比长序列“he hit me with a pie”更高，但后者显然是更完整、更优的翻译。

这种偏置会导致波束搜索错误地优先选择短序列，而非语义更完整的长序列。


##### 2. 长度归一化的解决方案
为消除长度偏置，需要对得分进行**“按长度归一化”**——将总得分除以序列长度$t$，得到“平均每步的对数概率”。这样，无论序列长短，都以“每步的平均概率”作为衡量标准，确保长序列不会因长度劣势被误判。

例如，**序列A（长度2，总得分-4.0）的归一化得分是-2.0；序列B（长度5，总得分-7.5）的归一化得分是-1.5。此时序列B的归一化得分更高，会被选为更优序列。**

简言之，**波束搜索的收尾阶段通过“长度归一化”解决了原始得分的长度偏置问题，确保最终选择的是“每步平均概率最高”的最优序列，而非单纯长度最短的序列。**

### Advantages of NMT 

神经机器翻译（NMT）的优势

![image-20251029164541027](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029164541027.png)



- **标题**：Advantages of NMT → 神经机器翻译（NMT）的优势
- **对比说明**：Compared to SMT, NMT has many advantages: → 与统计机器翻译（SMT）相比，NMT有诸多优势：
- **性能更优**：
  - Better performance → 性能更优
    - More fluent → 更流畅
    - Better use of context → 更善于利用上下文
    - Better use of phrase similarities → 更善于利用短语相似性
- **端到端优化的单一神经网络**：
  - A single neural network to be optimized end-to-end → 可端到端优化的单一神经网络
    - No subcomponents to be individually optimized → 无需对各子组件单独优化
- **人工工程投入更少**：
  - Requires much less human engineering effort → **所需人工工程投入大幅减少**
    - No feature engineering → **无需特征工程**
    - Same method for all language pairs → **所有语言对采用同一套方法**


#### 解释
这张幻灯片总结了**神经机器翻译（NMT）相对于统计机器翻译（SMT）的核心优势**，体现了NMT在“性能、可优化性、工程效率”上的突破：

##### 1. 性能维度：从“生硬”到“流畅自然”
- **更流畅**：NMT生成的译文在语法和语义连贯性上远超SMT。例如，SMT可能逐词翻译导致语句生硬，而NMT能生成“he hit me with a pie”这类自然流畅的表达；
- **上下文利用更优**：NMT的编码器能捕捉长距离上下文依赖（如代词指代、语义连贯性），而SMT对长文本的上下文处理能力较弱；
- **短语相似性利用更优**：NMT能学习到“短语级”的翻译模式（如固定搭配、习语），而SMT依赖人工设计的短语表，灵活性和覆盖度不足。


##### 2. 优化维度：从“组件拼凑”到“端到端统一”
SMT由“词对齐、短语提取、翻译模型、语言模型”等多个子组件构成，每个组件需单独优化且存在“组件间误差累积”问题；而NMT是**单一的编码器-解码器神经网络**，可通过端到端的反向传播直接优化“源语句→目标语句”的整体映射，避免了子组件间的协同问题。


##### 3. 工程维度：从“定制化”到“通用化”
- **无需特征工程**：SMT需要人工设计大量特征（如词性、句法结构），而NMT直接从数据中学习特征表示，大幅减少人工成本；
- **多语言对通用**：同一套NMT架构可应用于所有语言对（如英法、中英、中日），而SMT需为不同语言对定制化开发，扩展性极差。


简言之，NMT的这些优势使其在机器翻译领域迅速取代SMT，成为当前主流的翻译范式，也为后续基于Transformer的更先进翻译模型奠定了基础。

### Disadvantages of NMT? 

神经机器翻译（NMT）的劣势？

![image-20251029165015928](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029165015928.png)



- **标题**：Disadvantages of NMT? → 神经机器翻译（NMT）的劣势？
- **对比说明**：Compared to SMT: → 与统计机器翻译（SMT）相比：
- **可解释性差**：
  - NMT is less **interpretable** → NMT的可解释性更差
    - Hard to **debug** → 难以调试
- **难以控制**：
  - NMT is difficult to control → NMT难以控制
    - For example, can’t easily specify rules or guidelines for translation → 例如，无法轻松指定翻译规则或准则
    - **Safety concerns**! → 存在安全隐患！


### 解释
这张幻灯片总结了**神经机器翻译（NMT）相对于统计机器翻译（SMT）的核心劣势**，体现了NMT在“可解释性”和“可控性”上的不足：

#### 1. 可解释性维度：从“透明”到“黑箱”
SMT的各个组件（如词对齐、短语表、语言模型）是可解释的，工程师能明确知道“翻译结果由哪些规则或特征驱动”；而NMT是**端到端的神经网络黑箱**，其翻译决策依赖于神经元的隐式特征学习，难以拆解“为什么生成这个译文”，一旦出现翻译错误，也很难定位问题根源（如到底是编码器没理解源语句，还是解码器生成逻辑有问题）。


#### 2. 可控性维度：从“规则驱动”到“数据驱动”
SMT可通过人工设计规则、短语表来严格控制翻译风格（如强制使用某类术语、遵循特定语法规范）；而NMT的翻译行为完全由训练数据驱动，若需调整翻译风格（如从口语化改为书面化），只能通过“修改训练数据+重新训练模型”实现，无法像SMT那样“即时通过规则干预”。这种不可控性还会引发**安全隐患**——例如，NMT可能生成有害、偏见或不符合伦理的内容，且难以通过简单规则来规避。

简言之，NMT的这些劣势是其“端到端数据驱动”特性的副作用，后续研究（如引入注意力可视化、可控解码策略）也在不断尝试解决这些问题，以平衡NMT的性能优势和可解释、可控性需求。

### How do we evaluate Machine Translation?

![image-20251029165534857](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029165534857.png)



- **标题**：How do we evaluate Machine Translation? → 如何评估机器翻译？
- **方法**：BLEU (Bilingual Evaluation Understudy) → BLEU（双语评估替补）
- **说明框**：You’ll see BLEU in detail in Assignment 4! → 你会在作业4中详细学习BLEU！
- **BLEU的核心逻辑**：
  - BLEU compares the machine-written translation to one or several human-written translation(s), and computes a similarity score based on: → BLEU将机器翻译与一个或多个人工翻译进行比较，并基于以下因素计算相似度得分：
    - *n*-gram precision (usually for 1, 2, 3 and 4-grams) → **n-gram精确率（通常针对1、2、3、4元组）**
    - Plus a **penalty**(惩罚) for too-short system translations → 加上对过短系统翻译的惩罚
- **BLEU的特点**：
  - BLEU is useful but i**mperfect** → BLEU有用但不完美
    - There are many valid ways to translate a sentence → 一个句子有多种合理的翻译方式
    - So a good translation can get a poor BLEU score because it has low *n*-gram overlap with the human translation 😢 → **因此，一个好的翻译可能因与人工翻译的n-gram重叠度低而得到较差的BLEU分数** 😢
- **参考文献**：Source: “BLEU: a Method for Automatic Evaluation of Machine Translation”, Papineni et al, 2002. http://aclweb.org/anthology/P02-1040


### 解释
这张幻灯片介绍了机器翻译领域**最常用的自动评估指标——BLEU**，它是衡量“机器翻译与人工翻译相似度”的核心工具：

#### 1. BLEU的计算逻辑
BLEU通过“n-gram精确率”和“长度惩罚”来评分：
- **n-gram精确率**：统计机器翻译中“1元组（单词）、2元组（词对）、3元组（词三元组）、4元组（词四元组）”与人工翻译的重叠比例，重叠度越高，精确率越高；
- **长度惩罚**：如果机器翻译的长度远短于人工翻译，会被扣分，避免模型生成过短的残缺翻译。


#### 2. BLEU的价值与缺陷
- **价值**：是机器翻译领域“自动化、可复现”的评估标准，能快速对比不同模型的翻译质量，推动了神经机器翻译的技术迭代；
- **缺陷**：无法完全替代人工评估，因为“好的翻译可能与参考译文的n-gram重叠度低”（例如，人工翻译用了同义短语，机器翻译用了不同但正确的表达）。


#### 3. 工程意义
尽管BLEU有缺陷，但仍是工业界和学术界评估机器翻译的**基准指标**。例如，在论文中对比不同NMT模型时，BLEU分数是核心的量化依据；在实际产品迭代中，也会通过BLEU监控模型优化方向。


简言之，**BLEU是机器翻译自动评估的“基石”，它提供了客观的量化标准，同时也需结合人工评估来全面判断翻译质量。**

### MT progress over time 

机器翻译（MT）的发展历程

>  “over time” 常见的中文翻译有 **“随着时间的推移”“久而久之”“长期以来”** 等

![image-20251029170854566](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029170854566.png)



- **标题**：MT progress over time → 机器翻译（MT）的发展历程
- **说明**：[Edinburgh En-De WMT newstest2013 Cased BLEU; NMT 2015 from U. Montréal; NMT 2019 FAIR on newstest2019] → [爱丁堡英德WMT newstest2013有大小写BLEU；2015年蒙特利尔大学的NMT；2019年FAIR在newstest2019上的NMT]
- **模型类型**：
  - Phrase-based SMT → 基于**短语**的统计机器翻译（SMT）
  - Syntax-based SMT → 基于**句法**的统计机器翻译（SMT）
  - Neural MT → 神经机器翻译（NMT）
- **数据来源**：Sources: http://www.meta-net.eu/events/meta-forum-2016/slides/09_sennrich.pdf & http://matrix.statmt.org/


#### 解释
这张图表直观呈现了**机器翻译技术随时间的性能演进**，以“英德翻译任务的BLEU分数”为量化指标，清晰对比了统计机器翻译（SMT）和神经机器翻译（NMT）的发展轨迹：

##### 1. 技术代际的性能跃迁
- **2013-2015年**：基于短语和句法的SMT性能停滞（BLEU分数约20-22），而2015年NMT（神经机器翻译）首次登场就实现了性能突破（BLEU分数约18-20）；
- **2016年后**：NMT性能呈“爆发式增长”，2019年BLEU分数突破40，远超同期SMT的性能（约24-26）。


##### 2. 技术路线的兴衰
- **SMT的局限**：基于短语和句法的SMT在2015年前是主流，但性能提升已触及天花板，无法满足高质量翻译需求；
- **NMT的崛起**：NMT凭借“端到端学习、上下文建模能力强”的优势，在2015年后迅速取代SMT，成为机器翻译的主流技术，并持续推动性能突破。


##### 3. 行业意义
这张图是机器翻译领域“技术迭代”的缩影，体现了**神经模型对传统统计模型的超越**，也解释了为何当前主流翻译系统（如谷歌翻译、百度翻译）均基于NMT或其升级版（如基于Transformer的模型）构建。

简言之，**这张图表以BLEU分数为锚点，清晰展示了机器翻译从“统计时代”迈入“神经时代”的性能跨越，凸显了NMT在翻译质量上的革命性进步。**

### NMT：perhaps the biggest success story of NLP Deep Learning?

 神经机器翻译（NMT）：或许是自然语言处理（NLP）深度学习领域最成功的案例？

![image-20251029171549155](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029171549155.png)



- **发展历程**：Neural Machine Translation went from a fringe research attempt in 2014 to the leading standard method in 2016 → 神经机器翻译从2014年的边缘研究尝试，发展为2016年的主流标准方法
- **关键时间节点**：
  - 2014: First seq2seq paper published → **2014年：首篇序列到序列（seq2seq）论文发表**
  - 2016: Google Translate switches from SMT to NMT – and by 2018 everyone has → 2016年：谷歌翻译从统计机器翻译（SMT）切换到NMT——**到2018年，所有主流厂商均采**用（图示厂商包括Microsoft、SYSTRAN、Google、facebook、百度、网易、腾讯、搜狗搜索）
- **突破性意义**：
  - This is amazing! → 这太惊人了！
  - SMT systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by a small group of engineers in a few months → **由数百名工程师历时多年构建的SMT系统，被一小支工程师团队在数月内训练的NMT系统超越**


#### 解释
这张幻灯片聚焦**神经机器翻译（NMT）的行业颠覆性影响**，展现了其从“边缘研究”到“行业标准”的爆发式发展：

##### 1. 技术迭代的速度
NMT仅用2年时间（2014-2016）就完成了从“学术尝试”到“主流方法”的跨越：
- 2014年，seq2seq架构的提出为NMT奠定了基础；
- 2016年，谷歌翻译全面切换到NMT，引发行业跟风；
- 2018年，微软、百度、腾讯等全球主流厂商均采用NMT技术，标志着其成为机器翻译的事实标准。


##### 2. 对传统技术的降维打击
传统统计机器翻译（SMT）是“数百名工程师历时多年”的工程结晶，而NMT仅需“一小支团队数月”的研发，就在翻译质量上实现了超越。这种“小成本、快迭代、高性能”的特性，彻底改写了机器翻译的技术格局。


##### 3. 行业与学术的双重意义
- **行业端**：NMT推动了谷歌翻译、百度翻译等产品的用户体验跃升，让高质量机器翻译从“技术奢侈品”变为“大众消费品”；
- **学术端**：NMT的成功验证了“深度学习+端到端学习”在NLP复杂任务中的可行性，为后续Transformer、大语言模型等技术的爆发埋下了伏笔。

简言之，NMT是深度学习在NLP领域“最具标志性的成功案例”，它不仅实现了机器翻译质量的跃迁，更重塑了整个NLP领域的技术路线和行业生态。

### So, is Machine Translation solved? 

 那么，机器翻译问题已经解决了吗？

![image-20251029171809001](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029171809001.png)



- **标题**：So, is Machine Translation solved? → 那么，机器翻译问题已经解决了吗？
- **回答**：Nope! → 没有！
- **仍存在的诸多难点**：Many difficulties remain:
  - Out-of-vocabulary words → 未登录词（**词汇表外的单词**）
  - Domain mismatch between train and test data → **训练数据与测试数据的领域不匹配**
  - Maintaining context over longer text → **长文本的上下文保持**
  - Low-resource language pairs → **低资源语言对**
  - Failures to accurately capture sentence meaning → 无法准确捕捉句子含义
  - Pronoun (or zero pronoun) resolution errors → **代词（或零代词）消解错误**
  - Morphological agreement errors → **形态一致性错误**


#### 解释
这张幻灯片点明了**机器翻译仍未解决的核心挑战**，说明尽管NMT取得了巨大进步，但在诸多场景下仍有局限：

##### 1. 技术层面的挑战
- **未登录词**：遇到生僻词、新词（如网络热词）时，模型无法准确翻译；
- **领域不匹配**：训练数据是新闻领域，测试数据是医学领域时，翻译质量会大幅下降；
- **长文本上下文**：对于小说、论文等长文本，模型难以保持前后语义的一致性；
- **低资源语言对**：如非洲部落语言与英语的翻译，因平行语料极少，模型性能极差。


##### 2. 语义与语法层面的挑战
- **句子含义捕捉**：对于隐喻、双关等复杂语义，模型常理解偏差；
- **代词消解**：如“他说小明很努力，他会成功的”中，两个“他”的指代易混淆；
- **形态一致性**：在法语、德语等语言中，名词的性数、动词的时态语态一致性易出错。


简言之，机器翻译仍是一个“未完全解决”的开放问题，这些挑战也为学术界和工业界提供了持续的研究方向，推动技术不断迭代。



![image-20251029172027916](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029172027916.png)



- **回答**：Nope! → 没有！
- **难点**：Using common sense is still hard → **运用常识仍然很困难**
- **示例**：
  - 英语：*paper jam*（打印机卡纸）
  - 西班牙语翻译：*Mermelada de papel*（字面意为“纸果酱”）
- **图示**：
  - 左图：打印机卡纸的卡通图；
  - 右图：装着碎纸的玻璃罐，配问号，讽刺翻译的荒谬。


### 解释
这张幻灯片以**“常识运用不足”**为例，揭示机器翻译的核心短板之一：

“*paper jam*”是打印机领域的常用术语，意为“卡纸”。但机器翻译因缺乏“打印机工作原理、词汇多义性”的常识，错误地将“jam”理解为“果酱”，从而生成了“纸果酱（Mermelada de papel）”的荒谬翻译。

这一案例凸显了机器翻译的关键局限：**模型仅能基于统计规律学习词汇和语法的映射，却无法像人类一样理解“词汇在真实场景中的语义”，尤其是涉及领域知识、隐喻、多义词的场景，常识的缺失会导致翻译完全偏离原意**。

这类问题也推动了后续研究方向，例如结合知识图谱、引入外部常识库，或发展更具推理能力的大语言模型，以提升机器翻译的常识理解和语义准确性。

![image-20251029172138874](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029172138874.png)



- **回答**：Nope! → 没有！
- **难点**：NMT picks up **biases** in training data → ==神经机器翻译（NMT）会学习到训练数据中的偏见==
- **示例**：
  - 马来语：*Dia bekerja sebagai jururawat.*（未指定性别）→ 英语：*She works as a nurse.*（她是一名护士）
  - 马来语：*Dia bekerja sebagai pengaturcara.*（未指定性别）→ 英语：*He works as a programmer.*（他是一名程序员）
- **说明**：**Didn’t specify gender → 未指定性别**


#### 解释
这张幻灯片揭示了**神经机器翻译的“偏见学习”问题**，即模型会无意识地学习训练数据中隐含的社会偏见：

在示例中，马来语的“Dia”是中性代词（不区分性别），但NMT模型却将“护士”默认翻译为女性（*She*）、“程序员”默认翻译为男性（*He*）。这种偏差源于训练数据中“护士职业女性占比高、程序员职业男性占比高”的统计规律，模型将这种统计偏见内化为翻译规则，从而在未指定性别的情况下做出了带有性别倾向的翻译。

这类问题反映了NMT的核心局限：**模型仅能基于数据中的统计关联学习映射，无法辨别“统计规律”与“社会偏见”的边界**。这一挑战也推动了“去偏见NLP模型”的研究，例如通过数据增强、对抗训练等方法减少模型对有害偏见的学习。



![image-20251029172255356](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029172255356.png)

### 提取文字与翻译
- **标题**：So is Machine Translation solved? → 那么，机器翻译问题已经解决了吗？
- **回答**：Nope! → 没有！
- **难点**：**Uninterpretable systems do strange things** → ==不可解释的系统会做出奇怪的行为==
- **补充说明**：(But I think this problem has been fixed in Google Translate by 2021?) → （但我认为到2021年谷歌翻译已经修复了这个问题？）
- **示例**：
  - 索马里语输入（实际应为爱尔兰语）：*ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag ag*
  - 英语翻译：*As the name of the LORD was written in the Hebrew language, it was written in the language of the Hebrew Nation*（耶和华的名是用希伯来文写的，是用希伯来民族的语言写的）


#### 解释
这张幻灯片以**“不可解释的奇怪翻译行为”**为例，进一步说明机器翻译尚未完全解决的问题：

**输入的“ag”是爱尔兰语中的常见词素（如用于动词变位**），但模型却错误地将其与希伯来语宗教语境的内容关联，生成了完全不相关的翻译。这种“无厘头”的错误源于NMT模型的**黑箱特性**——它能学习到数据中的统计关联，却无法解释“为何将‘ag’映射到希伯来宗教文本”，一旦训练数据中存在噪声或小众关联，就可能产生这类奇怪输出。

尽管谷歌翻译等主流系统在2021年后通过模型迭代（如引入更鲁棒的训练策略、多语言知识融合）大幅减少了此类问题，但它仍反映了机器翻译“不可解释性”带来的隐患——工程师难以预判和调试模型的异常行为。

### NMT research continues 

神经机器翻译（NMT）研究仍在继续

![image-20251029172423774](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029172423774.png)

### 提取文字与翻译
- **标题**：NMT research continues → 神经机器翻译（NMT）研究仍在继续
- **定位**：NMT is a flagship task for NLP Deep Learning → **NMT是自然语言处理（NLP）深度学习领域的标杆任务**
- **研究贡献**：
  - NMT research has pioneered many of the recent innovations of NLP Deep Learning → NMT研究开创了NLP深度学习领域的诸多近期创新
- **2021年的研究态势**：
  - In 2021: NMT research continues to **thrive**(繁荣) → 2021年：NMT研究持续繁荣
    - Researchers have found many, many improvements to the “vanilla” seq2seq NMT system we’ve just presented → 研究者们已对我们刚刚介绍的“基础版”序列到序列（seq2seq）NMT系统提出了诸多改进
    - But we’ll present in a minute one improvement so integral that it is the new vanilla... → **但我们马上会介绍一项至关重要的改进，它已成为新的“基础版”……**
- **核心预告**：ATTENTION → 注意力（机制）


#### 解释
这张幻灯片承上启下，点明了**NMT在NLP深度学习领域的标杆地位**，并引出了后续的核心改进——**注意力机制（Attention）**：

NMT不仅是机器翻译的核心技术，更是NLP深度学习创新的“试验田”。从早期的seq2seq架构开始，NMT研究推动了诸多技术突破。到2021年，针对基础seq2seq NMT的改进已非常丰富，但其中**注意力机制**是最具革命性的一项——==它解决了早期seq2seq“固定长度语义编码瓶颈”的问题，让解码器能动态关注编码器的不同位置，从而大幅提升翻译质量和长文本处理能力==，最终成为NMT乃至整个NLP领域的“新基础组件”。

> ##### 一、早期 seq2seq 的 “固定长度语义编码瓶颈”
>
> 早期 seq2seq 模型的结构是 **“编码器（Encoder）→ 解码器（Decoder）”**：
>
> - **编码器**：将输入的源语言序列（如法语句子）编码成一个**固定长度的向量**（称为 “上下文向量”），用来概括整个源序列的语义；
> - **解码器**：基于这个固定长度的向量，逐词生成目标语言序列（如英语句子）。
>
> 这个设计的核心缺陷是：**无论源序列多长，语义信息都要压缩到一个固定长度的向量中**。
>
> 举个例子：
>
> - 源序列是短句子 *“il m'entarté”*（他用馅饼砸了我），编码后的向量能容纳足够的语义；
> - 源序列是长文本（如一段小说、一篇论文），编码后的向量会因 “容量不足” 丢失大量细节，导致解码器生成的译文语义残缺、逻辑混乱。
>
> 这种 “把长序列硬塞进固定长度向量” 的限制，就是 **“固定长度语义编码瓶颈”**—— 长文本的语义信息无法被充分保留，翻译质量随文本长度急剧下降。

这一铺垫也说明，注意力机制的出现是NMT技术迭代的关键里程碑，为理解后续基于Transformer的更先进模型（如GPT、BERT）奠定了基础。

## Assignment 4: Cherokee-English machine translation! 

作业4：切罗基语-英语机器翻译！

![image-20251029174454867](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029174454867.png)



- **标题**：Assignment 4: Cherokee-English machine translation! → 作业4：切罗基语-英语机器翻译！
- **语言现状**：
  - Cherokee is an endangered Native American language – about 2000 fluent speakers → 切罗基语是一种濒危的美洲原住民语言——约有2000名流利使用者
- **资源情况**：
  - Extremely low resource: About 20k parallel sentences available, most from the bible → 资源极度匮乏：约有2万条平行语句可用，大部分来自《圣经》
- **文本示例**：
  - 切罗基语文本（ syllabary 书写系统）
  - 对应英语翻译：*Long ago were seven boys who used to spend all their time down by the townhouse playing games, rolling a stone wheel along the ground, sliding and striking it with a stick*（很久以前有七个男孩，他们过去总在市政厅附近消磨时间，玩游戏、在地上滚石轮、用棍子滑动并击打它）
- **书写系统**：
  - Writing system is a syllabary of symbols for each CV unit (85 letters) → **书写系统是一种音节文字，每个辅音-元音（CV）单元对应一个符号（共85个字母）**
- **致谢**：
  - Many thanks to Shiyue Zhang, Benjamin Frey, and Mohit Bansal from UNC Chapel Hill for the resources for this assignment! → 非常感谢北卡罗来纳大学教堂山分校的张诗悦、本杰明·弗雷和莫希特·班萨尔为本次作业提供的资源！
- **技术现状**：
  - Cherokee is not available on Google Translate! 😭 → 谷歌翻译不支持切罗基语！😭


#### 解释
这张幻灯片介绍了**“切罗基语-英语机器翻译”作业的背景**，它是“低资源语言翻译”研究的典型场景：

切罗基语是濒危语言，使用者少、平行语料极少（仅约2万条，且多来自《圣经》），属于**“低资源语言对”**。这类语言的机器翻译是NMT的核心挑战之一——因数据不足，传统NMT模型难以学习到足够的翻译模式。

同时，切罗基语的书写系统是“音节文字”（每个符号对应一个辅音-元音单元），这也增加了建模的复杂度。而“谷歌翻译不支持该语言”的现状，进一步凸显了低资源语言翻译的研究价值——通过作业这类实践，研究者可探索“数据增强、迁移学习、多语言共享编码”等方法，为濒危语言的机器翻译提供技术支持，助力语言保护。

简言之，该作业以切罗基语为例，让学习者直面“低资源、特殊书写系统”的机器翻译挑战，是NMT在实际场景中“技术落地与社会价值”结合的体现。

![image-20251029175015541](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029175015541.png)



- **标题**：Cherokee → 切罗基（族/语）
- **历史与分布**：
  - Cherokee originally lived in western North Carolina and eastern Tennessee → 切罗基人最初居住在北卡罗来纳州西部和田纳西州东部
  - Most speakers now in Oklahoma, following the Trail of Tears; some in NC → 经历“血泪之路”后，多数使用者现居俄克拉荷马州；部分在北卡罗来纳州
- **书写系统**：
  - Writing system Invented by Sequoyah around 1820 – someone who was previously illiterate → 书写系统由**塞阔雅（Sequoyah）**于1820年左右发明——他此前是文盲
  - Very effective: In the following decades Cherokee literacy was higher than for white people in the southeastern United States → **非常有效：在接下来的几十年里，切罗基人的识字率高于美国东南部的白人**
- **资源链接**：https://www.cherokee.org
- **图示**：
  - 右上：塞阔雅的画像（手持切罗基语书写文本）；
  - 右下：“血泪之路”地图（展示切罗基人被迫迁徙的路线）；
  - 左下：切罗基族毕业生（体现语言文化传承）；
  - 中下：切罗基族居民（展现当代族群生活）


### 解释
这张幻灯片补充了**切罗基语的文化、历史背景**，为理解其“濒危性”和“机器翻译研究价值”提供了语境：

切罗基语的书写系统是由文盲塞阔雅发明的音节文字，这一创新让切罗基人的识字率一度超越美国东南部的白人，对文化传承意义重大。但“血泪之路”（印第安人被迫迁徙的暴行）导致族群分散，语言使用者锐减，成为濒危语言。

在这样的背景下，“切罗基语-英语机器翻译”的研究不仅是技术挑战，更承载着**“通过AI助力濒危语言保护、文化传承”**的社会价值——机器翻译可帮助少数族裔更便捷地学习、使用本族语言，也能让外界更易接触其文化遗产（如《圣经》外的文本）。

简言之，**该内容将“低资源机器翻译”的技术问题与“语言濒危、文化传承”的社会议题结合，凸显了NMT研究在人文领域的深层意义。**

## Section 3: Attention

![image-20251029180405665](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029180405665.png)



- **标题**：Sequence-to-sequence: the bottleneck problem → 序列到序列：瓶颈问题
- **模块说明**：
  - Encoding of the source sentence. → 源语句的编码
  - Source sentence (input) → 源语句（输入）：*il a m’entarté*（法语，对应英语“he hit me with a pie”）
  - Target sentence (output) → 目标语句（输出）：*he hit me with a pie <END>*
  - Encoder RNN → 编码器循环神经网络（RNN）
  - Decoder RNN → 解码器循环神经网络（RNN）
- **问题提示**：Problems with this architecture? → **这种架构的问题是什么？**
- “Encoding of the source sentence. This needs to capture all information about the source sentence. Information bottleneck!” 译为 “**源语句的编码。这需要捕捉源语句的所有信息。信息瓶颈！**”
- 它指出，编码器必须把源语句的**全部语义信息**压缩到一个固定长度的向量中，但这个向量的 “容量” 有限，无法承载长文本或复杂语义的所有细节，从而形成了 “信息瓶颈”—— 这也是早期 seq2seq 在长文本翻译、复杂语义理解上表现不佳的根本原因。


#### 解释
这张图展示了**早期序列到序列（seq2seq）模型的架构**，==其核心是“编码器RNN将源语句编码为一个固定长度的向量，解码器RNN基于该向量生成目标语句”。==

#### 架构的核心缺陷（瓶颈问题）
如前所述，这种设计存在**“固定长度语义编码瓶颈”**：
- 编码器必须将整个源语句（无论长短）的语义压缩到**一个固定维度的向量**中；
- 解码器生成每个目标词时，仅能依赖这个单一向量，导致长文本的语义信息大量丢失，翻译质量急剧下降。

==例如，若源语句是一篇长文章，编码器的固定向量无法容纳足够细节，解码器生成的译文会出现语义断裂、逻辑混乱的问题。==

这一缺陷也直接推动了**注意力机制（Attention）**的诞生——让解码器能动态关注编码器的不同位置，从而突破“固定向量瓶颈”，提升长文本翻译的质量。

### Attention

![image-20251029180707538](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029180707538.png)

- **核心价值**：
  - Attention provides a solution to the bottleneck problem. → 注意力机制为瓶颈问题提供了解决方案
- **核心思想**：
  - Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence → **核心思想**：==在解码器的每一步，通过与编码器的直接连接，聚焦到源序列的特定部分.==
- **图示与公式说明**：
  - First, we will show via diagram (no equations), then we will show with equations → 首先我们将通过图示（无公式）展示，之后再结合公式说明
- **图示隐喻**：放大镜（象征“聚焦”的核心逻辑）


#### 解释
这张幻灯片是**注意力机制的入门介绍**，核心是阐明其“解决seq2seq瓶颈、动态聚焦源序列”的设计逻辑：

注意力机制的出现，是为了突破早期seq2seq“固定长度语义编码瓶颈”。它的核心创新是**让解码器在生成每个词时，都能直接“关注”编码器中与当前翻译最相关的源序列部分**，而非依赖单一的固定向量。

用“放大镜”的隐喻理解：**解码器生成每个词时，就像用放大镜“聚焦”到源序列的某一段（比如生成“hit”时，聚焦到源序列中表示“动作”的部分），从而精准获取当前翻译所需的语义细节。**



![image-20251029180844643](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029180844643.png)



![image-20251029180859253](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029180859253.png)

**这个过程是Q\*K**

![image-20251029180909072](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029180909072.png)

**这个过程是softmax(Q\*K)**

- **说明框 1**：*On this decoder timestep, we’re mostly focusing on the first encoder hidden state (“he”)* → ==在当前解码器时间步，我们主要聚焦于第一个编码器隐藏状态（“he”）==
- **说明框 2**：*Take softmax to turn the scores into a probability distribution* → ==用 softmax 将得分转换为概率分布==

![image-20251029181156683](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029181156683.png)

- *Use the attention distribution to take a weighted sum of the encoder hidden states. The attention output mostly contains information from the hidden states that received high attention.* → **利用注意力分布对编码器隐藏状态进行加权求和。**注意力输出主要包含获得高注意力的隐藏状态的信息。


#### 加权求和生成上下文向量（Context Vector）
用注意力权重对编码器所有隐藏状态进行**加权求和**，得到当前解码器时间步的上下文向量 $ c_t $：
$$c_t = \sum_{i=1}^N \alpha_{ti} \cdot h_i$$




![image-20251029181626267](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029181626267.png)

*Concatenate attention output with decoder hidden state, then use to compute \*y*^1 as before* → 将注意力输出与解码器隐藏状态拼接，然后像之前一样用于计算 *y*^1

*Sometimes we take the attention output from the previous step, and also feed it into the decoder (along with the usual decoder input). We do this in Assignment 4.* → **有时我们会将上一步的注意力输出也输入到解码器中（与常规解码器输入一起）**。我们在作业 4 中会这样做。

---

我们以“翻译法语 *il a m’entarté* 为英语 *he hit me with a pie*”中生成第一个词 *he* 为例，具体看看这个过程：

##### 步骤1：编码器生成隐藏状态
编码器 RNN 处理源句 *il a m’entarté* 后，生成一系列隐藏状态 $ h_1, h_2, h_3, h_4 $，分别对应单词 *il, a, m’, entarté*。

##### 步骤2：解码器初始化与注意力计算
解码器以 `<START>` 为起始输入，生成初始隐藏状态 $ s_0 $。然后计算注意力得分、分布，并加权求和得到**注意力输出（上下文向量）** $ c_1 $。此时 $ c_1 $ 主要包含 $ h_1 $（对应 *il*，语义上与 *he* 强相关）的信息。

##### 步骤3：拼接并预测目标词
- **拼接**：将注意力输出 $ c_1 $ 与解码器当前隐藏状态 $ s_1 $（由 $ s_0 $ 和 `<START>` 生成）进行拼接，得到一个新的向量 $ [s_1; c_1] $（“;” 表示拼接）。
- **预测**：将这个拼接后的向量输入到全连接层（或其他预测模块），经过激活函数（如 softmax）后，输出词汇表中每个单词的概率。其中概率最高的单词就是预测的 $ \hat{y}_1 $，也就是 *he*。

简单来说，就是“把源句里和当前翻译最相关的信息（注意力输出），跟解码器自己的生成状态（隐藏状态）结合起来，一起判断下一个该说啥词”。

![image-20251029183116358](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251029183116358.png)