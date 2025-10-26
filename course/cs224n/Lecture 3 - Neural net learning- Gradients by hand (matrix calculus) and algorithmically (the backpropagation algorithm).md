---
created: 2025-01-31T17:08
updated: 2025-02-01T00:20
---
神经网络学习：手动计算梯度（矩阵微积分）和通过算法计算梯度（反向传播算法）

修改
[[矩阵求导的本质与分子布局、分母布局的本质（矩阵求导——本质篇）]]

## 1. Introduction
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131171937.png)

### Name Entity Recognition
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131172217.png)
**命名实体识别** Named Entity Recognition (NER)  
- The task: find and classify(查找并分类) names in text, for example:
	- Last night, Paris Hilton wowed in a sequin gown. 
	-                     PER PER 
- Possible uses(**用途**): 
	- Tracking mentions of particular entities in documents  跟踪文档中特定实体的提及情况
	- For question answering, answers are usually named entities 对于问答任务，答案通常是命名实体
- Often followed by Named Entity Linking/Canonicalization into Knowledge Base 通常紧接着进行命名实体链接/规范化到知识库中
1. **任务定义**：命名实体识别是自然语言处理中的一项基础任务，<mark style="background: #D2B3FFA6;">旨在从文本里找出特定的名称，并将它们分类为不同的实体类别，常见的类别包括人名（PER）、地点（LOC）、日期（DATE）等。</mark>例如在“Last night, Paris Hilton wowed in a sequin gown.”中，识别出“Paris Hilton”为两个人名实体；在“Samuel Quinn was arrested in the Hilton Hotel in Paris in April 1989.”中，“Samuel Quinn”是人名，“Hilton Hotel”“Paris”是地点，“April 1989”是日期。 
2. **用途** - **文档实体跟踪**：可以用于监控和记录文档中特定实体被提及的情况，有助于对文档内容进行结构化分析，比如在新闻报道分析中跟踪特定人物、机构等实体的出现频率和上下文。 - **问答系统**：在问答任务里，很多时候答案就是命名实体，准确识别命名实体有助于提取问题的答案，例如<mark style="background: #D2B3FFA6;">问“谁写了《哈姆雷特》？”，答案“莎士比亚”是人名实体，NER可为问答系统提供基础的信息提取支持。</mark> 3. **后续任务**：命名实体识别之后，通常会进行命名实体链接或规范化操作，即将识别出的实体与知识库中的对应实体进行关联或统一规范表示，以便进一步的知识检索、推理等操作。
### 简单命名实体识别：使用二元逻辑回归分类的窗口分类法 Simple NER: Window classification using binary logistic classifier
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131173033.png)

- Idea: classify each word in its context window of neighboring words 在相邻词的上下文窗口中对每个词进行分类 
- Train **logistic classifier on hand - labeled data** to classify center word {yes/no} for each class based on a concatenation of word vectors in a window 基于窗口中词向量的拼接，在手标数据上**训练逻辑回归分类器**，以对每个类别的中心词分类为{是/否} 
	- Really, we <mark style="background: #D2B3FFA6;">usually use multi - class softmax</mark>, but trying to keep it simple 😊通常使用多分类softmax
	- Example: Classify “Paris” as +/- location in context of sentence with window length 2: the museums in Paris are amazing to see. $X_{window}=[x_{museums}\ x_{in}\ x_{Paris}\ x_{are}\ x_{amazing}]^{T}$ 
	- Resulting vector $x_{window}=x\in R^{5d}$, a column vector!  **$5d$ 维的列向量（$d$ 为单个词向量的维度）**
	- To classify all words: run classifier for each class on the vector centered on each word in the sentence 针对句子中以每个词为中心的向量，为每个类别运行分类器 
1. **训练方式**：通过手标数据来训练逻辑回归分类器。具体做法是将窗口内的词向量进行拼接，然后基于拼接后的向量判断中心词是否属于某个类别（是/否）。虽然实际中常用多分类softmax，但这里为简化说明采用二元逻辑回归。 
2.  **示例说明**：以句子“the museums in Paris are amazing to see.”为例，当窗口长度为2时，对于“Paris”，其窗口内的词为“the”“museums”“in”“Paris”“are”“amazing”，将这些词的词向量拼接成向量 $X_{window}$，它是一个 $5d$ 维的列向量（$d$ 为单个词向量的维度）。  ^9ef350
3. **整体分类操作**：要对句子中的所有词进行分类，就<mark style="background: #D2B3FFA6;">针对每个词构建其对应的上下文窗口向量，然后针对每个类别运行分类器，从而判断每个词所属的命名实体类别。</mark> 这种方法利用词的上下文信息，通过简单的分类器训练来实现命名实体识别，是一种较为基础的NER实现思路。

### 命名实体识别：中心词是否为location的二分类 NER: Binary classification for center word being location
![image.png|699](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131174225.png)
 - We do **supervised training** and want high score if it’s a location 我们进行有监督训练，并且如果（中心词）是地点，希望得到高分。$$ \displaylines{
 J_{t}(\theta)=\sigma(s)=\frac{1}{1 + e^{-s}} \\
 s = u^{T}h \\
 h = f(Wx + b) \\
 x\ (\text{input}) \\
 x = [x_{museums}\ x_{in}\ x_{Paris}\ x_{are}\ x_{amazing}]
 }$$
 1. **任务概述**：这是在命名实体识别（NER）中，针对中心词是否为地点进行的二分类任务。采用有监督训练的方式，目标是当中心词确实是地点时，分类器能给出较高的分数。 
 2. **核心公式** 
	 - $J_{t}(\theta)=\sigma(s)=\frac{1}{1 + e^{-s}}$ ：这是logistic函数（sigmoid函数），用于将输入值 $s$ 转换为一个介于0和1之间的概率值，表示模型预测中心词属于地点类别的概率。 
	 - $s = u^{T}h$ ：$s$ 是一个标量，通过将向量 $u$ 的转置与向量 $h$ 做点积得到，<mark style="background: #D2B3FFA6;">这里 $u$ 是模型的参数向量(u是其他向量)</mark>，$h$ 是中间隐藏层的输出向量。 
	 - $h = f(Wx + b)$ ：$h$ 是隐藏层的输出，$x$ 是输入向量（如句子中以中心词为核心的窗口内词向量拼接而成的向量，示例中 $x = [x_{museums}\ x_{in}\ x_{Paris}\ x_{are}\ x_{amazing}]$ ），$W$ 是权重矩阵，$b$ 是偏置向量，$f$ 是激活函数，用于对 $Wx + b$ 的结果进行非线性变换。 
3. **模型流程**：输入向量 $x$ 首先经过线性变换 $Wx + b$ ，再通过激活函数 $f$ 得到隐藏层输出 $h$ ，$h$ 与参数向量 $u$ 做点积得到 $s$ ，最后通过logistic函数将 $s$ 转换为预测的概率值 $J_{t}(\theta)$ ，以此判断中心词是否为地点。 这是一个简单的基于神经网络结构的二分类模型在命名实体识别中判断地点类别的应用。

### 记住：随机梯度下降 Remember : Stochastic Gradient Descent
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131175407.png)
### 英文原文 Remember: Stochastic Gradient Descent 
Update equation: $$ \displaylines{ \theta^{new}=\theta^{old}-\alpha\nabla_{\theta}J(\theta)\\ \alpha = \text{step size or learning rate}
}$$ i.e., for each parameter: $\theta_{j}^{new}=\theta_{j}^{old}-\alpha\frac{\partial J(\theta)}{\partial\theta_{j}^{old}}$
In deep learning, we update the data representation (e.g., word vectors) too! <mark style="background: FFFF00;">更新数据表示（例如，词向量）</mark>
How can we compute $\nabla_{\theta}J(\theta)$? 
1. By hand  手动
2. Algorithmically: the backpropagation algorithm  反向传播算法

 1.**梯度计算方法**：计算目标函数关于参数的梯度$\nabla_{\theta}J(\theta)$ 有两种常见方式。一是手动计算，通过数学推导和求导法则直接计算梯度；二是使用反向传播算法，这是一种高效的算法，尤其适用于深度神经网络，通过链式法则自动计算梯度，从而实现参数的更新和模型的优化。

### 手动计算梯度Computing Gradients by Hand
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131183931.png)
 - Matrix calculus: Fully vectorized gradients  矩阵微积分：完全向量化的梯度
 - “Multivariable calculus is just like single - variable calculus if you use matrices” - Much faster and more useful than non - vectorized gradients 
 - But doing a non - vectorized gradient can be good for intuition; recall the first lecture for an example
 - **Lecture notes and matrix calculus notes cover this material in more detail** 
 - **You might also review Math 51, which has a new online textbook: http://web.stanford.edu/class/math51/textbook.html or maybe you’re luckier if you did Engr 108** 
  “如果你使用矩阵，多变量微积分就和单变量微积分很像” 
  - 比非向量化的梯度快得多且更有用 
  - 但计算非向量化的梯度有助于形成直观理解；
  - 回想一下第一讲中的例子 - 课程笔记和矩阵微积分笔记更详细地涵盖了这些内容 
  - 你也可以复习数学51课程，它有一本新的在线教科书：http://web.stanford.edu/class/math51/textbook.html ，或者如果你学过工程108课程，可能会更幸运 
  1. **向量化梯度优势**：<mark style="background: #D2B3FFA6;">完全向量化的梯度相比非向量化的梯度，在计算速度和实用性方面更具优势，能够更高效地处理多变量函数的梯度计算问题。 </mark>
  2.  **非向量化梯度价值**：虽然向量化梯度有诸多优点，<mark style="background: #D2B3FFA6;">但计算非向量化的梯度对于培养直观的数学理解有帮助</mark>，可回顾第一讲的例子来体会。 
### 梯度 Gradients
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131184633.png)
“如果我们对输入做一点改变，输出会改变多少？” 
在 $x = 1$ 时，它的变化大约是输入变化的 3 倍：$1.01^{3}=1.03$ 
在 $x = 4$ 时，它的变化大约是输入变化的 48 倍：$4.01^{3}=64.48$   (64.48-64)/0.01 大约=48
![image.png|630](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131185103.png)

### Jacobian Matrix: Generalization of the Gradient
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131185323.png)
### 雅可比矩阵：梯度的推广 Jacobian Matrix: Generalization of the Gradient 

> 神经网络通常涉及<mark style="background: #D2B3FFA6;">多输入多输出的函数关系。例如，在多层神经网络中，一层的输入是多个神经元的输出（多输入），经过该层的计算后又产生多个神经元的输出（多输出）。雅可比矩阵能够很好地描述这种多输入多输出函数的变化情况。它是一个由函数的各个输出关于各个输入的偏导数组成的矩阵，通过雅可比矩阵可以清晰地了解到输入的微小变化如何影响输出，这对于分析神经网络的行为至关重要。</mark>
- Given a function with $m$ outputs and $n$ inputs 给定一个具有 $m$ 个输出和 $n$ 个输入的函数
$$f(x)=[f_{1}(x_{1},x_{2},\ldots,x_{n}),\ldots,f_{m}(x_{1},x_{2},\ldots,x_{n})]$$
- It’s Jacobian is an $m\times n$ matrix of partial derivatives 它的雅可比矩阵是一个由偏导数组成的 $m\times n$ 矩阵
$$\frac{\partial f}{\partial x}=\begin{bmatrix}\frac{\partial f_{1}}{\partial x_{1}}&\cdots&\frac{\partial f_{1}}{\partial x_{n}}\\\vdots&\ddots&\vdots\\\frac{\partial f_{m}}{\partial x_{1}}&\cdots&\frac{\partial f_{m}}{\partial x_{n}}\end{bmatrix} \ \ \  (\frac{\partial f}{\partial x})_{ij}=\frac{\partial f_{i}}{\partial x_{j}}$$
1. **函数设定**：考虑一个多输入多输出的函数 $f(x)$ ，其中有 $n$ 个输入变量 $x_1,x_2,\ldots,x_n$ ，产生 $m$ 个输出 $f_1,f_2,\ldots,f_m$ 。这种函数在多元函数和机器学习等领域常见，比如在神经网络中，输入层到隐藏层或输出层的映射可以看作是这种多输入多输出的函数关系。 
2. **雅可比矩阵定义**：雅可比矩阵是对该函数的一种描述，它是一个 $m\times n$ 的矩阵，矩阵中的元素是函数的各个输出关于各个输入的偏导数。具体来说，**矩阵中第 $i$ 行第 $j$ 列的元素 $(\frac{\partial f}{\partial x})_{ij}$ 等于 $\frac{\partial f_{i}}{\partial x_{j}}$ ，即第 $i$ 个输出函数关于第 $j$ 个输入变量的偏导数。** 
3. **意义和应用**：雅可比矩阵可以理解为梯度概念的推广，在单变量函数中，梯度就是导数，描述函数的变化率；在多输入多输出的多元函数中，雅可比矩阵描述了函数在各个方向上的变化情况，在求解多元函数的极值、优化问题以及在神经网络的反向传播算法等方面都有重要应用，它能够帮助我们了解输入的微小变化如何影响输出。
### 链式法则 Chain Rule
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131190035.png)

### Example Jacobian: 逐元素激活函数 Elementwise activation Function
![image.png|544](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131191748.png)
$$\displaylines{h = f(z),\text{ what is }\frac{\partial h}{\partial z}?\quad h,z\in\mathbb{R}^{n}\\
h_{i}=f(z_{i}) \\
\begin{align*} \left(\frac{\partial h}{\partial z}\right)_{ij}&=\frac{\partial h_{i}}{\partial z_{j}}=\frac{\partial}{\partial z_{j}}f(z_{i})\\ 
&=\begin{cases} f^{\prime}(z_{i})&\text{if }i = j\\ 0&\text{if otherwise} \end{cases} regular\ 1-variable\ derivative(常规的单变量梯度)\end{align*} \\
\frac{\partial h}{\partial z}=\begin{pmatrix} f^{\prime}(z_{1})&0\\ 0&\ddots&\\ 0&f^{\prime}(z_{n}) \end{pmatrix}=\text{diag}(f^{\prime}(z)) 
}$$
1. **问题设定**：考虑一个逐元素的激活函数 $h = f(z)$ ，其中 $h$ 和 $z$ 都是 $n$ 维向量，即 $h,z\in\mathbb{R}^{n}$ ，并且 $h$ 的每个元素 $h_{i}$ 是 $z$ 的对应元素 $z_{i}$ 通过函数 $f$ 计算得到，即 $h_{i}=f(z_{i})$ 。目标是求解 $h$ 关于 $z$ 的雅可比矩阵 $\frac{\partial h}{\partial z}$ 。
2. **雅可比矩阵元素计算**：根据雅可比矩阵的定义，其元素 $\left(\frac{\partial h}{\partial z}\right)_{ij}$ 等于 $\frac{\partial h_{i}}{\partial z_{j}}$ 。因为 $h_{i}=f(z_{i})$ ，所以当 $j = i$ 时，$\frac{\partial h_{i}}{\partial z_{j}}=\frac{\partial}{\partial z_{i}}f(z_{i})$ ，这就是一元函数 $f$ 在 $z_{i}$ 处的导数 $f^{\prime}(z_{i})$ ；当 $j\neq i$ 时，$h_{i}$ 与 $z_{j}$ 无关，所以 $\frac{\partial h_{i}}{\partial z_{j}} = 0$ 。 
> 解释: **$\frac{\partial h}{\partial z}$ ：由于 $h = f(z)$ ，其中 $f$ 是激活函数，根据之前关于逐元素激活函数雅可比矩阵的结论，$\frac{\partial h}{\partial z}=\text{diag}(f^{\prime}(z))$ 。$f^{\prime}(z)$ 是激活函数 $f$ 对 $z$ 的导数，$\text{diag}(f^{\prime}(z))$ 是一个对角矩阵，其对角元素为 $f^{\prime}(z)$ 的各个元素。** 

> 假设激活函数 $f(z)$ 是 sigmoid 函数：$f(z)=\frac{1}{1 + e^{-z}}$ ，其导数为 $f^{\prime}(z)=f(z)(1 - f(z))$ 。 设 $z$ 是一个三维向量 $z = [z_1,z_2,z_3]^T$ ，那么： 首先计算 $f(z)$ ： 
> - $h_1 = f(z_1)=\frac{1}{1 + e^{-z_1}}$ 
> - $h_2 = f(z_2)=\frac{1}{1 + e^{-z_2}}$ 
> - $h_3 = f(z_3)=\frac{1}{1 + e^{-z_3}}$ 所以 $h = [h_1,h_2,h_3]^T = [f(z_1),f(z_2),f(z_3)]^T$ 。 然后计算 $f^{\prime}(z)$ ： 
> - $f^{\prime}(z_1)=f(z_1)(1 - f(z_1))=\frac{1}{1 + e^{-z_1}}(1-\frac{1}{1 + e^{-z_1}})$
> - $f^{\prime}(z_2)=f(z_2)(1 - f(z_2))=\frac{1}{1 + e^{-z_2}}(1-\frac{1}{1 + e^{-z_2}})$ 
> - $f^{\prime}(z_3)=f(z_3)(1 - f(z_3))=\frac{1}{1 + e^{-z_3}}(1-\frac{1}{1 + e^{-z_3}})$ 根据 $\frac{\partial h}{\partial z}=\text{diag}(f^{\prime}(z))$ ，此时 $\frac{\partial h}{\partial z}$ 是一个 $3\times3$ 的对角矩阵：$$  \frac{\partial h}{\partial z}=\begin{bmatrix} f^{\prime}(z_1) & 0 & 0 \\ 0 & f^{\prime}(z_2) & 0 \\ 0 & 0 & f^{\prime}(z_3) \end{bmatrix} $$ 即：$$ \frac{\partial h}{\partial z}=\begin{bmatrix} \frac{1}{1 + e^{-z_1}}(1-\frac{1}{1 + e^{-z_1}}) & 0 & 0 \\ 0 & \frac{1}{1 + e^{-z_2}}(1-\frac{1}{1 + e^{-z_2}}) & 0 \\ 0 & 0 & \frac{1}{1 + e^{-z_3}}(1-\frac{1}{1 + e^{-z_3}}) \end{bmatrix} $$ 这个例子展示了对于一个具体的激活函数（sigmoid 函数）和给定维度的输入向量 $z$ ，如何得到 $\frac{\partial h}{\partial z}=\text{diag}(f^{\prime}(z))$ 这种对角矩阵形式的雅可比矩阵。
><mark style="background: #D2B3FFA6;"> 恰好这里的 f都是一样的激活函数。</mark>

1. **雅可比矩阵形式**：综合上述计算，$h$ 关于 $z$ 的雅可比矩阵 $\frac{\partial h}{\partial z}$ 是一个对角矩阵，其对角线上的元素分别是 $f^{\prime}(z_{1}),f^{\prime}(z_{2}),\cdots,f^{\prime}(z_{n})$ ，可以表示为 $\text{diag}(f^{\prime}(z))$ 。在神经网络中，激活函数常以这种逐元素的方式作用于输入向量，计算其雅可比矩阵对于理解函数的局部变化性质以及在反向传播算法中计算梯度等方面具有重要作用。

### 其他雅可比行列式 other Jacobian
![image.png|681](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131191748.png)
$$
\begin{align*} \frac{\partial}{\partial x}(Wx + b)&=W\\ \frac{\partial}{\partial b}(Wx + b)&=I\ (\text{Identity matrix})\\ \frac{\partial}{\partial u}(u^{T}h)&=h^{T} \end{align*} 
$$
**Fine print: This is the correct Jacobian. Later we discuss the “shape convention”; using it the answer would be $h$.** 
**细节说明：这是正确的雅可比矩阵形式。稍后我们将讨论“形状约定”；根据它答案可能是 $h$ 。**
1. **第一个式子**：$\frac{\partial}{\partial x}(Wx + b)=W$ 。这里 $W$ 是矩阵，$x$ 和 $b$ 是向量。$Wx + b$ 是一个线性变换，对其关于 $x$ 求雅可比矩阵。从矩阵微积分角度，$Wx + b$ 对 $x$ 的每一个元素求偏导，根据线性代数和矩阵求导规则，结果就是矩阵 $W$ 。在神经网络中，这类似于输入 $x$ 经过线性层 $Wx + b$ 变换后，求该变换关于输入 $x$ 的导数，用于反向传播时计算梯度。 
2. **第二个式子**：$\frac{\partial}{\partial b}(Wx + b)=I$ ，**其中 $I$ 是单位矩阵。因为 $Wx$ 与 $b$ 无关，对 $Wx + b$ 关于 $b$ 求偏导，$b$ 每一个元素的偏导数在对应位置为 $1$ ，其他位置为 $0$ ，所以结果是单位矩阵。** 这在神经网络训练中，对于更新偏置 $b$ 的梯度计算很重要。 
3. **第三个式子**：$\frac{\partial}{\partial u}(u^{T}h)=h^{T}$ ，$u$ 和 $h$ 是向量，$u^{T}h$ 是内积运算。根据向量求导规则，对 $u^{T}h$ 关于 $u$ 求导得到 $h^{T}$ 。注释提到根据“形状约定”答案可能是 $h$ ，这涉及到在不同的矩阵向量求导的形状约定下，结果的表示形式可能有所不同，后续会讨论该约定。这些求导结果在神经网络的反向传播算法中用于计算各个参数的梯度，以更新模型参数。
### Back to our Neural Net!
![image.png|670](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131192504.png)
**实际上，我们关心的是损失 $J_{t}$ 的梯度，但为了简化(simplicity)，我们将计算分数的梯度 。**

#### 1. Break up equations into simple pieces
![image.png|589](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131192939.png)
仔细定义你的变量并留意它们的维度！
#### 2.Apply the chain rule
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131192939.png)
#### 3.Write out Jacobian
理解
[[矩阵求导的本质与分子布局、分母布局的本质（矩阵求导——本质篇）]]
![image.png|694](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131193542.png)

<mark style="background: #D2B3FFA6;">使用的是分子布局，在分子布局中，标量对向量求导结果为行向量</mark>
$$
\frac{\partial (u^Th)} {\partial u} = \frac{\partial (h^Tu)} {\partial u} =h^T
$$
而对h求导
$$
\frac{\partial (u^Th)} {\partial h} = \frac{\partial (h^Tu)} {\partial h} = u^T
$$

$$
$$
$$\displaylines{ \frac{\partial s}{\partial b}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial b} \\ =u^{T}\text{diag}(f^{\prime}(z))I \\ =u^{T}\circ f^{\prime}(z)}  $$
> $\circ$是Hadamard积，是矩阵逐元素相乘
> [[矩阵相关#^fab6ec]]
> $f^{\prime}(z)=[1,5,7]$把对角矩阵$diag(f ^{\prime)}(z)$的元素展成了行向量，因此可以使用Hadamard积。

1. **各部分导数计算** - 由 $\frac{\partial}{\partial u}(u^{T}h)=h^{T}$ ，因为 $s = u^{T}h$ ，所以 $\frac{\partial s}{\partial h}=u^{T}$ 。 
	- 对于 $h = f(z)$ ，根据之前得出的 $\frac{\partial}{\partial z}(f(z))=\text{diag}(f^{\prime}(z))$ ，可得 $\frac{\partial h}{\partial z}=\text{diag}(f^{\prime}(z))$ ，这里 $f^{\prime}(z)$ 是激活函数 $f$ 的导数，$\text{diag}(f^{\prime}(z))$ 是由 $f^{\prime}(z)$ 构成的对角矩阵。 
	- 由于 $z = Wx + b$ ，根据 $\frac{\partial}{\partial b}(Wx + b)=I$ （$I$ 为单位矩阵），所以 $\frac{\partial z}{\partial b}=I$ 。 
2.  **最终结果**：将上述各部分导数相乘，得到 $\frac{\partial s}{\partial b}=u^{T}\text{diag}(f^{\prime}(z))I$ ，又因为单位矩阵 $I$ 与矩阵相乘不改变矩阵，所以进一步简化为 $\frac{\partial s}{\partial b}=u^{T}\circ f^{\prime}(z)$ 。此结果在神经网络训练中计算关于偏置 $b$ 的梯度时非常关键，有助于通过随机梯度下降等优化算法更新偏置参数，以最小化损失函数。
> 要理解为什么$u^{T}\text{diag}(f^{\prime}(z))$可以写成$u^{T}\circ f^{\prime}(z)$，需要从矩阵运算和两者的特性来分析，以下为你展开介绍： 
> - **$\text{diag}(f^{\prime}(z))$的特性**：$\text{diag}(f^{\prime}(z))$是一个对角矩阵，其对角线上的元素是$f^{\prime}(z)$的各个元素，而其他非对角元素均为$0$。假设$f^{\prime}(z) = [a_1, a_2, \cdots, a_n]$，那么$\text{diag}(f^{\prime}(z))$就是一个$n\times n$的矩阵，形式为$\begin{bmatrix}a_1 & 0 & \cdots & 0\\0 & a_2 & \cdots & 0\\\vdots & \vdots & \ddots & \vdots\\0 & 0 & \cdots & a_n\end{bmatrix}$。 
> - $u^{T}\text{diag}(f^{\prime}(z))$的计算**：设$u^{T}=[u_1,u_2,\cdots,u_n$，当$u^{T}$与$\text{diag}(f^{\prime}(z))$相乘时，根据矩阵乘法规则，结果是一个$1\times n$的行向量。具体计算为$[u_1a_1, u_2a_2, \cdots, u_na_n]$。 
> - **与逐元素乘积的等价性**：从上面的计算结果可以看出，$u^{T}\text{diag}(f^{\prime}(z))$得到的结果正好是$u^{T}\)与\(f^{\prime}(z)$对应元素相乘的形式，这与逐元素乘积（哈达玛积）$u^{T}\circ f^{\prime}(z)$的运算结果是完全一致的。所以在这种特定情况下，为了书写简洁和表达方便，就可以把$u^{T}\text{diag}(f^{\prime}(z))$写成$u^{T}\circ f^{\prime}(z)$。 例如，设$u^{T}=[2,3,4]$，$f^{\prime}(z)=[1,5,7]$，那么$\text{diag}(f^{\prime}(z))=\begin{bmatrix}1 & 0 & 0\\0 & 5 & 0\\0 & 0 & 7\end{bmatrix}$。 计算$u^{T}\text{diag}(f^{\prime}(z))$可得： $$\begin{align*} u^{T}\text{diag}(f^{\prime}(z))&=[2,3,4]\begin{bmatrix}1 & 0 & 0\\0 & 5 & 0\\0 & 0 & 7\end{bmatrix}\\ &=[2\times1,3\times5,4\times7]\\ &=[2,15,28] \end{align*} $$ 而$u^{T}\circ f^{\prime}(z)=[2\times1,3\times5,4\times7]=[2,15,28]$，二者结果相同。

### 复用计算结果 Re-using Computation
![image.png|607](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131203353.png)
The same! Let’s avoid duplicated computation... 
<mark style="background: #D2B3FFA6;">是一样的！让我们避免重复计算... </mark>
解释 1. **复用思想**：发现计算 $\frac{\partial s}{\partial W}$ 和 $\frac{\partial s}{\partial b}$ 的链式法则表达式中，$\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}$ 这部分是相同的。基于此，强调可以避免重复计算这部分，以提高计算效率，这体现了在神经网络梯度计算中复用中间计算结果的思想，有助于减少计算量，尤其是在大规模神经网络训练中，这种优化可以显著提升训练速度。

![image.png|495](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131211201.png)

定义 $\delta = \frac{\partial s}{\partial h}\frac{\partial h}{\partial z}=u^{T}\circ f^{\prime}(z)$，它被称为局部误差信号。

### 关于矩阵的导数：输出形状  Derivative with respect to Matrix: Output shape
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131211622.png)

-  $\frac{\partial s}{\partial W}$ 是什么样的？$W\in\mathbb{R}^{n\times m}$（$W$ 是 $n\times m$ 的实数矩阵） 
- 1 个输出，$nm$ 个输入：1 行 $nm$ 列的雅可比矩阵？ 
- 然后进行 $\theta^{new}=\theta^{old}-\alpha\nabla_{\theta}J(\theta)$（参数更新公式）会很不方便 
- 相反，我们抛开纯数学（方法），采用<mark style="background: FFFF00;">形状约定(shape convention)</mark>：梯度的形状与参数的形状相同！ 
1. **问题提出**：探讨当对矩阵 $W$（$W\in\mathbb{R}^{n\times m}$ ，即 $n$ 行 $m$ 列的矩阵）求 $s$ 关于 $W$ 的导数 $\frac{\partial s}{\partial W}$ 时，其形状应该是怎样的。从常规数学角度看，若有 1 个输出，$nm$ 个输入（因为矩阵 $W$ 有 $nm$ 个元素），可能会认为 $\frac{\partial s}{\partial W}$ 是一个 1 行 $nm$ 列的雅可比矩阵。 
> 1.  - <mark style="background: #D2B3FFA6;">雅可比矩阵用于描述一个多输入多输出函数的一阶偏导数。</mark>当有一个函数 $s$ ，其输入是一个 $n\times m$ 的矩阵 $W$ 时，从本质上来说，$W$ 可以看作是由 $nm$ 个独立的输入元素组成的。 - 由于 $s$ 是一个标量输出（1 个输出），对于每个输入元素 $w_{ij}$（$i = 1,\cdots,n$；$j = 1,\cdots,m$），都有一个对应的偏导数 $\frac{\partial s}{\partial w_{ij}}$ 。 - 把这些偏导数排列起来，就形成了一个 1 行 $nm$ 列的矩阵，即雅可比矩阵。从矩阵求导的角度，这是根据标量对多个变量（这里是矩阵的元素）求导的规则得到的。在这种情况下，求导结果的每一个元素对应着 $s$ 对 $W$ 中一个元素的变化率。 2. **例子** - 假设 $n = 2$ ，$m = 2$ ，即 $W=\begin{bmatrix}w_{11}&w_{12}\\w_{21}&w_{22}\end{bmatrix}$ ，并且 $s = w_{11}+ 2w_{12}+3w_{21}+4w_{22}$ 。 - 计算偏导数： - $\frac{\partial s}{\partial w_{11}} = 1$ ； - $\frac{\partial s}{\partial w_{12}} = 2$ ； - $\frac{\partial s}{\partial w_{21}} = 3$ ； - $\frac{\partial s}{\partial w_{22}} = 4$ 。 - 那么 $\frac{\partial s}{\partial W}$ 按照 1 行 $nm$ （这里 $nm = 2\times2 = 4$ ）列的形式排列，就是 $\frac{\partial s}{\partial W}=\begin{bmatrix}1&2&3&4\end{bmatrix}$ ，这就是一个 1 行 4 列的雅可比矩阵，符合 1 个输出，$nm$ 个输入时雅可比矩阵的形式。 然而，在实际的神经网络计算和参数更新中，这种形式会带来不便，所以引入了形状约定，使梯度的形状与参数 $W$ 的形状相同，即 $\frac{\partial s}{\partial W}$ 为 $2\times2$ 的矩阵形式，以方便后续的计算和操作。
> 

2. **形状约定**：为了解决上述不便，引入“形状约定”，即规定梯度的形状与参数的形状相同。这样，$\frac{\partial s}{\partial W}$ 的形状就与 $W$ 一样是 $n\times m$ 。这种约定在实际的神经网络训练和优化算法中，使得参数更新等操作更加直观和简便，减少了数据结构转换等额外的操作，提高了计算效率和代码实现的便利性。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131212333.png)
- 答案是：$\frac{\partial s}{\partial W}=\delta^{T}x^{T}$ $\delta$ 是 $z$ 处的局部误差信号 $x$ 是局部输入信号  
- **推理过程** 
	- 因为 $\delta$ 是局部误差信号，在最终结果中会包含 $\delta$ 。
	- 对于 $\frac{\partial z}{\partial W}$ ，由于 $z = Wx + b$ ，根据矩阵求导规则，对 $z$ 关于 $W$ 求导，结果与 $x$ 有关。从 $z = Wx + b$ 可以看出，$z$ 是关于 $W$ 和 $x$ 的线性组合，对 $W$ 求导时，$b$ 为常数项不影响，根据矩阵乘法的求导性质，$\frac{\partial z}{\partial W}=x^{T}$ 。
	- 结合前面的 $\delta$ ，考虑到矩阵运算的维度匹配等规则，最终得到 $\frac{\partial s}{\partial W}=\delta^{T}x^{T}$ 。这里 $\delta$ 和 $x$ 都是向量，通过转置操作使得矩阵乘法在维度上合理，以满足计算要求。在神经网络的反向传播算法中，这个结果用于计算损失函数关于权重矩阵 $W$ 的梯度，以便后续更新权重参数，优化神经网络的性能。

### Deriving local input gradient in backprop

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131212931.png)

### x为什么转置 Why the Transposes?
![image.png|697](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131213442.png)
 - 简单解释：这样能使维度匹配！ 
 - 这是检查计算的有用技巧！ 
 - 完整解释在讲义中 
 - 每个输入对应每个输出 - <mark style="background: #D2B3FFA6;">你想要得到外积  outer product</mark>

[[点积、叉积、内积、外积【汇总对比】-CSDN博客]]

 1. **维度分析**： - 已知权重矩阵 $W$ 的维度是 $n\times m$ ，$\delta$ 是局部误差信号，$x$ 是局部输入信号。为了使矩阵乘法的维度合理，$\delta$ 需是 $n\times 1$ 的列向量（转置后为 $1\times n$ 行向量 $\delta^{T}$ ），$x$ 需是 $m\times 1$ 的列向量（转置后为 $1\times m$ 行向量 $x^{T}$ ）。 
 - 这样 $\delta^{T}x^{T}$ ，$[n\times 1]$ 的列向量转置后与 $[1\times m]$ 的行向量相乘，得到维度为 $n\times m$ 的矩阵，与 $\frac{\partial s}{\partial W}$ 应有的维度（和 $W$ 的维度一致，因为要对 $W$ 的每个元素求偏导，其结果形状和 $W$ 相同）相匹配。从矩阵乘法运算角度，$\begin{bmatrix}\delta_{1}\\\vdots\\\delta_{n}\end{bmatrix}[x_{1},\ldots,x_{m}]$ 得到的就是一个 $n\times m$ 的矩阵，元素为 $\delta_{i}x_{j}$ （$i = 1,\ldots,n$；$j = 1,\ldots,m$ ）。 
 2. **实际意义**：**这种转置操作从物理意义上看，实现了每个输入（$x$ 的元素）与每个输出（对应于 $\delta$ 的元素相关的计算）之间的关联，得到的是外积形式**。在神经网络中，这对于准确计算损失函数关于权重矩阵 $W$ 的梯度至关重要，因为梯度的准确计算是后续利用优化算法更新权重的基础。同时，通过检查维度是否匹配，也可以作为一种检查计算是否正确的实用技巧。
### 导数应该是什么形状？ What shape should derivatives be?
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131214705.png)

- 同样地，$\frac{\partial s}{\partial b}=h^{T}\circ f^{\prime}(z)$ 是一个行向量。 
- 但形状约定表明我们的梯度应该是一个列向量，因为 $b$ 是一个列向量…… 
- 雅可比形式（使链式法则易于应用）和形状约定（使随机梯度下降（SGD）易于实现）之间存在不一致。 
- 我们期望作业中的答案遵循形状约定。 
- 但雅可比形式对于计算答案很有用。 
1. **导数形状的矛盾**：对于 $\frac{\partial s}{\partial b}$ ，根据前面的计算和运算规则得到 $\frac{\partial s}{\partial b}=h^{T}\circ f^{\prime}(z)$ ，其结果是一个行向量。然而，按照形状约定（在神经网络中为了方便实现随机梯度下降等优化算法），由于偏置 $b$ 是列向量，梯度的形状应该与参数 $b$ 的形状相同，所以 $\frac{\partial s}{\partial b}$ 应该是列向量。这就产生了雅可比形式（按照标准的矩阵求导得到的形式，便于链式法则的应用）和形状约定之间的不一致。 
2. **应用中的考量**：在实际应用中，虽然雅可比形式对于计算导数很有用，它基于严谨的数学求导规则，能方便地通过链式法则进行逐步推导计算。但在作业或实际编程实现中，为了使随机梯度下降等算法的实现更加直观和简便，通常期望答案遵循形状约定，即梯度的形状与参数的形状保持一致，这样在更新参数（如 $b = b-\alpha\nabla b$ ，其中 $\nabla b$ 为关于 $b$ 的梯度，$\alpha$ 为学习率）时更加方便，无需进行额外的维度转换等操作。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131215054.png)
两种选择： 
1. 尽可能多地使用雅可比形式，最后重塑形状以遵循形状约定： 
	- 就是我们刚才所做的。但最后对 $\frac{\partial s}{\partial b}$ 进行转置，使导数成为一个列向量，结果为 $\delta^{T}$。 
2. 始终遵循形状约定 
- 通过查看维度来确定何时进行转置和/或重新排列项。 
- 到达隐藏层的误差信号 $\delta$ 与该隐藏层具有相同的维度。 

1. **第一种选择**：在计算导数时，优先使用雅可比形式。因为雅可比形式基于严格的数学求导规则，便于运用链式法则进行逐步推导计算。例如在复杂的神经网络导数计算中，按照雅可比形式可以清晰地通过链式法则将导数分解为多个简单的求导步骤。但由于形状约定（梯度的形状应与参数形状一致，便于随机梯度下降等算法实现），在计算的最后阶段，需要对得到的结果进行形状调整。如对于 $\frac{\partial s}{\partial b}$ ，其雅可比形式计算结果可能是行向量，为了符合形状约定（$b$ 通常是列向量，所以梯度也应为列向量），要对其进行转置操作，得到 $\delta^{T}$，使其成为列向量形式。 
2. **第二种选择**：从计算的一开始就始终遵循形状约定。在每一步计算中，都通过查看各个量的维度，来确定是否需要对矩阵或向量进行转置操作，以及重新排列各项。这样做的好处是在实现随机梯度下降等优化算法时更加直观和简便，无需在最后阶段进行额外的形状调整。同时，还提到到达隐藏层的误差信号 $\delta$ 的维度与该隐藏层的维度相同，这有助于在遵循形状约定的过程中，正确处理误差信号以及进行相关的计算和操作。

3. 反向传播 Backpropagation
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131215351.png)
### 英文原文 3.  We’ve almost shown you backpropagation It’s taking derivatives and using the (generalized, multivariate, or matrix) chain rule Other trick: We re - use derivatives computed for higher layers in computing derivatives for lower layers to minimize computation ### 中文翻译  
- 我们几乎已经向你展示了反向传播 
	- 它是进行求导，并使用（广义的、多元的或矩阵的(generalized, multivariate, or matrix）链式法则 
- 另一个技巧： <mark style="background: #D2B3FFA6;">我们在计算较低层的导数时，复用为较高层计算的导数，以最小化计算量 </mark>
1. **反向传播本质**：反向传播是训练神经网络的关键算法。其核心在于计算导数，通过运用广义的、多元的或矩阵形式的链式法则来实现。<mark style="background: FFFF00;">神经网络通常是一个复杂的复合函数，链式法则能将对输出的误差关于网络参数（如权重和偏置）的导数计算，分解为多个简单的求导步骤，从输出层逐步反向传播到输入层，从而计算出每个参数的梯度。 </mark>
2. **复用技巧**：在计算过程中，为了减少计算量，采用复用导数的技巧。在神经网络中，较高层的导数计算结果往往包含了一些中间信息，这些信息在计算较低层的导数时可以被再次利用。例如，在计算某一层的权重和偏置的导数时，可能已经在计算上一层的导数过程中得到了一些相关的中间结果，通过复用这些结果，避免了重复计算，提高了计算效率，尤其在大规模的深度神经网络训练中，这种优化能显著节省计算资源和时间。

### 计算图与反向传播  Computation Graphs and Backpropagation
![image.png|667](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131225006.png)
 - Software represents our neural net equations as a graph 软件将我们的神经网络方程表示为一个图
	 - Source nodes: inputs 
	 - Interior nodes(内部节点): operations 操作
	 - Edges pass along result of the operation(边传递操作的结果) 

1. **计算图的表示**：<mark style="background: #D2B3FFA6;">在神经网络中，软件通常将神经网络的方程以计算图的形式呈现。</mark>计算图是一种有向图，它清晰地展示了数据在网络中的流动和计算过程。 
2. **节点类型** - **源节点**：代表输入，如图中的 \(x\) 是输入数据，它是整个计算过程的起始。 - **内部节点**：表示各种操作，比如矩阵乘法（\(Wx\) ）、加法（\(Wx + b\) ）以及激活函数运算（\(h = f(z)\) 中的 \(f\) ）等。这些操作按照一定的顺序对输入数据进行变换和处理。 
3. **边的作用**：边用于传递操作的结果，从一个节点的输出传递到下一个节点作为其输入。这种表示方式有助于理解神经网络的计算过程，并且在反向传播算法中，计算图可以帮助确定梯度的传递路径，便于计算每个参数的梯度，从而更新神经网络的参数。
4. <mark style="background: FFFF00;">这个方向是 Forward propagation(前向传播)</mark>

### 反向传播 Backpropagation
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131230632.png)

反向传播
- 然后沿着边反向传播
- 传递梯度

#### 反向传播：单个节点 Backpropagation: Single Node
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131230917.png)
### 英文原文  - Each node has a local gradient - The gradient of its output with respect to its input \[h = f(z)\] 
每个节点都有一个局部梯度 
- 即其输出关于输入的梯度 - - [downstream gradient] = [upstream gradient] x [local gradient] 

1. **局部梯度的定义**：在反向传播中，对于单个节点（如由 h = f(z) 表示的节点，其中 f 是激活函数），它具有一个局部梯度，这个局部梯度是该节点的输出 h 关于其输入 z的梯度，用 $\frac{\partial h}{\partial z}$ 表示。 
2. **梯度传播规则**：在反向传播过程中，梯度的传递遵循特定规则，<mark style="background: #D2B3FFA6;">即下游梯度等于上游梯度乘以局部梯度。</mark>例如，在图中，$\frac{\partial s}{\partial z}$ 是下游梯度(s 关于 z 的梯度），$\frac{\partial s}{\partial h}$ 是上游梯度（s 关于 h 的梯度），$\frac{\partial h}{\partial z}$ 是局部梯度，根据链式法则，就有 $\frac{\partial s}{\partial z}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}$ 。<mark style="background: #D2B3FFA6;">这种梯度传播方式在整个神经网络的反向传播中起着关键作用，它使得误差能够从输出层逐步反向传播到输入层，从而计算出每个参数的梯度，以便更新神经网络的参数，优化网络性能。</mark>

#### 多个输入 Multiple inputs
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131231436.png)
#### Example
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131232917.png)

以y举例，当y增加0.1,也就是y=2.1
- a = 3.1
- b = 2.1
- f = ab = 6.51
- 倍数 (6.51-6)/0.1 约等于 5.1也就是 5
当y减少0.1,也就是y=1.9
- a = 2.9
- b =1.9
- f = ab = 5.51
- 倍数 (6-5.51) / 0.1 = 4.9
因为这个梯度是在 y =2计算的，所以变换一小步，实际影响f就不是5倍了，是附近的一个值如5.1和4.9.

#### 向外分支的梯度和 Gradients sum at outward branches
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131233522.png)
对应y被应用到了加的2部分，他们的梯度应该相加。

#### 节点直观理解 Node Intuitions 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131233706.png)
- 加法（+）“分配”上游梯度 
- 取最大值（max）“路由”上游梯度 
- 乘法（\*）“切换”上游梯度 
1. **节点对梯度的作用** 
	- **加法节点**：在反向传播中，加法操作会“分配”上游梯度。这意味着上游梯度会按照一定规则分配到加法操作的各个输入上。例如，若从乘法节点传来的上游梯度为某个值，这个梯度会被分配到 \(x\) 和 \(y\) 对应的路径上。 
	- **取最大值节点**：<mark style="background: #D2B3FFA6;">取最大值操作“路由”上游梯度。</mark><mark style="background: FFFF00;">对于 \(y\) 和 \(z\) ，因为 \(y>z\) ，在反向传播时，梯度只会通过 \(y\) 对应的路径传递，就像将梯度“路由”到了 \(y\) 这条路径上。</mark> 
	- **乘法节点**：乘法操作“切换”上游梯度。它会根据乘法两边的值对上游梯度进行某种变换，改变梯度的大小和方向等特性，以适应反向传播的需求。 这些对节点在反向传播中作用的直观理解，有助于更好地分析和优化神经网络的训练过程，理解梯度在网络中的流动和变化。
### Efficiency: compute all gradients at once 一次性计算所有梯度
![image.png|685](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131234259.png)
- 正确方法：
    - 一次性计算所有梯度
    - 类似于我们手动计算梯度时使用  的情况
    - 避免了重复计算。
 ### 一般计算图中的反向传播 Back - Prop in General Computation Graph
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131234939.png)
1. 前向传播（Fprop）: visit nodes in topological sort order  按拓扑排序顺序访问节点
	- Compute value of node given <mark style="background: #D2B3FFA6;">predecessors</mark> 根据前驱节点计算节点的值
2. Bprop: 
	- initialize output gradient = 1  初始化输出梯度
	- 以相反顺序访问节点(visit nodes in reverse order): Compute gradient wrt each node using gradient wrt successors 使用关于后继节点的梯度来计算关于每个节点的梯度<mark style="background: FFFF00;"> “wrt” 是 “with respect to” 的缩写，在数学和科学领域中常见，中文意思是 “关于；相对于” 。</mark>
	- $\{y_1,y_2,\ldots y_n\} = \text{successors of }x$ 
	- $\frac{\partial z}{\partial x}=\sum_{i = 1}^{n}\frac{\partial z}{\partial y_i}\frac{\partial y_i}{\partial x}$ 
- Done correctly, big O() complexity of fprop and bprop is **the same**  <mark style="background: FFFF00;">如果正确执行，前向传播和反向传播的大 O 复杂度是相同的</mark> 
- In general, our nets have regular layer - structure and so we can use matrices and Jacobians... 一般来说，我们的网络具有规则的层结构，因此我们可以使用矩阵和雅可比矩阵……
1. **前向传播（Fprop）**：在计算图中，前向传播按照拓扑排序的顺序访问节点。拓扑排序确保在计算某个节点的值时，其所有前驱节点的值都已经计算完成。例如，对于一个节点，它的计算依赖于其他节点的输出，拓扑排序能保证这些依赖的节点先被计算，从而正确计算该节点的值。 
2. **反向传播（Bprop）** 
	- 首先将输出梯度初始化为 1。这是反向传播的起始步骤，为后续的梯度计算提供了基础。 
	- **然后以与前向传播相反的顺序访问节点。在计算关于某个节点 $x$的梯度 $\frac{\partial z}{\partial x}$ 时，利用关于其所有后继节点 $y_i$$(i = 1,\ldots,n$）的梯度 $\frac{\partial z}{\partial y_i}$ 以及 $y_i$ 关于 $x$ 的梯度 $\frac{\partial y_i}{\partial x}$ ，通过公式 $\frac{\partial z}{\partial x}=\sum_{i = 1}^{n}\frac{\partial z}{\partial y_i}\frac{\partial y_i}{\partial x}$ 来计算。这个公式本质上是链式法则在计算图中的应用，它将对最终输出 $z$ 的梯度通过中间节点 $y_i$ 反向传播到节点 $x$ 。 
3. **复杂度和应用**：如果正确执行前向传播和反向传播，它们的大 O 复杂度是相同的。由于一般的神经网络具有规则的层结构，所以可以利用矩阵和雅可比矩阵来高效地表示和计算这些梯度，这有助于简化计算过程并提高计算效率。

### 自动微分 Automatic Differentiation 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201000057.png)

- The gradient computation can be automatically inferred from the symbolic expression of the fprop 
- Each node type needs to know how to compute its output and how to compute the gradient wrt its inputs given the gradient wrt its output 
- Modern DL frameworks (Tensorflow, PyTorch, etc.) do backpropagation for you but mainly leave layer/node writer to hand - calculate the local derivative 
- 梯度计算可以从正向传播（fprop）的符号表达式中自动推导出来 
- <mark style="background: #D2B3FFA6;">每种节点类型需要知道如何计算其输出，以及在给定关于其输出的梯度时，如何计算关于其输入的梯度 </mark>
- 现代深度学习框架（如TensorFlow、PyTorch等）会为你执行反向传播，但主要还是让layer/node 编写者手动计算局部导数 
1. **梯度自动推导**：自动微分的<mark style="background: #D2B3FFA6;">核心在于，能够依据正向传播的符号表达式，自动地计算梯度</mark>。在神经网络的正向传播过程中，数据按照一定的计算规则在各个节点间流动并产生输出。自动微分可以根据这些计算规则，利用链式法则等数学原理，自动推导出每个参数的梯度。 2. **节点的计算要求**：<mark style="background: #D2B3FFA6;">对于计算图中的每一种节点类型，都需要具备两种计算能力。一是计算自身输出的能力，即根据输入数据和节点的运算规则得出输出值。二是在已知关于自身输出的梯度时，能够计算关于其输入的梯度。</mark>这是实现反向传播的基础，因为反向传播就是通过节点间的这种梯度传递来计算整个网络的梯度。 
2. **深度学习框架的作用**：像TensorFlow和PyTorch这样的现代深度学习框架，它们内部集成了反向传播算法，能够自动执行反向传播过程。<mark style="background: #D2B3FFA6;">然而，对于一些自定义的层或节点，框架主要还是要求开发者手动计算局部导数。</mark>

### 反向传播实现 Backprop Implementations
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201000410.png)
- forward 按拓扑排序的顺序执行前向传播计算。
- backward 按forward的逆序进行反向传播
### Implementation: forward / backward API 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201000830.png)

### 手动梯度检查 Manual Gradient checking: Numeric Gradient 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201001029.png)
### 英文原文 - For small $h (\approx 1e - 4)$, $f^{\prime}(x)\approx\frac{f(x + h)-f(x - h)}{2h}$ 
- Easy to implement correctly 
- But approximate and very slow: 
	- You have to recompute $f$ for every parameter of our model 
- Useful for checking your implementation 
	- In the old days, we hand-wrote everything, doing this everywhere was the key test 
	- Now much less needed; you can use it to check layers are correctly implemented
数值梯度 
- 对于小的 $h$（约为 $1\times10^{-4}$），$f^{\prime}(x)\approx\frac{f(x + h)-f(x - h)}{2h}$ 
- 易于正确实现 
- 但具有近似性且非常慢： 
	- 你必须为模型的每个参数重新计算 $f$ 
- 对检查实现是否正确很有用
	- 在过去，我们手写所有内容，在各处进行此操作是关键测试 
	- 现在需求少得多；你可以用它来检查层是否正确实现
1. **优缺点** 
	- **优点**：易于正确实现，只要按照公式进行编程计算即可。 
	- **缺点**：具有近似性，因为它只是对导数的一个近似估计，并非精确值。而且计算速度非常慢，因为对于模型的每个参数都需要重新计算函数 $f$ 的值。在实际的深度学习模型中，参数数量往往非常庞大，这会导致计算量急剧增加。 
2. **应用场景**：<mark style="background: FFFF00;">在过去深度学习框架不发达，所有代码都需要手动编写时，手动梯度检查是一项关键测试，用于验证梯度计算的正确性。而现在，随着自动微分等技术在深度学习框架中的广泛应用，手动梯度检查的需求大幅减少，但它仍然可以用于检查自定义层或模块是否正确实现，通过对比数值梯度和自动微分得到的梯度，来判断实现是否准确。</mark>


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201001600.png)

Summary  
We’ve mastered the core technology of neural nets!
- Backpropagation: recursively (and hence efficiently) apply the chain rule along computation graph
    - [downstream gradient] = [upstream gradient] x [local gradient]
- Forward pass: compute results of operations and save intermediate values
- Backward pass: apply chain rule to compute gradients
我们已经掌握了神经网络的核心技术！

- 反向传播：沿着计算图递归地（因此也是高效地）应用链式法则
    - [下游梯度] = [上游梯度] × [局部梯度]
- 前向传播：计算操作结果并保存中间值
- 反向传播：应用链式法则计算梯度

### 为什么学关于梯度这么多细节 Why learn all these details about gradients?
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201001715.png)
为什么要学习关于梯度的所有这些细节？

- 现代深度学习框架会为你计算梯度！
    - 本周五来参加 PyTorch 介绍会！
- 但是当编译器或系统已经为你实现时，为什么还要学习相关课程呢？
    - 理解底层的运行机制是很有用的！(Understanding what is going on under the hood is useful!)
- 反向传播并不总是完美运行(Backpropagation doesn't always work perfectly)
    - 理解原因对于调试和改进模型至关重要(- Understanding why is crucial for debugging and improving models)
    - 查看 Karpathy 的文章（在教学大纲中）：
        - [https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
    - 未来课程中的例子：梯度爆炸和梯度消失( exploding and vanishing gradients)