---
created: 2024-10-10T22:53
updated: 2025-01-29T21:47
---
## 1. main point
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241010230510.png)
我们希望教什么？
1. 应用于NLP的有效现代深度学习方法的基础。
- 首先是基础知识，然后是NLP中使用的关键方法：循环网络、注意力、变换器等。
2. 对人类语言的总体理解以及理解和产生语言的困难
3. 对NLP中一些主要问题的理解和构建系统的能力（在PyTorch中）：
- 词义、依存关系解析、机器翻译、问答

example 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101211602.png)
- 两个女人在吵架，同样的语言，不同人的理解不一样
```ad-tip
这个漫画来自 _xkcd_，讨论了语言的混乱性和交流的本质，围绕着 "I could care less"（我可能更不在乎）和 "I couldn't care less"（我一点都不在乎）之间的争论展开。以下是对话翻译和解释：
### **人物对话（汉英对照版）**

#### **第一部分（语法纠正争论 | Grammar Correction Debate）**

👩‍🦰：「……反正，我可能更不在乎。」  
👩‍🦰: "…Anyway, I could care less."

👩：「你是想说 ‘I couldn't care less’ 吧？你这样说（I could care less）意味着你仍然在乎一点。」  
👩: "I think you mean you _couldn’t_ care less. Saying _you could care less_ implies you care at least some amount."

👦：「呃……我不知道。」  
👦: "I dunno."

---

#### **第二部分（语言的本质 | The Nature of Language）**

👦（画面转黑暗背景，沉思状）：  
👦 (Floating in darkness, deep in thought):

「我们这些极度复杂的大脑在虚空中漂浮，徒劳地试图彼此连接，只能盲目地将词语抛向黑暗……」  
_"We're these unbelievably complicated brains drifting through a void, trying in vain to connect with one another by blindly flinging words out into the darkness…"_

「每一个措辞、拼写、时间点、语调都承载着无数的信号、语境、潜台词等等……  
而每个听者都会以自己的方式解读这些信号。」  
_"Every choice of phrasing and spelling and tone and timing carries countless signals and contexts and subtexts and more, and every listener interprets those signals in their own way."_

「语言并不是一个严格的系统，语言是辉煌的混沌。」  
_"Language isn’t a formal system. Language is glorious chaos."_

---

#### **第三部分（沟通的意义 | The Purpose of Communication）**

👩：「你永远无法确定任何词对任何人意味着什么。」  
👩: "You can never know for sure what _any_ words will mean to _anyone_."

「你所能做的，就是努力去猜测你的话语会如何影响他人，  
这样你才有可能找到那些能让对方感受到与你想表达的情感相近的词。」  
_"All you can do is try to get better at guessing how your words affect people, so you can have a chance of finding the ones that will make them feel something like what you want them to feel."_

「除此之外，一切都是毫无意义的。」  
_"Everything else is pointless."_

---

#### **第四部分（对方的回应 | The Response）**

👦：「所以，你是在告诉我你如何理解这些词，是因为你希望让我感觉不那么孤单？  
如果是的话，谢谢你，那对我来说很重要。」  
👦: "I assume you’re giving me tips on how you interpret words because you want me to feel less alone.  
If so, then thank you. That means a lot."

👦：「但如果你只是单纯地在我的句子上打个勾，  
然后用它来炫耀自己多懂语法……」  
👦: "But if you’re just running my sentences past some mental checklist so you can show off how well you know it…"

👦（微笑，带有讽刺意味）：「那我就真的可能更不在乎了。」  
👦 (Smiling, sarcastically): "Then I _could_ care less."

### **解释**

这个漫画讽刺了一种常见的语言争论——纠正别人说 "I could care less" 而非 "I couldn't care less"。  
但作者 _Randall Munroe_ 通过漫画中的角色表达了更深层的观点：**语言不是固定的逻辑系统，而是混乱但美丽的交流工具。**

- 语言的意义取决于听众如何解读，而不是纯粹的语法规则。
- 纠正别人用词的真正目的应该是帮助沟通，而不是炫耀自己的知识。
- 如果只是纠正而没有关心对方的感受，那这种纠正就毫无意义。

最后一句 _"Then I could care less."_ 讽刺地回击了最初的纠正者，强调如果只是为了炫耀知识，那他就真的"不在乎"了。
```

- 人类的沟通能力使得人类比其他生物更有优势。

- 计算机和人工智能的关键问题：==如何让计算机能够理解人类传达的意思==

## 2.机器翻译和自然语言处理
### (1)机器翻译 neural machine translation
- Trained on text data, neural machine translation is quite good!
### (2)NLP 自然语言处理
NLP的最大进展是GPT3
- GPT3: A first step on the path to universal models 通用模型的第一步
- 我们不用再去专门的设定具体的功能，比如检测垃圾邮件，色情信息，任何语言的信息，只是建立所有这些不同任务的独立监督分类器,我们刚刚建立了一个可以理解的模型。
==and just building all these separate supervised
classifiers for every different task，we've now just built up a  model  that  understands .

==所以它所能做的只是预测后面的单词,左边输入要生成什么，模型就会生成后面的文本，实际上是一次预测一个单词，然后生成的单词又作为输入去预测，循环往复。
So exactly what it does is it just predicts following words.

#### <1>GPT3 功能
- 我们给出问题，和prompt，GPT3可以给出合理的问答后续
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101214712.png)
>    S: I broke the window.  
	S：我打破了窗户。
	Q: What did I break?  
	Q：我打破了什么？
	S: I gracefully saved the day.  
	S：我优雅地拯救了这一天。
	Q: What did I gracefully save?  
	Q：我优雅地拯救了什么？
- 翻译sql
GPT3可以理解语义和SQL，进行转化
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101222157.png)

#### <2>单词意义的理解 the meaning of word
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101222333.png)
**Webster dictionary对meaning的定义:** 
> 集中在idea(想法)这里
- the idea that is represented by a word, phrase, etc.
- the idea that a person wants to express by using words, signs, etc.
- the idea that is expressed in a work of writing, art, etc.
**Commonest linguistic way of thinking of meaning:**
最常见的语言意义思维方式：
符号(symbol) 和 想法或者事情(idea or thing) 的对应(pair) 相当于 denotational semantics 指称语义

### (3)计算机中的有用意义 usable meaning in computer 
How do we have usable meaning in a computer?
我们如何在计算机中拥有有用的意义？

传统意义上在NLP中使用meaning,是充分使用词典(dictionary),特别是词库，比如流行的**WordNet**,它组织了word and terms(单词和术语)分为两个set，一个是同义词集合(synonym sets)另一个是(hypernyms)

**Common NLP solution:**
Use, e.g., WordNet, a thesaurus(分类词典) containing lists of synonym sets and hypernyms ("is a"relationships).
常见的NLP解决方案：例如，使用WordNet，一个包含同义词集和超义词列表（“is a”关系）的同义词词典。

![image.png|613](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129141117.png)

### (4)WordNet等资源的问题 Problems with resources like WordNet 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129142422.png)

- 作为一种资源很棒，但缺少细微差别
	- 例如，“proficient(精通)”被列为“good(良好)”的同义词,这仅在某些情况下是正确的
-   缺少单词的新含义
	- 例如，邪恶的、坏蛋的、漂亮的、巫师的、天才的、忍者的、最棒的。(wicked, badass, nifty, wizard, genius, ninja, bombest)
	- 无法保持最新状态(up-to-date)！
- 主观的(subjective)
- 需要人类劳动来创造和适应。
- 无法计算准确的单词相似度

### (5)离散符号的单词 words as discrete symbols 
Representing words as discrete symbols
将单词表示为离散符号

ln traditional NLP, we regard words as discrete symbols:hotel, conference, motel- a localist representation
在传统的自然语言处理中，**我们将词语视为离散的符号**：旅馆、会议、汽车旅馆--“a localist representation（局部表征）in deep learning

> 在深度学习中，**localist representation**（局部表示）指的是将每个概念、类别或特定特征由单个神经元或极少量神经元独立表示。例如，在一个神经网络中，某个特定的神经元可能专门对“猫”这一概念负责，而另一个神经元可能代表“狗”。
> 相比之下，**distributed representation**（分布式表示）是更常见的做法，它将信息编码为多个神经元的激活模式。例如，在Transformer模型中，一个词的含义可能由多个隐藏单元的组合决定，而不是单独的神经元。
> **局部表示的优点是可解释性更强，但缺点是容易导致泛化能力不足（generalization issue），因为它缺乏对数据的广泛特征提取能力。**

可以使用 one-hot vectors 也就是独热编码进行表示，维度是词表的长度。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129143851.png)

当想要搜 Seattle motel ，想去匹配 Seattle hotel，但motel和hotel的独热编码
- "These two vectors are orthogonal" → 这两个向量是**正交的**
- "There is no natural notion of similarity for one-hot vectors!" → **独热（one-hot）向量**本身没有自然的**相似性**概念！
- **Solution:** → **解决方案：**
    - "Could try to rely on WordNet’s list of synonyms to get similarity?" → **可以尝试依赖 WordNet 的同义词列表来获得相似性？**
        - "But it is well-known to fail badly: incompleteness, etc." → **但这方法众所周知效果很差，比如不完整性等问题。**
    - "**Instead: learn to encode similarity in the vectors themselves**" → **相反，我们应该让向量本身学习编码相似性。**
更好的方法是**让模型学习如何在向量中编码相似性**，这也是**词向量（word embeddings）** 方法 （如 Word2Vec、GloVe、BERT）的核心思想。这些方法能将语义上相似的词映射到相近的向量空间，从而提供更合理的相似性度量。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129144104.png)

### (6)单词分布式表示 word distributed representation 

用上下文表示单词 Representing words by their context
→ **用上下文表示单词**
![image.png|603](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129145003.png)

**Translation:**
- **Distributional semantics: A word’s meaning is given by the words that frequently appear close-by** → **分布式语义：一个单词的含义由其附近频繁出现的单词决定。**
    - _“You shall know a word by the company it keeps”_ (J. R. Firth 1957: 11) → **“你可以通过一个单词的伴随词来了解它的含义。”**（J. R. Firth 1957）
    - **<mark style="background: FFFF00;">One of the most successful ideas of modern statistical NLP! → 现代统计自然语言处理（NLP）最成功的思想之一！ </mark>**
- **When a word _w_ appears in a text, its context is the set of words that appear nearby (within a fixed-size window).**  
    → **当一个单词 _w_ 出现在文本中时，它的上下文（context）是其附近（在一个固定大小的窗口内）出现的单词集合。**
- **Use the many contexts of _w_ to build up a representation of _w_**  
    → **利用 _w_ 的多个上下文来构建 _w_ 的表示。**
- **These context words will represent banking**  
    → **这些上下文单词将用于表示 “banking”。**
这张幻灯片介绍了**分布式语义（distributional semantics)的核心思想，即**“一个单词的意义由其上下文决定”**。这一理念由语言学家 J. R. Firth 提出，被广泛应用于现代自然语言处理（NLP），特别是在**词向量（word embeddings）**的训练中（如 Word2Vec、GloVe 等）。

举例来说，在多个句子中，**banking** 一词的上下文可能包括 **crises（危机）、regulation（监管）、system（系统）** 等。这些上下文单词可以用来学习 **banking** 的含义，从而在向量空间中形成**语义相似性**（semantic similarity）。这种方法比传统的**one-hot编码**更具表达能力，因为它能捕捉单词之间的**语义关系**。

what we want to do is based on looking at the words that occur in context as vectors that we want to build up dense real valued vector for each word, that in some sense represents the meaning of that word and the way it will represent the meaning of that word is that this vector will be useful for predicting other words that occur in the context.
我们的目标是基于**上下文中出现的单词**，将它们表示为向量，并构建**稠密的实值向量**，使每个单词的向量在某种程度上能够表示该单词的**意义**。而这种向量表示单词意义的方式是，使该向量在预测**上下文中出现的其他单词**时具有**实际的作用**。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129150507.png)
**Translation:**

- **Word vectors** → **词向量**
- **We will build a dense vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts**  
    → **我们将为每个单词构建一个稠密向量，使其与出现在相似上下文中的单词的向量相似。**
- **banking =**  
    → **banking 的向量表示如下：** $$\begin{bmatrix} 0.286 \\ 0.792 \\ -0.177 \\ -0.107 \\ 0.109 \\ -0.542 \\ 0.349 \\ 0.271 \end{bmatrix}​
$$
- **Note: word vectors are also called word embeddings or (neural) word representations**  
    → **注意：词向量（word vectors）也被称为** **词嵌入（word embeddings）** 或 **（神经网络）词表示（neural word representations）。**
- **They are a distributed representation**  
    → 它们是一种 分布式表示（distributed representation）。

这张幻灯片介绍了**词向量（word vectors）**的概念。在自然语言处理中，我们不再使用**one-hot编码**（独热编码）来表示单词，而是用**密集向量（dense vectors）**，其中每个单词都会被映射到一个多维空间中的点。

这种方法的核心思想是：
- **语义相似的单词**在向量空间中**距离较近**。
- 这些向量是通过**上下文信息**训练得到的，因此能捕捉单词的**语义信息**。
- **分布式表示（distributed representation）**意味着单词的含义由向量的多个维度共同编码，而不是由某一个特定的维度决定。

例如，在 Word2Vec、GloVe 或 BERT 这样的词向量模型中，**banking** 的向量表示可能接近 **finance（金融）、investment（投资）** 这样的词，而远离 **apple（苹果）或 table（桌子）**。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129151351.png)

## 3、引入Word2vec:Overview 核心思想

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129151221.png)
**Translation:**
```ad-tip
**Word2vec** (Mikolov et al. 2013) is a framework for learning word vectors  
→ **Word2vec**（Mikolov 等，2013）是一个用于学习词向量的框架
#### **Idea（核心思想）：**

- **We have a large corpus ("body") of text**  
    → **我们有一个庞大的语料库（文本集合）**
- **Every word in a fixed vocabulary is represented by a vector**  
    → **固定词汇表中的每个单词都用一个向量表示**
- **Go through each position _t_ in the text, which has a center word _c_ and context ("outside") words _o_**  
    → **遍历文本中的每个位置 _t_，其中包含一个中心词 _c_ 和上下文（“外部”）单词 _o_**
- **Use the similarity of the word vectors for _c_ and _o_ to calculate the probability of _o_ given _c_ (or vice versa)**  
    → **利用 _c_ 和 _o_ 的词向量相似性来计算在给定 _c_（或反之）情况下出现 _o_ 的概率**
- <mark style="background: FFFF00;">Keep adjusting the word vectors to maximize this probability
    → 不断调整词向量，使该概率最大化</mark>
```
 

**Explanation:**  
这张幻灯片介绍了 **Word2Vec** 的基本概念，它是一个**用于学习词向量的无监督算法**，主要有两种方法：

1. **CBOW（Continuous Bag of Words）**：根据**上下文**预测**中心词**。
2. **Skip-gram**：根据**中心词**预测**上下文单词**。

Word2Vec 的核心思想是，**通过上下文学习单词的向量表示**，使得语义相似的单词在向量空间中彼此靠近。例如，在训练后，**king - man + woman ≈ queen**，这表明 Word2Vec 可以学习到单词之间的语义关系。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129152356.png)

### (1)目标函数 Word2vec: objective function 

the big question is ,what are we doing for working out the probability of a word occurring in the context of the center word?
**关键问题是，我们如何计算一个单词在中心词上下文中出现的概率？**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129152904.png)

**标题**：Word2vec：目标函数
**正文**：对于每个位置$t = 1, \ldots, T$，在给定中心词$w_t$的情况下，在固定大小为$m$的窗口内预测上下文词。
数据似然性：$$ Likelihood =L(\theta)=\prod_{t = 1}^{T}\prod_{\substack{-m\leq j\leq m\\j\neq0}}P(w_{t + j}|w_t;\theta)$$ $\theta$是所有要优化的变量。 
目标函数(**有时被称为代价函数或损失函数**)$J(\theta)$是（平均）负对数似然性： $$J(\theta)=-\frac{1}{T}\log L(\theta)=-\frac{1}{T}\sum_{t = 1}^{T}\sum_{\substack{-m\leq j\leq m\\j\neq0}}\log P(w_{t + j}|w_t;\theta)$$ 最小化目标函数$\Leftrightarrow$最大化预测准确性。 

在Word2vec中，上述目标函数的设计目的是通过给定中心词来预测其上下文词，从而学习到词的分布式表示。
以下是对其的详细解释： 
- **似然性函数**：$L(\theta)=\prod_{t = 1}^{T}\prod_{\substack{-m\leq j\leq m\\j\neq0}}P(w_{t + j}|w_t;\theta)$ **表示在整个文本序列（长度为$T$）中，对于每个位置$t$的中心词$w_t$，在以它为中心的大小为$m$的窗口内，预测上下文词的联合概率**。这里$\theta$是模型中所有待优化的参数，比如词向量等。通过这个连乘形式，将所有位置的预测概率联合起来，衡量模型在整个文本上对上下文词预测的可能性大小。
- **目标函数**：$J(\theta)=-\frac{1}{T}\log L(\theta)$ 是负对数似然函数。取对数是为了将连乘运算转换为加法运算，方便计算和优化；**<mark style="background: #FF0000;">前面加负号(一般都是最小化)是因为通常优化过程是最小化目标函数，而似然性是希望越大越好，所以取负对数后变成最小化问题。</mark>**$-\frac{1}{T}$是对整个文本序列求平均，使得目标函数的值更具代表性和稳定性。最小化这个目标函数就意味着要最大化模型对于上下文词的预测准确性，即让模型能够更准确地根据中心词预测出其周围的上下文词，从而使得学习到的词向量能够更好地捕捉词与词之间的语义关系 。

![image.png|520](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129193538.png)
概率计算问题 
- 接着提出问题“How to calculate $P(w_{t + j}|w_t;\theta)$?”，即如何计算给定中心词$w_t$时上下文词$w_{t + j}$的条件概率。 
概率计算方法 
 - 答案是**对每个词$w$使用两个向量：当$w$是中心词时用$v_w$，当$w$是上下文词时用$u_w$。** 
 - 对于中心词$c$和上下文词$o$，条件概率$P(o|c)$的计算公式为$P(o|c)=\frac{\exp(u_o^T v_c)}{\sum_{w\in V}\exp(u_w^T v_c)}$ 。
	 - 这里$u_o^T v_c$是上下文词$o$的向量$u_o$和中心词$c$的向量$v_c$的内积，用来表示词之间的相似性
	 - 分母$\sum_{w\in V}\exp(u_w^T v_c)$是对词汇表$V$中所有词的相关计算结果求和，通过这种方式得到给定中心词$c$时上下文词$o$出现的概率，用于后续目标函数的计算和模型优化。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129195318.png)

主要围绕Word2vec中计算上下文词概率时使用的softmax函数展开解释，具体内容如下：
#### 条件概率公式分析 
- 对于条件概率公式$$P(o|c)=\frac{\exp(u_o^T v_c)}{\sum_{w\in V}\exp(u_w^T v_c)}$$
- ①处指出分子中的点积$u_o^T v_c$（$u^T v = u \cdot v = \sum_{i = 1}^{n}u_i v_i$ ）用于比较上下文词$o$和中心词$c$的相似度。**点积越大，说明两个词的相似度越高，对应的概率也就越大。** 
- ②处表明指数运算$\exp$的作用是**使任何值变为正数。** 这是因为在概率计算中，概率值必须是非负的，通过指数运算满足了这一要求。 
- ③处说明**分母对词汇表$V$中所有词进行计算，目的是在整个词汇表上进行归一化，从而得到一个概率分布。** 即通过这种方式确保$P(o|c)$的结果是一个介于0和1之间的值，且所有可能的上下文词的概率之和为1。
#### softmax函数介绍 
- 指出上述计算$P(o|c)$的过程是softmax函数的一个例子。softmax函数将$\mathbb{R}^n$（$n$维实数空间）中的任意值$x_i$映射到$(0, 1)^n$（$n$维的开区间$(0, 1)$）中的概率分布$p_i$，其表达式为$\text{softmax}(x_i)=\frac{\exp(x_i)}{\sum_{j = 1}^{n}\exp(x_j)} = p_i$ 。
- 对softmax函数名称的解释： 
	- “max”是因为它会放大最大的$x_i$对应的概率，使得在概率分布中，较大的$x_i$对应的概率值相对更大。 
	- “soft”是因为它仍然会给较小的$x_i$分配一定的概率，而不是像硬最大化（只给最大值分配概率1，其他为0）那样。 
	- **最后提到softmax函数在深度学习中经常被使用，虽然它的名字有点奇怪，因为它返回的是一个概率分布。 **

### (2)优化方法 梯度下降

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129200034.png)
训练Word2vec模型时优化参数以最小化损失的相关内容： 
- **训练模型的方法**：为了训练模型，需要逐渐调整参数以最小化损失。 
- **参数表示**：$\theta$表示所有的模型参数，是一个长向量。在词向量维度为$d$，词汇量为$V$的情况下，由于每个词有两个向量（中心词向量和上下文词向量），所以$\theta \in \mathbb{R}^{2d*V}$，例如向量中包含“aardvark”（土豚）、“a”、“zebra”（斑马）等词对应的中心词向量$v$和上下文词向量$u$ 。
- **优化方式**：通过**梯度下降法来优化这些参数（如右侧图示所示）**，并且需要计算所有向量的梯度。 总结来说，图片说明了在Word2vec模型训练中参数的表示形式以及使用梯度下降优化参数来最小化损失的过程。
![Capture_20250129_210440.jpg|699](https://gitee.com/zhang-junjie123/picture/raw/master/image/Capture_20250129_210440.jpg)


