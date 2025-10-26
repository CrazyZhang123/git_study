---
created: 2025-01-29T21:35
updated: 2025-01-31T17:08
---

## 1.Main idea of word2vec

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250129213548.png)
这张图片介绍了一种词向量学习算法，步骤如下： 
1. **初始化**：以随机词向量开始。 
2. **遍历语料库**：对整个语料库中的每个词进行迭代。 
3. **预测周围词**：使用词向量尝试预测周围的词，计算公式为$P(o|c)=\frac{\exp(u_{o}^{T}v_{c})}{\sum_{w\in V}\exp(u_{w}^{T}v_{c})}$，其中$P(o|c)$是给定中心词$c$时预测词$o$的概率，$u_{o}$和$v_{c}$分别是相应的词向量，$V$是词汇表。<mark style="background: #ADCCFFA6;">例如，对于中心词“into”，尝试预测其周围词“problems”“turning”“banking”“crises”等出现的概率。</mark> 
4. **学习更新**：更新词向量，使其能更好地预测实际的周围词。
5. 仅需做到这些，该算法就能学习到在词空间中很好地捕捉词的相似性和有意义方向的词向量！(Doing no more than this, this algorithm learns word vectors that capture well word similarity and meaningful directions in a wordspace!)

## 2.Word2vec的参数和计算 parameters and computations
![image.png|681](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130154917.png)
- **“Bag of words” model（“词袋” 模型）**：这是自然语言处理中的一种简单模型。它将文本看作是一组无序的词的集合，忽略词的顺序、语法和句法等信息，仅考虑词的出现频率等统计信息。比如对于 “我喜欢苹果” 和 “苹果我喜欢”，在词袋模型中，它们包含的词是一样的，被视为相似的文本。
- -**The model makes the same predictions at each position（该模型在每个位置做出相同的预测）**：在词袋模型里，不会考虑词在文本中的位置因素，不管某个词出现在句子开头、中间还是结尾，模型对它的处理方式和预测都是一样的。例如在分析 “苹果很甜” 和 “很甜苹果” 时，对 “苹果” 这个词的预测不会因位置不同而不同。
- **We want a model that gives a reasonably high probability estimate(估计) to all words that occur in the context (at all often)（我们想要一个能对出现在语境中的所有（常见）词给出合理高概率估计的模型）**：意思是我们期望一个模型，对于在特定语境中经常出现的词，都能给出相对较高的出现概率估计。比如在描述水果的语境中，对于 “苹果”“香蕉” 等常见词，模型应该能合理地赋予它们较高的出现概率，这样有助于模型更好地理解和处理相关文本。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130155507.png)
- 高维空间的vector和二维空间的vector有很多属性不同，一个word/vector在高维空间可以和许多不同的vector在不同的维度上接近。
## 3.Optimization: Gradient Descent 梯度下降
![image.png|657](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130160119.png)
 - 在自然语言处理学习词向量的任务中，**定义loss函数$J(\theta)$用于衡量模型预测和真实情况的差异，目标是让其值尽可能小，以获得更优词向量。** 
 - 梯度下降作为常用优化算法，通过改变参数$\theta$来降低$J(\theta)$的值。
 - 具体操作上，**从$\theta$的随机初始值开始，每次计算$J(\theta)$的梯度，沿负梯度方向（函数下降最快方向）按一定学习步长移动，多次迭代接近最小值。**
 - 虽然图中示例是凸函数（梯度下降在凸函数上易找全局最小值），但实际中目标函数可能并非凸函数，存在复杂情况。
 - 不过实践表明，**即便目标函数非凸，梯度下降通常也能取得较好效果，帮助获取有效词向量。**
### (1)简单梯度下降 simple Gradient Descent
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130160659.png)
> 简单的梯度下降基本不使用，会使用随机梯度下降。


### (2)随机梯度下降 Stochastic Gradient Descent 

 - Problem: $J(\theta)$ is a function of **all windows in the corpus (often, billions!)**
	 - So $\nabla_{\theta}J(\theta)$ is **very expensive to compute**
 - You would wait a **very long time** before making a single update! 
 - **Very bad idea** for pretty much all neural nets! 
 - Solution: **Stochastic gradient descent (SGD)** 
 - Repeatedly sample windows, and update after each one, or each small batch **反复对窗口进行采样，并在每个窗口或每个小批次之后进行更新**
 - Algorithm: 
```python 
 while True: 
 window = sample_window(corpus) 
 theta_grad = evaluate_gradient(J,window,theta) 
 theta = theta - alpha * theta_grad 
 ```
1. **问题阐述**：在自然语言处理等任务中，成本函数 $J(\theta)$ 通常依赖于语料库中的所有窗口（这里的窗口可以理解为文本中的一个片段）。由于语料库往往非常庞大，窗口数量可能达到数十亿个，这使得计算成本函数 $J(\theta)$ 关于参数 $\theta$ 的梯度 $\nabla_{\theta}J(\theta)$ 变得极其耗时和耗费计算资源。如果按照常规方式计算梯度来更新参数，每次更新前的等待时间会非常长，这种方式对于大多数神经网络来说是不可行的。
2. **解决方案**：随机梯度下降（SGD）是一种解决上述问题的优化算法。**它不再一次性考虑语料库中的所有窗口来计算梯度，而是通过反复从语料库中采样单个窗口或小批次窗口，然后基于这些采样得到的窗口来计算梯度并更新参数 $\theta$。** 这样每次更新的计算量大大减少，能够在更短的时间内完成多次参数更新，提高了训练效率。 
3. **最常使用的梯度下降算法**


词向量的随机梯度！[附注] 
- 对于随机梯度下降（SGD），在每个这样的窗口上迭代地计算梯度 
- 但是在每个窗口中，我们最多只有 $2m + 1$ 个词，所以 $\nabla_{\theta}J_{t}(\theta)$ 非常稀疏！$$  \nabla_{\theta}J_{t}(\theta)=\begin{bmatrix} 0 \\ \vdots \\ \nabla_{v_{like}} \\ \vdots \\ 0 \\ \nabla_{u_{I}} \\ \vdots \\ \nabla_{u_{learning}} \\ \vdots \end{bmatrix} \in \mathbb{R}^{2dV} $$
1. **梯度稀疏性**：每个窗口中最多只有 $2m + 1$ 个词（$m$ 通常表示窗口的半宽度）。这意味着在计算关于参数 $\theta$ 的梯度 $\nabla_{\theta}J_{t}(\theta)$ 时，**由于只有少数与窗口内词相关的参数会有非零梯度，其余大部分参数的梯度为零**，所以梯度向量 $\nabla_{\theta}J_{t}(\theta)$ 是非常稀疏的。 
2. **梯度向量示例**：给出的梯度向量 $\nabla_{\theta}J_{t}(\theta)$ 是一个维度为 $2dV$ 的向量（$d$ 可能是词向量的维度，$V$ 是词汇表的大小）。其中只有少数与特定词（如“like”“I”“learning” 等词对应的参数）相关的梯度项（如 $\nabla_{v_{like}}$，$\nabla_{u_{I}}$，$\nabla_{u_{learning}}$ ）是非零的，其余大部分位置为零，形象地展示了梯度的稀疏性。<mark style="background: #D2B3FFA6;"> 这种稀疏性在一定程度上可以减少计算量和存储需求，在处理大规模语料库和词向量学习任务时具有一定优势。</mark>

#### stochastic gradient with word vectors
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130162542.png)

- 我们可能只更新实际出现的词向量！ 
- 解决方案：要么你需要使用**稀疏矩阵更新操作**，仅更新完整嵌入矩阵 $U$ 和 $V$ 的某些行，要么你需要**为词向量保留一个哈希表**。 
- 如果你有数百万个词向量并进行分布式计算，避免发送巨大的更新是很重要的！ 
1. **词向量更新策略**：在词向量学习过程中，为了提高计算效率，提出只更新在文本中实际出现的词向量，而不是对所有词向量都进行更新操作。
2. **解决方案** - **稀疏矩阵更新**：可以利用稀疏矩阵更新操作，仅对完整嵌入矩阵 $U$ 和 $V$ 中与实际出现词对应的行进行更新。嵌入矩阵用于存储词向量，通过这种方式可以减少不必要的计算量。 
	- **哈希表**：另一种方法是为词向量维护一个哈希表，**通过哈希表可以快速定位和更新实际出现的词向量。**
### (3) 负采样 Negative sampling
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130165203.png)
这段内容围绕词向量模型展开，主要介绍了使用两个向量的原因、模型变体及训练优化方法，具体如下：
1. **使用两个向量的原因**：<mark style="background: #D2B3FFA6;">使用两个向量是为了便于优化，最后可对其取平均值。不过算法也能为每个词仅用一个向量来实现。 </mark>
2. **模型变体(variants)** 
	- **Skip - grams（SG）**：<mark style="background: FFFF00;">给定中心词，预测上下文（“外部”）词，且预测与词的位置无关。 (推荐)</mark>
	- **Continuous Bag of Words（CBOW）**：根据上下文词（词袋形式）预测中心词。此处重点介绍了Skip - gram模型。 
3. **训练优化方法**：训练过程中采用**负采样(negative sampling)** 提高效率。目前主要关注朴素softmax训练方法，该方法虽简单但计算成本高。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130165826.png)
- **Main idea**: 针对一个**真实词对**（中心词及其上下文窗口中的一个词）与**几个噪声词对**（中心词与一个随机词配对）训练二元逻辑回归。(**train binary logistic regressions** for a true pair (center word and a word in its context window) versus several noise pairs (the center word paired with a random word) )

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130200517.png)

- 来自论文：“Distributed Representations of Words and Phrases and their Compositionality” (Mikolov et al. 2013) 
- 总体目标函数（他们进行最大化操作）：$$ J(\theta)=\frac{1}{T}\sum_{t = 1}^{T}J_{t}(\theta) \space$$$$ J_{t}(\theta)=\log\sigma(u_{o}^{T}v_{c})+\sum_{i = 1}^{k}\mathbb{E}_{j\sim P(w)}[\log\sigma(-u_{j}^{T}v_{c})]$$
- 逻辑/ sigmoid函数：$\sigma(x)=\frac{1}{1 + e^{-x}}$（我们很快就会成为好朋友） 
- <mark style="background: #D2B3FFA6;"> 我们最大化第一个对数项中两个词共现的概率，并最小化噪声词的概率</mark>(We maximize the probability of two words co-occurring in first log and minimize probability of noise words )

 1. **目标函数** - 总体目标函数 $J(\theta)$ 是对T 个时间步的 $J_{t}(\theta)$ 取平均。其中 $J_{t}(\theta)$ 由两部分组成，第一部分 $\log\sigma(u_{o}^{T}v_{c})$ 用于衡量真实词对（中心词 $c$ 与上下文词 $o$ ）的共现概率，通过sigmoid函数 $\sigma(x)$ 进行变换后取对数；第二部分 $\sum_{i = 1}^{k}\mathbb{E}_{j\sim P(w)}[\log\sigma(-u_{j}^{T}v_{c})]$ 是对 k 个噪声词的期望，**目的是最小化噪声词（即与中心词不相关的随机词 j ）与中心词共现的概率。**
	- $\mathbb{E}_{j\sim P(w)}$ **表示基于分布 $P(w)$ 对变量 $j$ 求期望**。 在词向量相关的负采样场景中，$P(w)$ 通常是一个词在语料库中出现的概率分布 。$j$ 是从这个分布中采样得到的噪声词（即与中心词在语义上不相关的随机词）。通过对 $j$ 求期望，是为了综合考虑多个从分布 $P(w)$ 中采样得到的噪声词的情况，在目标函数中最小化这些噪声词与中心词共现的概率，从而让模型更聚焦于学习真实词对（中心词和其上下文词）之间的关系，提升词向量学习的效果。**简单来说，它是在负采样过程中，衡量噪声词影响的一种数学方式，把从特定分布采样的多个噪声词纳入到计算中，以优化词向量的训练。**
 1. **sigmoid函数**：$\sigma(x)=\frac{1}{1 + e^{-x}}$ 是逻辑/ sigmoid函数，其输出值在0到1之间，常用于将输入映射为概率值，在上述目标函数中起到**将词向量的内积结果转换为概率度量**的作用。 4. **优化目的**：通过最大化目标函数 $J(\theta)$，**实现最大化真实词对的共现概率，同时最小化噪声词与中心词共现的概率，从而学习到更好的词向量表示，使得在向量空间中语义相近的词距离更近 。**

#### 带负采样的跳字（skip - gram）模型 The skip - gram model with negative sampling (HW2)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130202202.png)

中文翻译 （作业2） 
- 符号表示更类似于课堂内容和作业2：$$J_{neg - sample}(u_{o},v_{c},U)=-\log\sigma(u_{o}^{T}v_{c})-\sum_{k\in\{K\ sampled\ indices\}}\log\sigma(-u_{k}^{T}v_{c})$$
- 我们采用 $k$ 个负样本（使用词的概率 (using word probabilities) ）
- 最大化真实外部(real outside word)词出现的概率，最小化随机词(random words)出现在中心词周围的概率 - 按照 $P(w)=U(w)^{3/4}/Z$ 进行采样，即一元语法分布 $U(w)$ 的 $3/4$ 次方（我们在起始代码中提供了此函数）,$Z$ 是归一化常数。
- **这个幂次使得不太频繁出现的词被采样的频率更高**(The power makes less frequent words be sampled more often )
1. **负采样操作**：从词的概率分布中选取 $k$ 个负样本。目的是通过训练，让模型能够区分真实的上下文词（最大化其出现概率）和随机选取的噪声词（最小化其在中心词周围出现的概率）。 
2. **采样分布**：采用 $P(w)=U(w)^{3/4}/Z$ 的方式进行采样，其中 $U(w)$ 是一元语法分布（即单个词在语料库中出现的概率分布），对其取 $3/4$ 次方后再进行归一化（除以 $Z$ ，$Z$ 是归一化常数）。这样做的效果是**调整了词的采样概率，使得原本出现频率较低的词有更大机会被采样到，有助于更全面地学习词向量，避免模型过度偏向高频词。**

## 4. 为什么不直接捕捉共现次数 Why not capture co - occurrence counts directly?   Count based方法
![image.png|650](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130203152.png)
Building a co - occurrence matrix $X$ 
- 2 options: 
- Window: Similar to word2vec, use window around each word $\rightarrow$ captures some syntactic and semantic information 
- Word - document co - occurrence matrix will give general topics (all sports terms will have similar entries) leading to “Latent Semantic Analysis” 
4、 为什么不直接捕捉共现次数？ 
构建一个共现矩阵 $X$ 
- 两种选择：windows vs. full document 
- 窗口：类似于word2vec，使用每个词周围的窗口 $\rightarrow$ 捕捉一些句法和语义(syntactic and semantic)信息 
	- 和 word2vec 类似，以每个词为中心设置一个窗口，通过统计窗口内词的共现次数，能够捕捉到一些句法（如词的语法结构关系）和语义（如词的含义关联）方面的信息。**例如在 “我喜欢苹果” 这句话中，以 “苹果” 为中心词设置窗口，能捕捉到 “喜欢” 和 “苹果” 的共现关系等。**
- **词 - 文档共现矩阵将给出一般主题（所有体育术语将有相似的条目），从而引出“潜在语义分析”**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130203811.png)
- 窗口长度为 1（更常见的是 5 - 10）
	- Context window is specified by a number and the direction.<br>比如一个**Context Window Size = 2的示意图**如下： 
     ![](https://i-blog.csdnimg.cn/blog_migrate/c40984201263d3fbf7a25b911c3fbb07.png)
- 对称的（左上下文或右上下文无关紧要）
- Example corpus:
    - I like deep learning
    - I like NLP
    - I enjoy flying  
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130212340.png)

示例
从句子 “I like deep learning” 中提取出的词汇表为 `["I", "like", "deep", "learning"]`。
- 对于词 “I”：
    - 其窗口内右侧词为 “like”，所以 “I” 和 “like” 的共现次数为 1。“I” 与词汇表中其他词（“deep”、“learning”）在窗口 `window = 1` 的情况下没有共现，共现次数为 0。
- 对于词 “like”：
    - 其窗口内左侧词为 “I”，右侧词为 “deep”，所以 “like” 与 “I” 的共现次数为 1，“like” 与 “deep” 的共现次数为 1。“like” 与 “learning” 没有共现，共现次数为 0。
- 对于词 “deep”：
    - 其窗口内左侧词为 “like”，右侧词为 “learning”，所以 “deep” 与 “like” 的共现次数为 1，“deep” 与 “learning” 的共现次数为 1。“deep” 与 “I” 没有共现，共现次数为 0。
- 对于词 “learning”：
    - 其窗口内左侧词为 “deep”，所以 “learning” 与 “deep” 的共现次数为 1。“learning” 与 “I”、“like” 没有共现，共现次数为 0。
### (1)共现向量  Co-occurrence vectors
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130211823.png)

- 简单计数共现向量(Simple count co - occurrence vectors)
	- 向量的大小随着词汇量的增加而增大 
	- 维度非常高：需要大量的存储（尽管是稀疏的） 
	- 后续的分类模型存在稀疏性问题(Subsequent classification models have sparsity issues ) $\rightarrow$ 模型的鲁棒性较差( Models are less robust)
- 低维向量 
- 思路：在固定的少量维度中存储“大部分”重要信息：一个密集向量 - <mark style="background: #D2B3FFA6;">通常为25 - 1000维</mark>，类似于word2vec 
#### <1>Classic Method: Dimensionality Reduction on X (HW1)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130213602.png)
- Singular Value Decomposition of co - occurrence matrix $X$ (对共现矩阵 $X$ 进行奇异值分解)
- Factorizes $X$ into $U\Sigma V^{T}$, where $U$ and $V$ are orthonormal(将 $X$ 分解为 $U\Sigma V^{T}$，其中 $U$ 和 $V$ 是正交矩阵) 
- Retain only $k$ singular values, in order to generalize. $\hat{X}$ is the best rank - $k$ approximation to $X$, in terms of least squares. Classic linear algebra result. Expensive to compute for large matrices. (为了实现泛化，仅保留 $k$ 个奇异值。就最小二乘法而言，$\hat{X}$ 是 $X$ 的最佳 $k$ 秩近似。这是经典的线性代数结果。对于大型矩阵，计算成本很高。 )
 1. **奇异值分解原理**：将共现矩阵 $X$ 分解为三个矩阵的乘积 $U\Sigma V^{T}$。其中 $U$ 和 $V$ 是正交矩阵，具有很好的数学性质；$\Sigma$ 是对角矩阵，对角线上的元素就是奇异值，它们反映了矩阵 $X$ 的重要特征。 
 2. **降维操作与效果**：为了达到降维目的并实现一定的泛化能力，**只保留 $k$ 个最大的奇异值，其余的奇异值设为0，这样得到的矩阵 $\hat{X}$ 就是原矩阵 $X$ 的一个最佳 $k$ 秩近似（从最小二乘法的角度来看，它与原矩阵 $X$ 的误差最小）**。通过这种方式，可以将高维的共现矩阵转换为低维表示。然而，对于大型的共现矩阵，奇异值分解的计算量非常大，需要耗费较多的计算资源和时间。 这也是在应用该方法时需要考虑的问题。
#### <2>对 $X$ 的改进方法 Hacks to X (several used in Rohde et al. 2005 in COALS)
![image.png|750](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130214254.png)
 - Running an **SVD** on raw counts **doesn’t work well** 
 - **Scaling the counts(计数进行缩放)** in the cells can help a lot 
	 - Problem: **功能词（the, he, has）过于频繁** function words (the, he, has) are too frequent $\rightarrow$ syntax has too much impact(语法影响太大). 
	 - Some fixes(解决方法): 
		 - **log** the frequencies 
		 - $\min(X, t)$, with $t \approx 100$ <mark style="background: #D2B3FFA6;">限制高频词</mark>
		 - **Ignore the function words** 
 - Ramped windows(**渐变窗口**) that count closer words more than further away words(<mark style="background: #D2B3FFA6;">对距离更近的词的计数权重高于距离较远的词</mark> )
 - Use **Pearson correlations(皮尔逊相关性)** instead of counts, then set negative values to 0 (<mark style="background: #D2B3FFA6;">然后将负值设为0</mark>)
 - Etc. 
 1. **渐变窗口**：采用渐变窗口的策略，即对与中心词距离更近的词赋予更高的计数权重，而距离较远的词权重较低。这样可以更合理地反映词之间的语义关联程度，因为通常距离近的词语义相关性更强。 
 2.  **相关性代替计数**：使用皮尔逊相关性来代替原始的共现计数，然后将得到的负值设为0。<mark style="background: #D2B3FFA6;">皮尔逊相关性可以衡量词之间的线性相关程度</mark>，相比于单纯的计数，可能更能捕捉词之间的语义关系，设负相关值为0可能是为了简化或突出正向的语义关联。这些方法都是为了改进基于共现矩阵的降维等操作，以更好地提取词的语义信息。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130215712.png)

## 5. 迈向 GloVe：基于计数的方法与直接预测的方法 Towards GloVe: Count based vs. direct prediction
![image.png|671](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130215925.png)

|                                                基于计数的方法                                                |                                                                 直接预测的方法                                                                 |
| :---------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| 潜在语义分析（LSA）、超平面词表示（HAL，Lund & Burgess提出）、COALS、赫林格主成分分析（Hellinger - PCA，Rohde等人，Lebret & Collobert提出） | 跳字模型/连续词袋模型（Skip - gram/CBOW，Mikolov等人提出）、神经网络语言模型（NNLM）、分层对数线性模型（HLBL）、循环神经网络（RNN，Bengio等人；Collobert & Weston；Huang等人；Mnih & Hinton提出） |
|                                                 训练速度快                                                 |                                                                随语料库规模扩展                                                                 |
|                                               统计信息使用高效                                                |                                                                统计信息使用低效                                                                 |
|                                              主要用于捕捉词的相似性                                              |                                                              在其他任务上产生更好的性能                                                              |
|                                            对大量计数给予不成比例的重要性                                            |                                                             可以捕捉词相似性之外的复杂模式                                                             |
 解释 
 1. **基于计数的方法特点** - **训练速度**：训练过程通常较快。 - **统计信息利用**：能高效地利用语料库中的统计信息，比如通过构建共现矩阵等方式。 - **应用重点**：主要聚焦于捕捉词与词之间的相似性。 - **局限性**：对共现计数中出现频率高的情况给予了过高的重要性，可能导致对低频词信息的忽视。 
 2. **直接预测的方法特点** - **规模适应性**：模型性能通常会随着语料库规模的增大而提升。 - **统计信息利用**：相对而言，在统计信息的利用效率上较低。 - **任务表现**：不仅能学习词向量，还能在诸如文本分类、情感分析等其他自然语言处理任务上取得较好的性能。 - **模式捕捉能力**：能够捕捉到词相似性之外的更复杂的语义和语法模式。这些对比有助于理解不同词向量学习方法的优势与不足，为选择合适的方法或进一步改进模型提供参考。

### (1)在向量差异中编码意义Encoding meaning in vector differences 
\[Pennington, Socher, and Manning, EMNLP 2014\]
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130221048.png)
关键见解：共现概率的比率可以编码意义成分(Crucial insight: Ratios of co - occurrence probabilities can encode meaning components)

 1. **研究来源与核心观点**：该内容来自2014年EMNLP会议上Pennington、Socher和Manning的研究。其核心观点是共现概率的比率能够对意义成分进行编码。这意味着可以通过分析不同词与其他词的共现概率之比，来挖掘词的语义信息。 
 2. **概率数据示例**：表格中给出了不同的 $x$ 值（固体、气体、水、时尚）分别在给定词“ice（冰）”和“steam（蒸汽）”条件下的共现概率 $P(x|ice)$ 和 $P(x|steam)$ 。例如，“固体”在“冰”的语境下的共现概率 $P(x = solid|ice)$ 为 $1.9\times10^{-4}$ ，在“蒸汽”的语境下的共现概率 $P(x = solid|steam)$ 为 $2.2\times10^{-5}$ 。 
 3. **概率比率分析**：通过计算 $P(x|ice)$ 与 $P(x|steam)$ 的比率，可以发现一些语义相关的信息。比如对于“固体”和“气体”，它们与“冰”和“蒸汽”的共现概率比率差异较大（8.9和 $8.5\times10^{-2}$ ），这反映了“冰”和“蒸汽”在与“固体”“气体”的语义关联上有明显不同；而“水”的比率为1.36，说明“水”与“冰”“蒸汽”都有一定关联；“时尚”的比率接近1（0.96），**表明“时尚”与“冰”“蒸汽”在这种共现关系上没有明显的偏向性。这种通过共现概率比率来编码意义的方式，为词向量表示语义提供了一种新的思路。**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130223926.png)
 问：我们如何在词向量空间中捕捉共现概率的比率作为线性意义成分？ 答：对数双线性模型：$w_{i}\cdot w_{j}=\log P(i|j)$ 利用向量差：$w_{x}\cdot(w_{a}-w_{b})=\log\frac{P(x|a)}{P(x|b)}$ 
  1. **问题核心**：探讨在词向量空间里将共现概率的比率表示为线性意义成分的方法。这是在词向量研究中关于如何更好地编码语义信息的问题。
  2. **对数双线性模型解答** 
	  - $w_{i}\cdot w_{j}=\log P(i|j)$：通过词向量的点积来表示在词$j$出现的条件下词$i$出现的概率的对数。这种方式将词的共现概率信息融入到词向量的运算中，利用点积的线性运算性质来捕捉语义关联。 
	  - $w_{x}\cdot(w_{a}-w_{b})=\log\frac{P(x|a)}{P(x|b)}$：利用向量差进一步拓展，该式表示词$x$的向量与词$a$和词$b$的向量差做点积，结果等于在词$a$和词$b$出现的条件下词$x$出现的概率之比的对数。通过这种向量差的点积运算，可以捕捉到不同词之间基于共现概率比率的语义关系，例如在类比任务中可以体现词之间的语义相似性和差异性等关系，从线性运算的角度对语义信息进行编码和表示。
### (2)Combining the best of both worlds 
GloVe [Pennington, Socher, and Manning, EMNLP 2014]
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130223850.png)
$$w_{i}\cdot w_{j}=\log P(i|j)$$$$J = \sum_{i,j = 1}^{V}f(X_{ij})(w_{i}^{T}\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^{2}$$
- 训练速度快 
- 可扩展到大型语料库 
- 即使在小语料库和小向量情况下也有良好性能 
1. **模型简介**：GloVe（Global Vectors for Word Representation）是由Pennington、Socher和Manning在2014年EMNLP会议上提出的词向量模型，旨在融合基于计数和直接预测两类方法的优点。 
2. **核心公式** 
	- $w_{i}\cdot w_{j}=\log P(i|j)$ ：表示两个词向量 $w_{i}$ 和 $w_{j}$ 的点积等于在词 $j$ 出现的条件下词 $i$ 出现的概率的对数。通过这种方式将词的共现概率信息融入到词向量的表示中。 
	- $J = \sum_{i,j = 1}^{V}f(X_{ij})(w_{i}^{T}\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^{2}$ ：这是GloVe模型的目标函数。其中 $V$ 是词汇表大小，$X_{ij}$ 是词 $i$ 和词 $j$ 的共现次数，<mark style="background: #D2B3FFA6;">$f(X_{ij})$ 是一个权重函数，用于调整不同共现次数的贡献程度</mark>；$w_{i}$ 和 $\tilde{w}_{j}$ 是词 $i$ 和词 $j$ 的不同向量表示，$b_{i}$ 和 $\tilde{b}_{j}$ 是偏置项。**该目标函数通过最小化预测的共现对数与实际共现对数之间的差异来训练词向量。** 
3. **模型特点** - **训练速度**：具有较快的训练速度，相比一些其他模型能够在更短时间内完成训练。 - **语料库适应性**：可以很好地扩展到大型语料库上，随着语料库规模的增大，依然能有效学习词向量。 - **性能稳定性**：即使在语料库规模较小和词向量维度较小的情况下，也能表现出良好的性能，具有较强的鲁棒性。

### (3)GloVe results 模型结果
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130224318.png)
## 6. 怎样评估词向量 How to evaluate word vectors?
![image.png|676](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130224856.png)
- 与自然语言处理中的一般评估相关：内在评估与外在评估(Intrinsic vs. extrinsic)
- 内在评估(Intrinsic)：
    - 在特定 / 中间子任务上进行评估
    - 计算速度快
    - 有助于理解系统
    - **除非与实际任务建立关联，否则其实际帮助作用不明确**
- 外在评估(extrinsic)：
    - 在实际任务上进行评估
    - 计算准确率可能需要**很长时间**
    - 不清楚是子系统本身的问题，还是其交互作用或其他子系统的问题
    - 如果用另一个子系统精确替换一个子系统能提高准确率→成功！
### (1)词向量的内在评估 Intrinsic word vector evaluation
![image.png|731](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130225250.png)
- 词向量类比(Word Vector Analogies)
	- man : woman 
	- king : ?
- 通过相加后的余弦距离在捕捉直观语义和句法类比问题上的表现来评估词向量 Evaluate word vectors by how well their cosine distance after addition captures intuitive semantic and syntactic analogy questions
- 在搜索中排除输入词！**Discarding the input words** from the search!
- Problem: What if the information is there but not linear?(如果信息存在但不是线性的怎么办？)
#### <1>GloVe可视化：公司-CEO
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130225629.png)
#### <2>GloVe可视化：比较级和最高级
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130225721.png)

- Glove 有一些线性特性，但不完美
#### <3>类比评估与超参数  Analogy evaluation and hyperparameters  
GloVe 词向量评估
Glove word vectors evaluation
![image.png|693](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130230129.png)
1.  **Sem.**：语义方面得分，衡量模型在捕捉语义类比关系上的表现，得分越高表明对语义类比的处理能力越强。
    - **Syn.**：句法方面得分，体现模型在处理句法类比关系上的性能，得分越高表示句法类比能力越好。
    - **Tot.**：总得分数，综合了语义和句法得分，反映模型在类比任务上的整体表现。
2. **模型表现**：从表格数据可以看出，GloVe 模型在语义得分上表现最佳（77.4），在总得分上也领先（71.7），说明其在捕捉语义和句法类比关系的综合能力上相对其他模型有优势。 不同的 SVD 变体和其他模型如 CBOW、SG 也各有不同的表现，展示了不同模型在类比任务中的差异。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130230542.png)
- 更多数据有帮助 More data helps
- Wikipedia is better than news text!
- Dimensionality
- Good dimension is ~300  较好的维度约为 300
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130230824.png)


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250130230948.png)
1. 从数据来看，随着数据规模的增大，模型在各数据集上的相关性得分有提升趋势。例如，SVD - L 模型在数据规模从 60 亿增加到 420 亿时，多个数据集上的得分都有所提高；GloVe 模型在 420 亿规模时，在多个数据集上的表现优于 60 亿规模时，且在一些数据集上（如 MC、RG）的得分相对其他模型较高，说明其词向量距离与人类判断的相关性较好。
2. **模型改进**：GloVe 论文中的一些方法也能对跳字（SG）模型起到改进作用，例如对两个向量取平均值等操作。 这表明不同模型之间可以相互借鉴优化思路。
### (2)词向量的外在评估 Extrinsic word vector evaluation
![image.png|721](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131000351.png)

- 词向量的外在评估：本课程中所有后续的自然语言处理任务。很快会有更多示例。
- 一个好的词向量应该能直接起作用的例子：命名实体识别(**named entity recognition**)，即识别对个人、组织或地点的引用(: identifying references to a person, organization or location)

## 7. 词义与词义歧义 Word senses and word sense ambiguity
![image.png|590](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131000854.png)

- Most words have lots of meanings! 大多数词都有很多含义！
	- Especially common words  常见词
	- Especially words that have existed for a long time  存在时间很长的词 
- Example: pike （梭子鱼；长矛等）
- Does one vector capture all these meanings or do we have a mess? 一个向量能否捕捉所有这些含义，还是会一团糟？ 
1. **词义现象**：指出在自然语言中，大多数词语具有多种含义，特别是常见词以及存在时间久远的词。这是因为语言在发展过程中，词语会根据不同的使用场景和历史演变产生多种语义。 
2. **示例说明**：以“pike”一词为例，它可以表示“梭子鱼”，也可以表示“长矛”等不同含义，体现了一词多义现象。
3. **问题探讨**：提出在词向量表示中面临的一个关键问题，即一个词向量是否能够捕捉一个词的所有不同含义。如果不能，可能会导致在自然语言处理任务中出现语义混淆等问题；而如果能，**如何实现这种有效的多义性表示也是一个挑战**。这引发了对于如何更好地构建词向量以处理词义歧义的思考。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131001135.png)
pike 
- A sharp point or staff 
- A type of elongated fish 
- A railroad line or system 
- A type of road 
- The future (coming down the pike) 
- A type of body position (as in diving) 
- To kill or pierce with a pike - To make one’s way (pike along) 
- In Australian English, pike means to pull out from doing something: I reckon he could have climbed that cliff, but he piked! 
中文翻译 pike 
- 尖端；长矛 
- 一种细长的鱼（梭子鱼） 
- 铁路线或铁路系统 
- 一种道路 
- 未来（coming down the pike 意为即将来临）
- 一种身体姿势（如在跳水运动中） 
- 用长矛杀死或刺穿 
- 行进（pike along 意为前行） 
- 在澳大利亚英语中，pike 表示退出做某事：我认为他本可以爬上那座悬崖，但他放弃了！
- 解释 **展示了“pike”这个单词的多种词义，体现了自然语言中一词多义的现象**。这些词义涵盖了名词、动词等不同词性，以及不同领域和语境下的含义，如武器、动物、交通、时间、动作等方面。这说明了在自然语言处理中处理词义时面临的复杂性，一个词在不同的上下文中可能具有截然不同的含义。

### (1)通过全局上下文和多个词原型改进词表示（Huang 等人，2012 年） Improving Word Representations Via Global Context And Multiple Word Prototypes 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131001532.png)
- Idea: Cluster word windows around words, retrain with each word assigned to multiple different clusters bank₁, bank₂, etc.对词周围的词窗口进行聚类，将每个词分配到多个不同的簇（如 bank₁、bank₂ 等）中并重新训练
 1. **论文及核心思路**：该内容源自Huang等人2012年的研究，旨在改进词表示。**核心想法是利用词的上下文信息进行聚类操作。具体来说，围绕每个词确定一个词窗口（即该词周围的一些词构成的集合），然后将这些词窗口聚类成不同的簇。** 
 2. **训练方式**：每个词会被分配到多个不同的簇中，之后基于这种分配方式重新训练词向量模型。这样做的目的是**让词能够从多个不同的上下文“原型”（即不同的簇代表的上下文模式）中学习语义信息，从而更好地处理一词多义等问题**，使得词向量能够更准确地表示词在不同语境下的含义。例如，“bank”这个词在不同语境下可能表示“银行”或“河岸”，通过这种聚类和多簇分配训练的方式，有望让模型学习到不同含义对应的不同上下文特征，进而提升词表示的质量。 图中展示的可能是部分词及其所在簇的关系等相关可视化内容，但因图片清晰度等限制，具体细节较难完全辨识。
 3. 实践上使用不适合，太复杂，需要区分语义，而且不同语义是有歧义的，不容易区分

### (2)词义的线性代数结构及其在多义词上的应用 Linear Algebraic Structure of Word Senses, with Applications to Polysemy (Arora, …, Ma, …, TACL 2018)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250131003502.png)

 - Different senses of a word reside in a linear superposition (weighted sum) in standard word embeddings like word2vec 
 - $v_{pike}=\alpha_{1}v_{pike_{1}}+\alpha_{2}v_{pike_{2}}+\alpha_{3}v_{pike_{3}}$ 
 - Where $\alpha_{1}=\frac{f_{1}}{f_{1}+f_{2}+f_{3}}$, etc., for frequency $f$ 
 - Surprising result: - Because of ideas from sparse coding you can actually separate out the senses (providing they are relatively common)! 

 - 在像word2vec这样的标准词嵌入中，一个词的不同词义存在于线性叠加（加权和）中 
	 - $v_{pike}=\alpha_{1}v_{pike_{1}}+\alpha_{2}v_{pike_{2}}+\alpha_{3}v_{pike_{3}}$ - 其中 $\alpha_{1}=\frac{f_{1}}{f_{1}+f_{2}+f_{3}}$ 等，$f$ 为频率 
 - 惊人的结果： 
 - 由于稀疏编码的思想，实际上可以分离出这些词义（前提是它们相对常见）！ 
 1. **研究主题**：来自Arora等人2018年发表于TACL的研究，聚焦于词义的线性代数结构以及其在多义词处理上的应用。 
 2. **词义表示原理**：在word2vec等标准词嵌入模型中，**一个多义词（如“pike”）的不同词义向量通过线性叠加（加权和）的形式存在**。公式$v_{pike}=\alpha_{1}v_{pike_{1}}+\alpha_{2}v_{pike_{2}}+\alpha_{3}v_{pike_{3}}$ 表示“pike”的词向量由其不同词义对应的词向量加权求和得到，权重 $\alpha_{i}$ 由对应词义出现的频率 $f_{i}$ 决定。 
 3. **研究成果**：**基于稀疏编码的思想，在词义相对常见的情况下，能够将多义词的不同词义分离出来**。这为解决多义词在词向量表示中的语义混淆问题提供了新的思路和方法。表格中展示了与“tie”相关的一些词，可能是用于说明词义分离或相关语义分析的示例，但具体用途需结合更多背景信息确定。
 4. 由于稀疏编码的思想和技术，<mark style="background: #D2B3FFA6;">在多义词的不同词义相对常见的情况下，能够将它们分离出来。稀疏编码的基本理念是用尽可能少的 “基向量” 的线性组合来表示数据。在多义词处理中，可将不同词义看作 “基向量”。通过特定的算法和计算，依据词在不同语境中的出现模式和频率等信息，从多义词的综合向量表示中解析出各个具体的词义向量。</mark>

英文：Lecture 3: Neural net learning: Gradients by hand (matrix calculus) and algorithmically (the backpropagation algorithm) 中文：第3讲：神经网络学习：手动计算梯度（矩阵微积分）和通过算法计算梯度（反向传播算法）