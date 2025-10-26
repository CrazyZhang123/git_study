---
created: 2025-01-12T15:10
updated: 2025-01-12T15:10
---
关于HugBert系列，前情提要
----------------

*   [【HugBert01】Huggingface Transformers，一个顶级自然语言处理框架](https://zhuanlan.zhihu.com/p/141527015)
*   [【HugBert02】热身运动：安装及向量编码](https://zhuanlan.zhihu.com/p/143161582)
*   [【HugBert03】最后一英里：下游NLP任务的微调 - 知乎](https://zhuanlan.zhihu.com/p/149779660)

写到第4篇，感觉还是很有必要把GLUE拿出来仔细捋捋。想当年刚刚入行的时候，也是被老师傅摁在诸如Reuters-21578之类的经典数据集上摩擦再摩擦。后来慢慢理解到，无论是机器学习任务还是自然语言处理的任务，从相关的数据集和基准指标入手，是一个非常好的研究方法，它会告诉你的不限于：

*   这项任务的目的和实际应用场景
*   数据集加工处理流程是怎样的，有哪些常见的坑
*   评估任务的指标有哪些，为什么选用这些指标
*   常用的[baseline算法](https://zhida.zhihu.com/search?content_id=121869437&content_type=Article&match_order=1&q=baseline%E7%AE%97%E6%B3%95&zhida_source=entity)及指标如何
*   当今该基准测试集上的SOTA的算法是哪些
*   ……

**对于理解BERT以及Huggingface Transformers来说，GLUE是一个非常重要的数据集。**

1 GLUE为何产生？
-----------

**GLUE**的全称是**General Language Understanding Evaluation**，在2018年，由纽约大学、华盛顿大学以及DeepMind的研究者们共同提出。这个基准由一系列自然语言理解数据集/任务组成，最主要的目标是鼓励开发出能够**在任务之间共享通用的语言知识的模型**。BERT正是建立在这样一种预训练知识共享的基础之上，在刚推出的时候狂刷11项GLUE任务的纪录，开启了一个新的时代。

在迁移学习模型盛行之前，NLP领域表现最好的都是面向专项任务的端到端模型，比如在命名实体识别任务上的BiLSTM+CRF。

而迁移学习范式则普遍是一对多的模式，即先进行预训练，再剥离预训练模型的输入输出层，得到由中间的神经网络所表达的通用知识层，再根据不同的任务适配不同的输入输出层，训练少量epoch后得到最终的模型。

因此，要评估一个预训练语言模型的好坏，之前单一任务的指标就不合适了，GLUE正是带着这样的使命面世的。

![](https://pic2.zhimg.com/v2-3b6a2bf3bbfea27a4a3a51d1bf8e04bb_1440w.jpg)

GLUE主要由9项任务组成，一个预训练模型在9项任务上的得分的平均即为它的最终得分。

2 GLUE由哪些任务组成？
--------------

下面这张表小结了GLUE中的9项主要任务。

![](https://pic3.zhimg.com/v2-92a718915c81f90f5d133731681fa8fe_1440w.jpg)

这9项主要任务可以归为3大类，

*   CoLA和SST-2属于“单句任务”，输入是一个单一的句子，输出是某种分类判断
*   MRPC、STS-B、QQP属于“句子相似任务”，输入是一个句子对，输出是这两个句子的相似度衡量
*   MNLI、QNLI、RTE、WNLI这几项都是“推断任务”，输入是一个句子对，输出是后句相对于前句语义关系的一个推断

我们逐一探讨。

### 1）CoLA

全称The Corpus of Linguistic Acceptability，它是一个二分类任务，输入一个句子判断它是否在语法上可接受，可接受为1，不可接受为0. 数据的来源主要是一些语言学的书刊。示例如下：

> Digitize is my happiest memory - 0  
> Herman hammered the metal flat. - 1

### 2）SST-2

该数据集来自Stanford Sentiment Treebank，针对来自于电影评论的文本进行正向或者负向的区分，因为是二极情感分析任务，因此被称作SST-2。示例如下：

> standard horror flick formula 0  
> comes across as darkly funny , energetic , and surprisingly gentle 1

### 3）MRPC

全称是 Microsoft Research Paraphrase Corpus，从新闻数据源中自动摘取出一些句子对，人工标注了它们是否在语义上是相同的。示例如下：

> A: More than 60 percent of the company 's 1,000 employees were killed in the attack . B: More than 60 per cent of its 1,000 employees at the time were killed .  
> Label: 1  
> A: Scientists have reported a 90 percent decline in large predatory fish in the world 's oceans since a half century ago .  
> B: Scientists reported a 90 percent decline in large predatory fish in the world 's oceans since a half-century ago , a dire assessment that drew immediate skepticism from commercial fishers .  
> Label: 0

### 4）QQP

全称是Quora Question Pairs，来自于著名的社区问答网站Quora，这项任务要求判断给定的两个问句是否在问同一件事情。示例如下：

> A: Should animals be kept in zoos? What are your views on zoos? Why?  
> B: Why animals should not be kept in zoos?  
> Label: 1  
> A: How can I make money from Fivesquid?  
> B: How can I make money from Clikbank?  
> Label: 0

### 5）STS-B

全称是Semantic Textual Similarity Benchmark，输入的句子对来自于新闻头条、视频和图像说明文字和自然语言推断数据，人工对两个句子的相似度打了1至5的分数（浮点数），分数越高则说明相似度越高。这是9项任务中唯一一项回归任务。示例如下，最后的数字为得分：

> A group of people are marching in place.  
> A group of people are dancing in concert.  
> 1.700  
> A large black bird is sitting in the water.  
> The birds are swimming in the water.  
> 2.6  
> From Florida to Alaska, thousands of revelers vowed to push for more legal rights, including same-sex marriages.  
> Thousands of revellers, celebrating the decision, vowed to push for more legal rights, including same-sex marriages.  
> 4.200

### 6）MNLI

全称是Multi-Genre Natural Language Inference Corpus，文本蕴含类任务，数据来自于10种不同的数据源，包含演讲、小说、政府报告等。文本蕴含类的任务也由两个句子组成，一个句子称为premise前提，另一个句子称为hypothesis假设。需要判断在给定前提的情况下假设是否成立entailment、相反contradiction或者中性neutral。示例：

> What is so original about women?  
> There are a lot of things original about women.  
> neutral  
> The official indicated, however, that SBA's policy may change since future [certifications](https://zhida.zhihu.com/search?content_id=121869437&content_type=Article&match_order=1&q=certifications&zhida_source=entity) will need to be justified more specifically and will be subject to judicial review.  
> The official indicated that there would be no change in the SBA's policy.  
> contradiction  
> She waved me away impatiently.  
> She was impatiently waving me away.  
> entailment

### 7）QNLI

全称是Stanford Question Answering Dataset，来源于SQuAD阅读理解数据集。原任务是要求针对一个问题，从一篇wikipedia的文章中标识出可能存在的答案。QNLI将该问题转化成了句子对分类问题，问题句不变，但从文章中自动抽出一些句子来作为答案，需要判断该答案是否合理。示例：

> What are the power variants in USB 3.0 ports?  
> As with previous USB versions, USB 3.0 ports come in low-power and high-power variants, providing 150 mA and 900 mA respectively, while simultaneously transmitting data at SuperSpeed rates.  
> entailment  
> What property of wood has a correlation to its density?  
> There is a strong relationship between the properties of wood and the properties of the particular tree that yielded it.  
> not\_entailment

### 8）RTE

全称是Recognizing Textual Entailment，数据来自于年度文本蕴含挑战赛。句子对之间标识了[entailment](https://zhida.zhihu.com/search?content_id=121869437&content_type=Article&match_order=5&q=entailment&zhida_source=entity)和not\_entailment(neutral and contradiction)两类。示例：

> Intel has decided to push back the launch date for its 4-GHz Pentium 4 desktop processor to the first quarter of 2005.  
> Intel would raise the clock speed of the company's flagship Pentium 4 processor to 4 GHz.  
> not\_entailment  
> Hepburn, a four-time Academy Award winner, died last June in Connecticut at age 96.  
> Hepburn, who won four Oscars, died last June aged 96.  
> entailment

### 9）WNLI

全称为Winograd Schema Challenge，该任务要求阅读一个含有代词的句子并判断出该代词指代的内容。GLUE的维护者们人工地将可能的指代对象组织成句子，和原句放在一起，并标注其指代是否相同，因此也是一个句子对分类任务。示例如下：

> Jane knocked on the door, and Susan answered it. She invited her to come out.  
> Jane invited her to come out.  
> 1  
> It was a summer afternoon, and the dog was sitting in the middle of the lawn. After a while, it got up and moved to a spot under the tree, because it was cooler.  
> The dog was cooler.  
> 0

3 各项任务的评估指标
-----------

GLUE中的9项任务都有各自的评估指标，如下表小结：

![](https://pic4.zhimg.com/v2-097318c6e2158d87c8747db5975e3b8f_1440w.jpg)

这里做几点说明：

因为9项任务中除了STS-B是一项回归任务，其余都是分类任务，准确率是最常用的指标，即：准确率 = 相同标签样本数 / 总样本数。

MRPC和QQP两项任务，由于类别分布不均匀，在报告准确率的同时，报告了F1得分。F1是准确率与召回率的调合平均，即F1 = 2 \* (precision \* recall) / (precision + recall)

Pearson和Spearman相关系数用于衡量预测值和标签值之间的相关性。Pearson系数计算两个向量归一后的[协方差](https://zhida.zhihu.com/search?content_id=121869437&content_type=Article&match_order=1&q=%E5%8D%8F%E6%96%B9%E5%B7%AE&zhida_source=entity)， \\text{pearson\_score}=\\frac{cov(x,y)}{\\sigma\_x\\sigma\_y}\\text{pearson\_score}=\\frac{cov(x,y)}{\\sigma\_x\\sigma\_y} ；Spearman系数是衡量两个变量的依赖性的非参数指标。

在类别样本数非常不均衡的情况下，Matthews相关系数常常被认为是一种更好的评价指标，一般用于二分类问题，参考：[Matthews Correlation Coefficient is The Best Classification Metric You’ve Never Heard Of](https://link.zhihu.com/?target=https%3A//towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a) 。

如果一项任务同时报告了多个指标，则任务得分为多指标的简单平均。一个模型的GLUE综合得分是各项任务得分的宏平均（macro average）。

transformers中与GLUE指标计算相关的代码主要在data/metrics/\_\_init\_\_.py当中，底层函数基本调用的是scipy.stats以及sklearn.metrics中相关的函数。

4 诊断数据集
-------

GLUE另外一个比较有特点的地方是提供了一个小规模手工挑选的诊断数据集，其中每个样本都有针对性地探测模型在自然语言理解某一方面的能力，比如逻辑、语义、知识、谓词论元结构等，方便研究人员理解模型和找到改进方向。

![](https://pic4.zhimg.com/v2-1baeb522283ec600cce6383ef318ae91_1440w.jpg)

5 Baseline
----------

GLUE的作者们同时提供了若干Baseline模型。

最简单的模型基于一个句子向量编码器，采用了一个双层的BiLSTM结构（单向1500维）+ Max Pooling + 300维GloVe词嵌入向量。对于单句任务，句子向量化后直接传给分类器；对于句子对任务，两个句子独立编码为u, v，并且将u, v, |u-v|, u\*v的结果拼接起来传给分类器。分类器是一个含有单个512维隐藏层的MLP。

6 Leaderboard
-------------

GLUE的作者们在网站上提供了模型得分排行榜，

[](https://link.zhihu.com/?target=https%3A//gluebenchmark.com/leaderboard)

当前最好成绩：

![](https://pic1.zhimg.com/v2-b6a0775ad873e7a91f8d4f0879a9697e_1440w.jpg)

来自中国的团队非常亮眼，其中来自平安、百度和阿里达摩院研究团队提交的模型占据了榜上的前三甲，ALBERT+DAAF+NAS、ERNIE、StructBERT等BERT模型变种表现最佳。

**值得一提的是GLUE提供的人类专家的Baseline得分仅有87.1分，排名第13，被一众机器模型碾压。**

参考
--

*   官网：[GLUE Benchmark](https://link.zhihu.com/?target=https%3A//gluebenchmark.com/)
*   [GLUE Explained: Understanding BERT Through Benchmarks · Chris McCormick](https://link.zhihu.com/?target=https%3A//mccormickml.com/2019/11/05/GLUE/)
*   [\[1804.07461\] GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.07461)
*   GLUE数据集小结表格：[https://docs.google.com/spreadsheets/d/1BrOdjJgky7FfeiwC\_VDURZuRPUFUAz\_jfczPPT35P00/edit?usp=sharing](https://link.zhihu.com/?target=https%3A//docs.google.com/spreadsheets/d/1BrOdjJgky7FfeiwC_VDURZuRPUFUAz_jfczPPT35P00/edit%3Fusp%3Dsharing)
*   中文语言理解测评基准CLUE：[https://github.com/CLUEbenchmark/CLUE](https://link.zhihu.com/?target=https%3A//github.com/CLUEbenchmark/CLUE)
*   SuperGLUE，更难的GLUE：[https://w4ngatang.github.io/static/papers/superglue.pdf](https://link.zhihu.com/?target=https%3A//w4ngatang.github.io/static/papers/superglue.pdf)

* * *

Let's hug Bert together. 下篇再见。

本文转自 <https://zhuanlan.zhihu.com/p/151818251>，如有侵权，请联系删除。