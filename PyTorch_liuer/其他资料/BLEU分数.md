---
created: 2024-11-17T11:53
updated: 2024-11-17T12:12
---
 

#### **BLEU综述：**

#### 

　　BLEU实质是对两个句子的**_共现词频率_**计算，但计算过程中使用好些技巧，追求计算的数值可以衡量这两句话的一致程度。  
　　BLEU容易陷入**_常用词_**和**_短译句_**的陷阱中，而给出较高的评分值。本文主要是对解决BLEU的这两个弊端的优化方法介绍。  
　　  
　　参考文献有二：  
　　- [《BLEU: a Method for Automatic Evaluation of Machine Translation 》](http://www.aclweb.org/anthology/P02-1040.pdf)  
　　 - [WIKIPEDIA中对BLEU的讲解](https://en.wikipedia.org/wiki/BLEU)  
　　

* * *

　　  

#### **一．BLEU是什么？**

#### 

　　首先要看清楚我们本篇文章的主人公是怎么拼写的——**{B-L-E-U}**，而不是{B-L-U-E}，简直了…..我叫了它两天的blue（蓝色）才发现原来e在u之前~~如果真要念出它的名字，音标是这样的：\[blε：\]\[blε：\]（波勒）。  
　　  
　　BLEU的全名为：bilingual evaluation understudy，即：**_双语互译质量评估辅助工具_**。它是用来评估机器翻译质量的工具。当然评估翻译质量这种事本应该由人来做，机器现在是无论如何也做不到像人类一样思考判断的（我想这就是自然语言处理现在遇到的瓶颈吧，随便某个方面都有牵扯上人类思维的地方，真难），但是人工处理过于耗时费力，所以才有了BLEU算法。

　　BLEU的设计思想与评判机器翻译好坏的思想是一致的：**_机器翻译结果越接近专业人工翻译的结果，则越好_**。BLEU算法实际上在做的事：判断两个句子的相似程度。我想知道一个句子翻译前后的表示是否意思一致，显然没法直接比较，那我就拿这个句子的标准人工翻译与我的机器翻译的结果作比较，如果它们是很相似的，说明我的翻译很成功。因此，**BLUE去做判断：一句机器翻译的话与其相对应的几个参考翻译作比较，算出一个综合分数。这个分数越高说明机器翻译得越好**。（注：BLEU算法是句子之间的比较，不是词组，也不是段落）  
　　  
　　**BLEU是做不到百分百的准确的，它只能做到个大概判断，它的目标也只是给出一个快且不差自动评估解决方案。**

* * *

#### **二．BLEU的优缺点有哪些？**

#### 

　　**优点**很明显：方便、快速、结果有参考价值  
　　  
　　**缺点**也不少，主要有：  

1.  1.  1\. 　不考虑语言表达（语法）上的准确性；  
        2.　 测评精度会受常用词的干扰；  
        3.　 短译句的测评精度有时会较高；  
        4\. 　没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定；

* * *

#### **三．如何去实现BLEU算法？**

#### 

　　首先，“机器翻译结果越接近专业人工翻译的结果，则越好”——要想让机器去评判一句话机器翻译好坏，得有两件工具：  

1.  **1.　衡量机器翻译结果越接近人工翻译结果的数值指标；**  
    **2.　一套人工翻译的高质量参考译文；**

　　其次，规范一下说法——  

1.  1.　对一个句子我们会得到好几种翻译结果（词汇、词序等的不同），我们将这些翻译结果叫做 **候选翻译集**（candidate1, candidate2, ……）;  
    2.　一个句子也会有好几个 **参考翻译**（reference1, reference2, ……）;  
    3.　我们下面计算的比值，说白了就是精度，记做 pnpnp\_n， n代表n-gram， 又叫做n-gram precision scoring—— **多元精度得分**（具体解释见3.2）；  
    4.　需要被翻译的语言，叫做源语言（source），翻译后的语言，叫做目标语言（target）；  
    　　

##### **3.1 最开始的BLEU算法**

　　其实最原始的BLEU算法很简单，我们每个人都有意无意做过这种事：两个句子，S1和S2，S1里头的词出现在S2里头越多，就说明这两个句子越一致。就像这样子：similarity(‘i like apple’, ‘i like english’)=2/3。  
　　  
　　分子是一个候选翻译的单词有多少出现在参考翻译中（出现过就记一次，不管是不是在同一句参考翻译里头），分母是这个候选翻译的词汇数。  
　　请看下面这个错误案例：

|  |  |  |  |  |  |  |  |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| Candidate | the | the | the | the | the | the | the |
| Reference1 | the | cat | is | on | the | mat |  |
| Reference2 | there | is | a | cat | on | the | mat |

计算过程：  

1.  1.　候选翻译的每个词——the，都在参考译文中出现，分子为7；  
    2.　候选翻译一共就7个词，分母为7；  
    3.　这个翻译的得分: 7/7 = 1！

　　很明显，这样算是错的，需要改进一下。  
　　  
　　  

##### **3.2 改进的多元精度（n-gram precision）**

　　专业一点，上面出现的错误可以理解为**_常用词干扰_**（over-generate “reasonable”words），比如the, on这样的词，所以极易造成翻译结果低劣评分结果却贼高的情况。  
　　  
　　另外，上面我们一个词一个词的去统计，以一个单词为单位的集合，我们统称uni-grams（一元组集）。如果是这样{“the cat”, “cat is”, “is on”, “on the”, “the mat”}，类似”the cat”两个相邻词一组就叫做bi-gram（二元组），以此类推：三元组、四元组、…、多元组（n-gram），集合变复数：n-grams。  
　　  
　　OK，上述算法问题其实处在分子的计算上，我们换成这个：  

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241117115853.png)

名词解释： _对于一条候选翻译——_  

1 .　 $Count_{w_i}$： 单词w_i的个数，如上表就是对“the”的计数为7；  
2.　 $Refj\_Count_{w_i}$:　词 w_i在第个 j参考翻译里出现的次数；  
3.　$Count_{w_i,j}^{clip}$：被称之为对于第 j个参考翻译，w_i的截断计数；  
4.　 $Count^{clip}$： w_i在所有参考翻译里的综合截断计数；

　　仍然看上表的举例，$Ref1\_Count_{'the'}$=2，所以$Count_{'the',1}^{clip}$=min(7,2)=2，同理$Count_{'the',2}^{clip}$=1，所以综合计数$Count^{clip}=max(1,2)$=2。  
　　  
　　分母不变，仍是候选句子的n-gram个数。这里分母为7。  
　　

　　**注**：这个地方的 分子截断计数 方法也不唯一，还有这样的：  
![image.png|495](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241117120651.png)

　　它是先对统计了在各个参考翻译里的最多出现次数 $Ref\_Count_{'the'}=max(2,1)$=2，然后 $Count^{clip}=min(7,2)$=1。  
　　  
　　其实 **_改进的n-gram精度得分可以用了衡量翻译评估的充分性和流畅性两个指标_**：一元组属于字符级别，关注的是翻译的充分性，就是衡量你的逐字逐字翻译能力； 多元组上升到了词汇级别的，关注点是翻译的流畅性，词组准了，说话自然相对流畅了。所以我们可以用多组多元精度得分来衡量翻译结果的。  


　　  

##### **3.3 改进的多元精度（modified n-gram precision）在文本段落翻译质量评估中的使用**

　　BLEU的处理办法其实还是一样，把多个句子当成一个句子罢了：  

$$
p_{n}=\frac{\sum_{c\in candidates}\sum_{n-gram \in c}Count\_{clip}(n-gram)}{\sum_{c^{'}\in candidates}\sum_{n-gram^{'}\in c^{'}}Count\_{clip}(n-gram^{'})}
$$
　　不要被这里的连加公式给欺骗了，它将候选段落的所有n-gram进行了截断统计作为分子，分母是候选段落的n-gram的个数。  
　　  
　　  

##### **3.4 将多个改进的多元精度（modified n-gram precision）进行组合**

　　在3.2提到，uni-gram下的指标可以衡量翻译的充分性，n-gram下的可以衡量翻译的流畅性，建议将它们组合使用。那么，应该如何正确的组合它们呢？  
　　  
　　没疑问，加总求和取平均。专业点的做法要根据所处的境况选择加权平均，甚至是对原式做一些变形。  
　　  
　　首先请看一下不同n-gram下的对某次翻译结果的精度计算：  
　　  
![这里写图片描述](https://img-blog.csdn.net/20170830113001250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE1ODQxNTc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

　　事实是这样，随着n-gram的增大，精度得分总体上成指数下降的，而且可以粗略的看成随着n而指数级的下降。我们这里采取**几何加权平均**，并且将各n-gram的作用视为等重要的，即取权重服从均匀分布。  

pave\=∏n\=1Npwnn−−−−−−⎷∑Nn\=1wn\=1∑Nn\=1wnexp(∑i\=1Nwn∗logpn)\=exp(1N∗∑i\=1Nlogpn)pave\=∏n\=1Npnwn∑n\=1Nwn\=1∑n\=1Nwnexp(∑i\=1Nwn∗logpn)\=exp(1N∗∑i\=1Nlogpn)

p\_{ave}=\\sqrt\[\\sum\_{n=1}^{N}w\_{n}\]{\\prod\_{n=1}^Np\_{n}^{w\_{n}}}=\\frac{1}{\\sum\_{n=1}^{N}w\_{n}}exp(\\sum\_{i=1}^{N}w\_{n}\*log^{p\_{n}})=exp(\\frac{1}{N}\*\\sum\_{i=1}^{N}log^{p\_{n}})  
pnpnp\_{n}为改进的多元精度， wnwnw\_{n}为赋予的权重。

　　对应到上图，公式简单表示为：  

pave\=exp(14∗(logp1+logp2+logp3+logp4))pave\=exp(14∗(logp1+logp2+logp3+logp4))

p\_{ave}=exp(\\frac{1}{4}\*(log^{p\_{1}}+log^{p\_{2}}+log^{p\_{3}}+log^{p\_{4}}))  
　　  

##### **3.5 译句较短惩罚（Sentence brevity penalty ）**

　　再仔细看改进n-gram精度测量，当译句比参考翻译都要长时，分母增大了，这就相对惩罚了译句较长的情况。译句较短就更严重了！比如说下面这样：

|  |  |  |  |  |  |  |  |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| Candidate | the | cat |  |  |  |  |  |
| Reference1 | the | cat | is | on | the | mat |  |
| Reference2 | there | is | a | cat | on | the | mat |

　　显然，这时候选翻译的精度得分又是1（12+12\\frac{1}{2}+\\frac{1}{2}）！**短译句**就是这样，很**容易得高分**…所以必须要设计一个有效的惩罚措施去控制。  
　　  
　　首先，定一个名词叫“**最佳匹配长度**”（best match length），就是，如果译句长度和任意一个参考翻译的长度相同，就认为它满足最佳匹配长度。这种情况下，就不要惩罚了，惩罚因子要设为1。  
　　

BP\={1…………if…c\>re1−rc……if…c≤rBP\={e1−rc……if…c≤r1…………if…c\>r

BP=\\lbrace^{1…………if…c>r}\_{e^{1-\\frac{r}{c}}……if…c \\leq r}

　　见上式，rr是一个参考翻译的词数，cc是一个候选翻译的词数，BP代表译句较短惩罚值。由此，最终BLEUBLEU值得计算公式为：  
　　

BLEU\=BP∗exp(∑n\=1Nwn∗logpn)BLEU\=BP∗exp(∑n\=1Nwn∗logpn)

BLEU=BP\*exp(\\sum\_{n=1}^{N}w\_{n}\*log^{p\_{n}})  
　　通过一次次的改进、纠正，这样的 **_BLEU算法已经基本可以快捷地给出相对有参考价值的评估分数了。做不到也不需要很精确，它只是给出了一个评判的参考线而已_**。

　　  

#### **以上就是BLEU知识点的主要内容。欢迎指正、补充。**

#### 

 

文章知识点与官方知识档案匹配，可进一步学习相关知识

[算法技能树](https://edu.csdn.net/skill/algorithm/?utm_source=csdn_ai_skill_tree_blog)[首页](https://edu.csdn.net/skill/algorithm/?utm_source=csdn_ai_skill_tree_blog)[概览](https://edu.csdn.net/skill/algorithm/?utm_source=csdn_ai_skill_tree_blog)65242 人正在系统学习中

本文转自 <https://blog.csdn.net/qq_31584157/article/details/77709454?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-77709454-blog-116102495.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-77709454-blog-116102495.235%5Ev43%5Econtrol&utm_relevant_index=5>，如有侵权，请联系删除。