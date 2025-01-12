---
created: 2024-11-10T20:08
updated: 2024-11-17T11:52
---
 

参考链接：https://zhuanlan.zhihu.com/p/338488036

#### 文章目录

*   [BLEU的含义](#BLEU_7)
*   [BLEU如何进行评估的](#BLEU_27)

  
主要是为了解决如何判断模型翻译语句的质量的问题。  
BLEU 可以低成本，快速的实现对模型结果的评估，从而促进模型架构的发展

[BLEU](https://so.csdn.net/so/search?q=BLEU&spm=1001.2101.3001.7020)的含义
-----------------------------------------------------------------------

BLEU的全名为：bilingual evaluation understudy，即：双语互译质量评估辅助工具。  
用于评估[机器翻译](https://so.csdn.net/so/search?q=%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91&spm=1001.2101.3001.7020)质量的好坏。

**设计思想：**  
机器翻译结果越接近专业人工翻译的结果，则越好。

BLEU算法实际上就是在判断两个句子的相似程度，即拿这个句子的标准人工翻译与机器翻译的结果作比较。

BLEU 并不是拿一个对应的参考翻译来做比较，而是多参考翻译，最后算出一个综合分数。 **其分数值越高越好**。

**优点：**  
方便、快速、结果有参考价值

**缺点：**

1.  不考虑语言表达（语法）上的准确性；
2.  测评精度会受常用词的干扰；
3.  短译句的测评精度有时会较高；
4.  没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定；

BLEU如何进行评估的
-----------

BLEU的评估算是也是在不断改进的。  
最先提出的算法是这样的：  
**两个句子，S1和S2，S1里头的词出现在S2里头越多，就说明这两个句子越一致。**、

改进之后的是这样的：  
考虑了the, on这样的词，所以极易造成翻译结果低劣评分结果却贼高的情况。![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3ddfe0b823e53c8b68425e4923f11c0b.png)
- 在BLEU分数的计算中，截断计数（clipping count）是一种调整机制，用来解决翻译中可能出现的重复词语导致的问题。在原始的精确度计算中，如果机器翻译的结果中某个词语出现了多次，而该词语在参考翻译中只出现了一次，那么原始的精确度计算会将所有的出现都视为正确的匹配。这可能会导致对含有大量重复词语的翻译给予过高的评价。

- 为了避免这种情况，BLEU引入了截断计数的概念。具体做法是，对于机器翻译结果中的每一个n-gram，其计数不会超过该n-gram在参考翻译中出现的最大次数。换句话说，**即使某个n-gram在机器翻译的结果中出现了很多次，但在计算匹配数时，最多只能算作参考翻译中出现的次数。** 这样做可以确保那些重复词语不会过度地增加匹配的数量，从而更加准确地反映翻译的质量。

改进的第三种方法：BLEU多元精度（n-gram precision）  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b272997a3bdc3aa97fe9b3ef2491e685.png)  
改进的n-gram精度得分可以用来衡量翻译评估的充分性和流畅性两个指标：一元组属于字符级别，关注的是翻译的充分性，就是衡量你的逐字翻译能力； 多元组上升到了词汇级别的，关注点是翻译的流畅性，词组准了，说话自然相对流畅了。所以我们可以用多组多元精度得分来衡量翻译结果的。

**关于n-gram的另一种解释：**  
根据n-gram可以划分成多种评价指标，常见的指标有BLEU-1、BLEU-2、BLEU-3、BLEU-4四种，其中n-gram指的是连续的单词个数为n  
BLEU-1衡量的是单词级别的准确性，更高阶的bleu可以衡量句子的流畅性。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8447069ba201215717a6ed7bc4a14958.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ef3600713bf6197a55feec4c616f5509.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/75bc5306ed05f9e6701bd42457550ccf.png)

For example：  
candidate: the cat sat on the mat

reference: the cat is on the mat

那么各个bleu的值如下：

就 bleu2 ,对 candidate中的5个词，{the cat，cat sat，sat on，on the，the mat} ，查找是否在reference中，发现有3个词在reference中，所以占比就是0.6  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2669eabed73588db03b0e1b078756cf6.png)

**更多详情：**

https://zhuanlan.zhihu.com/p/338488036  
https://zhuanlan.zhihu.com/p/223048748

 

文章知识点与官方知识档案匹配，可进一步学习相关知识

[Python入门技能树](https://edu.csdn.net/skill/python/python-3-248?utm_source=csdn_ai_skill_tree_blog)[人工智能](https://edu.csdn.net/skill/python/python-3-248?utm_source=csdn_ai_skill_tree_blog)[自然语言处理](https://edu.csdn.net/skill/python/python-3-248?utm_source=csdn_ai_skill_tree_blog)462590 人正在系统学习中

本文转自 <https://blog.csdn.net/NGUever15/article/details/123197549?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522CFB188C5-EB84-4E4E-A82B-EB4EEA0A430A%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=CFB188C5-EB84-4E4E-A82B-EB4EEA0A430A&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-123197549-null-null.142^v100^pc_search_result_base5&utm_term=BLEU%20score&spm=1018.2226.3001.4187>，如有侵权，请联系删除。