

目录

收起

1 BLEU指标介绍

1.1 公式

2 示例

2.1 计算n-gram精度

2.2 计算 brevity penalty (BP)

2.3 计算BLEU得分

3 代码示例

4 论文链接

BLEU（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标，它通过比较候选译文与参考译文之间的n-gram匹配程度来衡量译文的准确性。下面是对BLEU指标的详细介绍，包括公式、示例以及代码示例。

1 BLEU指标介绍
----------

### 1.1 公式

BLEU分数的计算涉及以下几个关键步骤：

1.  **计算n-gram匹配**：对于给定的n值，计算候选译文中所有n-gram在参考译文中出现的次数。  
    
2.  **修正的n-gram精度**：为了避免重复计算，每个n-gram在参考译文中出现的最大次数（不超过其在候选译文中出现的次数）被用于计算修正的n-gram精度。  
    
3.  **几何平均**：对于所有考虑的n值（通常n取1到4），计算修正的n-gram精度的几何平均值。  
    
4.  **长度惩罚**：如果候选译文的长度短于参考译文的最短长度，则应用[长度惩罚因子](https://zhida.zhihu.com/search?content_id=253817123&content_type=Article&match_order=1&q=%E9%95%BF%E5%BA%A6%E6%83%A9%E7%BD%9A%E5%9B%A0%E5%AD%90&zhida_source=entity)。  

BLEU分数的公式如下：

$$\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)$$

其中：

*   BP是长度惩罚因子，长度惩罚因子 BP 的公式如下：

$$BP = \begin{cases} 1 & \text{if } l_c > l_r \\ \exp\left(1 - \frac{l_r}{l_c}\right) & \text{if } l_c \leq l_r \end{cases}$$

其中 $l_c$ 是候选译文的长度，$l_r$ 是参考译文的最短长度。

*   N 是考虑的最大n-gram长度（通常取4）。  
    
*   $w_n$ 是n-gram的权重（通常取均匀权重，即 $\frac{1}{N}$。  
    
*   $p_n$ 是修正的n-gram精度。  
    

修正的n-gram精度（Modified n-gram Precision）是BLEU（Bilingual Evaluation Understudy）评分指标中的一个关键组成部分。它用于衡量候选译文中n-gram的准确程度，同时避免了因多次匹配同一参考译文中的n-gram而导致的过度计分。

修正的n-gram精度的详细公式如下：

首先，定义一些符号：

*   $\text{count}_{\text{clip}}(n\text{-gram}) $：表示候选译文中某个n-gram在参考译文中出现的最大次数（不超过其在候选译文中出现的次数）。
*   $\text{count}(n\text{-gram})$ ：表示候选译文中某个n-gram出现的次数。
*   $\text{ref}_i$ ：表示第i个参考译文。
*   $C$ ：表示候选译文。

对于给定的候选译文 C和一组参考译文 ${\text{ref}_1, \text{ref}_2, \ldots, \text{ref}_R}$ ，修正的n-gram精度$p_n$ 的计算公式为：

$ p_n = \frac{\sum_{\text{n-gram} \in C} \text{count}_{\text{clip}}(n\text{-gram})}{\sum_{\text{n-gram} \in C} \text{count}(n\text{-gram})}$

其中，$\text{count}_{\text{clip}}(n\text{-gram})$ 的计算方式如下：

$\text{count}_{\text{clip}}(n\text{-gram}) = \min \left( \text{count}(n\text{-gram}), \max_{1 \leq i \leq R} \text{count}_{\text{ref}_i}(n\text{-gram}) \right)$

这里， $\text{count}_{\text{ref}_i}(n\text{-gram})$ 表示n-gram在第i个参考译文中出现的次数。

简而言之，**修正的n-gram精度是通过比较候选译文中的n-gram在参考译文中出现的最大次数（不超过其在候选译文中出现的次数）与候选译文中该n-gram的总出现次数来计算的。**这种方法确保了不会因为多次匹配同一参考译文中的n-gram而给予过高的分数。

在计算BLEU分数时，通常会考虑多个n-gram长度（如1-gram到4-gram），并对它们的修正精度进行几何平均，同时还会应用长度惩罚因子来考虑候选译文长度的影响。

2 示例
----

假设我们有以下候选译文和参考译文：

*   候选译文：`the cat is on the mat`
*   参考译文1：`the cat is sitting on the mat`
*   参考译文2：`there is a cat on the mat`

BLEU（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标，它通过比较候选译文和参考译文之间的n-gram重叠来计算得分。BLEU得分范围在0到1之间，得分越高表示候选译文与参考译文越接近。

### 2.1 计算n-gram精度

*   **1-gram**：
*   候选译文中的1-gram：`the`, `cat`, `is`, `on`, `the`, `mat`
*   参考译文1中的1-gram：`the`, `cat`, `is`, `sitting`, `on`, `the`, `mat`
*   参考译文2中的1-gram：`there`, `is`, `a`, `cat`, `on`, `the`, `mat`
*   匹配的1-gram：`the`, `cat`, `is`, `on`, `the`, `mat`
*   1-gram精度： $p_1 = \frac{6}{6} = 1  $  分子 2 + 1+1+1+1 = 6
*   **2-gram**：  
*   候选译文中的2-gram：`the cat`, `cat is`, `is on`, `on the`, `the mat`
*   参考译文1中的2-gram：`the cat`, `cat is`, `is sitting`, `sitting on`, `on the`, `the mat`
*   参考译文2中的2-gram：`there is`, `is a`, `a cat`, `cat on`, `on the`, `the mat`
*   匹配的2-gram：`the cat`, `cat is`, `on the`, `the mat`
*   2-gram精度： $p_2 = \frac{4}{5} = 0.8$   分子 1+1+0+1+1=4
*   **3-gram**：  
*   候选译文中的3-gram：`the cat is`, `cat is on`, `is on the`, `on the mat`
*   参考译文1中的3-gram：`the cat is`, `cat is sitting`, `is sitting on`, `sitting on the`, `on the mat`
*   参考译文2中的3-gram：`there is a`, `is a cat`, `a cat on`, `cat on the`, `on the mat`
*   匹配的3-gram：`the cat is`, `on the mat`
*   3-gram精度： $p_3 = \frac{2}{4} = 0.5 $ 分子 1+ 0+0+1=2
*   **4-gram**：  
*   候选译文中的4-gram：`the cat is on`, `cat is on the`, `is on the mat`
*   参考译文1中的4-gram：`the cat is sitting`, `cat is sitting on`, `is sitting on the`, `sitting on the mat`
*   参考译文2中的4-gram：`there is a cat`, `is a cat on`, `a cat on the`, `cat on the mat`
*   匹配的4-gram：无

4-gram精度：$p_4 = \frac{0}{3} = 0$

### 2.2 计算 brevity penalty (BP)

*   候选译文长度 c = 6
*   参考译文1长度 r_1 = 7

参考译文2长度r_2 = 7

*   最短参考译文长度 r=7
*   因为c < r ，所以$BP = e^{(1 - 7/6)} = e^{-1/6} \approx 0.846$

### 2.3 计算BLEU得分

*   假设权重 $w_n = \frac{1}{4} $（均匀权重）
*   BLEU 得分计算：

$$BLEU = 0.846 \cdot \exp\left(\frac{1}{4} \cdot 0 + \frac{1}{4} \cdot (-0.223) + \\ \frac{1}{4} \cdot (-0.693) + \frac{1}{4} \cdot (-\infty)\right)$$

由于 $\log 0 = -\infty$ ， BLEU 得分为0。

由于4-gram精度为0，BLEU得分为0。这表明候选译文与参考译文在4-gram级别上没有重叠。

3 代码示例
------

以下是一个使用Python和NLTK库计算BLEU分数的示例代码：

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 候选译文
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

# 参考译文列表
references = [
    ['the', 'cat', 'is', 'sitting', 'on', 'the', 'mat'],
    ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']
]

# 使用默认的平滑方法（方法1）计算BLEU分数
bleu_score = sentence_bleu(references, candidate)
print(f'BLEU score (default smoothing): {bleu_score:.4f}')

# 使用其他平滑方法（例如方法7）计算BLEU分数
smoothing_function = SmoothingFunction().method7
bleu_score_with_smoothing = sentence_bleu(references, candidate, smoothing_function=smoothing_function)
print(f'BLEU score (with smoothing): {bleu_score_with_smoothing:.4f}')

BLEU score (default smoothing): 0.0000
BLEU score (with smoothing): 0.4644
c:\Anaconda3\lib\site-packages\nltk\translate\bleu_score.py:552: UserWarning: 
The hypothesis contains 0 counts of 4-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
  warnings.warn(_msg)
```

请注意，NLTK的`sentence_bleu`函数默认使用了一种平滑方法（方法1），但你也可以选择其他平滑方法，如示例中的方法7。平滑方法用于处理n-gram匹配次数为0的情况，以避免BLEU分数为0。

运行上述代码将输出候选译文的BLEU分数，该分数反映了候选译文与参考译文之间的相似程度。BLEU分数的范围从0到1，分数越高表示译文质量越好。

4 论文链接
------

[《BLEU: a Method for Automatic Evaluation of Machine Translation》](https://link.zhihu.com/?target=https%3A//aclanthology.org/P02-1040.pdf)

本文转自 <https://zhuanlan.zhihu.com/p/23975563718>，如有侵权，请联系删除。