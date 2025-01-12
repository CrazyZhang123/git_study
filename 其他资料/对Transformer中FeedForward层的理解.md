---
created: 2024-11-10T12:39
updated: 2024-11-10T12:39
---
 

#### 对FeedForward的理解

  
上一篇我们介绍了 [对Add&Norm层的理解](https://blog.csdn.net/weixin_51756104/article/details/127232344?spm=1001.2014.3001.5501)，有不大熟悉的可以看一下上篇文章。

今天来说一下Transformer中FeedForward层，首先还是先来回顾一下Transformer的基本结构：首先我们还是先来回顾一下Transformer的结构：Transformer结构主要分为两大部分，一是Encoder层结构，另一个则是Decoder层结构，Encoder 的输入由 Input Embedding 和 Positional Embedding 求和输入Multi-Head-Attention，然后又做了一个ADD&Norm，再通过Feed Forward进行输出。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/94d1bc4e446bb7f89d77bdd96f0e2348.png)  
FeedForward的输入是什么呢？是Multi-Head Attention的输出做了残差连接和Norm之后得数据，然后FeedForward做了两次线性[线性变换](https://so.csdn.net/so/search?q=%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2&spm=1001.2101.3001.7020)，为的是更加深入的提取特征。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/50cb6a35aa5a2f3cdd2725f54f5a4d86.png)  
可以看出在每次线性变换都引入了非线性激活函数Relu，在Multi-Head Attention中，主要是进行矩阵乘法，即都是线性变换，而线性变换的学习能力不如非线性变换的学习能力强，FeedForward的计算公式如下：max相当于Relu  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ce6d6677ef444833ad9cea5468c6aa94.png)

所以FeedForward的作用是：通过线性变换，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征

 

文章知识点与官方知识档案匹配，可进一步学习相关知识

[OpenCV技能树](https://edu.csdn.net/skill/opencv/opencv-a181ede3b8c7487fbcc212796c27ce77?utm_source=csdn_ai_skill_tree_blog)[OpenCV中的深度学习](https://edu.csdn.net/skill/opencv/opencv-a181ede3b8c7487fbcc212796c27ce77?utm_source=csdn_ai_skill_tree_blog)[图像分类](https://edu.csdn.net/skill/opencv/opencv-a181ede3b8c7487fbcc212796c27ce77?utm_source=csdn_ai_skill_tree_blog)29608 人正在系统学习中

本文转自 <https://blog.csdn.net/weixin_51756104/article/details/127250190?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522B972BEA6-83EA-435F-8752-AB1D72DB6A8C%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=B972BEA6-83EA-435F-8752-AB1D72DB6A8C&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-127250190-null-null.142^v100^pc_search_result_base5&utm_term=Feed%20Forward%20&spm=1018.2226.3001.4187>，如有侵权，请联系删除。