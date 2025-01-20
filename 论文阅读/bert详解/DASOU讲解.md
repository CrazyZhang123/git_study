---
created: 2025-01-17T21:56
updated: 2025-01-17T23:51
---
【BERT从零详细解读，看不懂来打我】 https://www.bilibili.com/video/BV1Ey4y1874y/?share_source=copy_web&vd_source=237c1070b237a59068e4928b71e5140e

### 基础架构

Bert使用的基础架构是transformer的encoder，而**BERT的模型架构是一个多层双向Transformer编码器**，base模型采用12层encoder,large模型采用24层encoder。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117215843.png)
![image.png|357](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117220220.png)
                    Base bert结构
base模型是12个encoder堆叠在一起，而不是transformer,transformer在attention is all you need里面，采用的是6个encoder(编码段),6个decoder(接码端)组合起来。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117220449.png)
                transformer架构
### Bert的输入

Bert的输入
**input = token emb + segment emb + position emb**
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117221134.png)

- 出现下面特色词元的原因是Bert有一个预训练任务是**下一句预测（NSP）任务，用于处理两个句子之间关系，这是一个二分类任务**
	- **每个序列的首个词元始终是一个特殊的分类词元（[CLS]）**。
	- 句子对会被合并到一个单一序列中。我们通过**两种方式来区分句子**。**用一个特殊词元（[SEP]）将它们分隔开**。
- 如何处理二分类任务，把CLS接一个二分类器。
>CLS向量不代表两(整个)个句子的语义信息。

三个嵌入理解
- token emb: 对词进行embeding
- segment emb: 对两个句子进行区分，可以看到标识每个句子，一个是EA，一个是EB。
- Position emb: 随机初始化来学习的position emb，这一点和transformer的position encoding的不一样，可以看下面的。

#### transformer的输入
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117220839.png)
**input = Input Embedding + Positional Encoding**
transformer的输入的处理是输入嵌入和位置编码，位置编码是使用正余弦去编码

### 如何做预训练: MLM+NSP
#### MLM-掩码语言模型

- AR—autoregressive，自回归模型; 只能考虑单侧的信息，典型的就是GPT
无监督目标函数
- AE—autoencoding，自编码模型﹔从损坏的输入数据中预测重建原始数据。可以使用上下文的信息，**Bert就是使用的AE**

解释——我爱吃饭
AR
**P(我爱吃饭)=P(我)P(爱|我)P(吃|我爱)P(饭|我爱吃);**
AE
**mask之后:【我爱mask饭】**
**P(我爱吃饭|我爱mask饭)=P(mask=吃|我爱饭)**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117232251.png)
在模型构建的时候，根据上下文来极力预测mask是吃。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117232625.png)

缺点
没有考虑mask之间的关系，没有考虑语序

mask操作
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117232708.png)
mask代码实现
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117232924.png)

### NSP

NSP样本如下:
1．从训练语料库中取出两个连续的段落作为正样本
2.从不同的文档中随机创建一对段落作为负样本

**缺点︰主题预测和连贯性预测合并为一个单项任务**

主题预测很容易，相对于连贯性预测比较难。
> 所以主题预测有时候被后面的模型抛弃了。

### 如何微调bert

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117233646.png)

## 3.如何提升BERT下游任务表现

正常情况下是不会自己去从头训练模型，都是基于现有模型的对特定的任务进行微调。
### 步骤
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250117234343.png)

#### 如何在相同领域数据中进行further pre-training
**1.动态mask:就是每次epoch去训练的时候mask，而不是一直使用同一个。**
- bert在训练使用的是固定mask，可以使用动态的mask进行改进
- 意思是bert原来是数据预处理和epoch迭代分开了，预处理完了就开始迭代所有epoch，mask在任意epoch中是一样的，up说的是每次epoch前做mask处理，任意epoch中的mask不一样
**2.n-gram mask:其实比如ERNIE和 SpanBert都是类似于做了实体词的mask**

### 参数
**Batch size**: 16,32一影响不太大
**Learning rate (Adam)**: 5e-5,3e-5,2e-5，尽可能小一点避免灾难性遗忘
**Number of epochs**: 3,4
Weighted decay修改后的adam，使用warmup，搭配线性衰减

以及可以采用**数据增强/自蒸馏/外部知识的融入**

