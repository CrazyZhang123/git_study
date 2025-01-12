---
created: 2024-11-10T12:30
updated: 2024-11-10T12:31
---
 

### 一、BatchNorm

#### 1\. 概念

对一个batch内的数据在通道尺度上计算均值和方差，将同批次同通道的[数据归一化](https://so.csdn.net/so/search?q=%E6%95%B0%E6%8D%AE%E5%BD%92%E4%B8%80%E5%8C%96&spm=1001.2101.3001.7020)为均值为0、方差为1的正态分布，最后用对归一化后的数据进行缩放和平移来还原数据本身的分布。

![](https://i-blog.csdnimg.cn/blog_migrate/99455d7ff0e7714a1d8db8b61f68d211.png)

上图展示了大小为\[3,4,2,2\]的tensor（批次大小为3，通道数为4，高为2，宽为2）的BatchNorm过程，**该过程是针对训练数据的且无缩放和平移**。可以看出，BatchNorm是对同一批次内同一通道的所有数据进行[归一化](https://so.csdn.net/so/search?q=%E5%BD%92%E4%B8%80%E5%8C%96&spm=1001.2101.3001.7020)。

在**训练**过程中，其计算过程如下：

![](https://i-blog.csdnimg.cn/blog_migrate/70e030c5d1d99154427dda96f72f6e28.png)

其中， μ B \\mu\_{\\mathcal{B}} μB​和 σ B 2 \\sigma^2\_{\\mathcal{B}} σB2​分别为当前批次下同一通道所有数据的均值和**有偏**方差， ϵ \\epsilon ϵ用来防止分母为0， γ \\gamma γ和 β \\beta β是可学习的参数（通道数为 C C C时，两个参数在当前特征层的总量为 2 × C 2\\times C 2×C），用来进行仿射变换，即通过缩放和平移使数据处于更好的分布上。

由于**测试**过程需要稳定的输出，所以并不是按照批次计算均值和方差，而是使用整个训练样本的均值和方差（通常由滑动平均法计算），如下：

![](https://i-blog.csdnimg.cn/blog_migrate/027a88db2f1cf976dcbf6576e488f31b.png)

其中， V a r \[ x \] Var\[x\] Var\[x\]指的是无偏方差，根据下式可以看出，将之前的有偏方差转为无偏方差乘上 m m − 1 \\frac{m}{m-1} m−1m​即可。

![](https://i-blog.csdnimg.cn/blog_migrate/b7f3d1bc9652a73df852380ca757b8c1.png)

所以，测试样本的BatchNorm过程如下：

![](https://i-blog.csdnimg.cn/blog_migrate/64c112dd394904c21e7c3bab3f5a5e45.png)

#### 2\. 作用

**优点：**  
(1) 加快模型训练时的收敛速度，避免梯度爆炸或者梯度消失。随着网络加深，非线性激活的输入分布发生偏移，落入饱和区，就会导致反向传播时出现梯度消失，这是训练收敛越来越慢的本质原因。BatchNorm通过归一化手段，将每层输入强行拉回均值0方差为1的[标准正态分布](https://so.csdn.net/so/search?q=%E6%A0%87%E5%87%86%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83&spm=1001.2101.3001.7020)，使得激活输入值分布在非线性函数梯度敏感区域，从而避免梯度消失问题，大大加快训练速度。**所以BatchNorm通常放在非线性激活函数之后，使输入到激活函数的值的分布更加稳定。**  
(2) 提高模型泛化能力，即防止过拟合。每个批次的均值、方差都有差异，对于整体而言相当于噪声，但各批次的均值、方差差异并不大，可以保证该噪声一定的随机性但又足够稳定（这也是在训练过程中不直接使用整个训练集均值和方差的原因），这种噪声的引入使模型更加鲁棒，避免对训练数据的过度拟合，从而提升了模型的泛化能力。  
**局限：**  
(1) BatchNorm更适合在大batch\_size中使用。  
因为BatchNorm非常依赖Batch的大小，当Batch值很小时，计算的均值和方差不稳定。  
(2) BatchNorm更适合在CV中使用，在NLP中效果不佳。  
因为CV中同批次的图像通常会被resize至相同大小，所以同批次特征的通道数相同且同通道的特征高宽一致；但在NLP中，一个批次包含多个句子，每个句子长度不尽相同，对同一位置的词向量进行归一化时每次可操作的词向量个数可能不同。例如我喜欢音乐、今天的早餐很丰盛，“我”和“今天的”、“喜欢”和“早餐”、“音乐”和“很”分别对应，“丰盛”却无对应的词向量。  
此外，在CV中同一通道特征来自同一卷积核，这意味着同一通道对应同一类特征，例如颜色、纹理、亮度等等，它们是可比较的；但在NLP中，同一位置的词向量通常无关联，例如我喜欢音乐、今天的早餐很丰盛，“我”和“今天的”、“喜欢”和“早餐”、“音乐”和“很”都没有关系且不可比较。

#### 3\. 实现

```python
import torch
import torch.nn as nn


class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # 滑动计算的均值和方差
        self._running_mean = torch.zeros(num_features)[None, :, None, None]
        self._running_var = torch.ones(num_features)[None, :, None, None]
        # 更新滑动均值和方差时的动量
        self._momentum = momentum
        # 防止分母计算为0
        self._eps = eps
        # 仿射变换参数，缩放和平移norm后的数据分布
        self._beta = torch.zeros(num_features)[None, :, None, None]
        self._gamma = torch.ones(num_features)[None, :, None, None]

    def forward(self, input):
        # input(N,C,H,W)
        if self.training:
            mean = torch.mean(input, dim=[0, 2, 3], keepdim=True)  # 计算均值
            var = input.var(dim=[0, 2, 3], unbiased=False, keepdim=True)  # 计算有偏方差
            m = input.numel() / input.size(1)  # 将var转为无偏估计时使用，一个批次一个通道的像素点数
            with torch.no_grad():  # 计算均值和方差的过程不需要梯度传输
                self._running_mean = self._momentum * mean + (1 - self._momentum) * self._running_mean
                self._running_var = self._momentum * (m / (m-1)) * var + (1 - self._momentum) * self._running_var
        else:
            # 测试阶段直接以running_mean和running_var为均值和方差
            mean = self._running_mean
            var = self._running_var

        input = (input - mean) / torch.sqrt(var + self._eps)  # 执行标准化

        return input * self._gamma + self._beta  # 仿射变换


if __name__ == "__main__":
    batch_size = 3
    channels = 4
    H = 2
    W = 2
    input = torch.randn(batch_size, channels, H, W)  # N*C*H*W

    myBN = MyBatchNorm(channels)
    MyO = myBN(input)
```

### 二、LayerNorm

#### 1\. 概念

LayerNorm不再在通道尺度上进行归一化，而是在词向量或样本尺度上进行归一化。将某一词向量或同一个样本所有词向量的值归一化至均值为0，方差为1。

![](https://i-blog.csdnimg.cn/blog_migrate/67d2ec351820a69195ad8da2693525ee.png)

上图展示了一个批次大小为N（N个句子）、句子长度为L、词向量长度为C的数据。一次词向量尺度上的LayerNorm是对图中所有词向量都进行归一化。一次样本尺度上的LayerNorm是对图中红色部分进行一次归一化，即有几个句子进行几次归一化。

LayerNorm不需要计算滑动平均，其计算公式如下：

![](https://i-blog.csdnimg.cn/blog_migrate/26c62e4e69ebbb754a5adf430619cf1a.png)

其中 E \[ x \] E\[x\] E\[x\]和 V a r \[ x \] Var\[x\] Var\[x\]是相应尺度上均值和**有偏**方差，仍然有可学习参数 γ \\gamma γ和 β \\beta β。

#### 2\. 作用

**优点：**  
(1) LayerNorm同样可以提升泛化能力和收敛速度。  
(2) 以样本或词向量为单位，对batch\_size无要求。  
**缺点：**  
(1) LayerNorm更适合在NLP中使用，它不改变词向量的方向，只改变词向量的模，但使不同样本的同通道特征失去了可比性。

#### 3\. 实现

```python
import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        # 防止分母计算为0
        self._eps = eps
        # 仿射变换参数，缩放和平移norm后的数据分布
        self._beta = 0
        self._gamma = 1

    def forward(self, input):
        # input(N,L,C)
        mean = torch.mean(input, dim=-1, keepdim=True)  # 计算均值
        var = input.var(dim=-1, unbiased=False, keepdim=True)  # 计算有偏方差
        input = (input - mean) / torch.sqrt(var + self._eps)  # 执行标准化

        return input * self._gamma + self._beta  # 仿射变换


if __name__ == '__main__':
    batch_size = 4
    length = 2
    hidden_dim = 3
    input = torch.rand(batch_size, length, hidden_dim)

    myLN = MyLayerNorm()
    MyO = myLN(input)
```

### 三、区别

1.  BatchNorm是对一个batch\_size样本内的每个通道的特征做归一化，LayerNorm是对每个样本的所有特征做归一化。
2.  BatchNorm抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系；LayerNorm抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系。
3.  BatchNorm训练与测试流程不同，训练使用实时参数，测试使用训练集总体参数，该参数在训练期间计算得到；LayerNorm不区分训练与测试，在样本内部完成归一化，不需要保留训练集的统计参数。
4.  BatchNorm适用batch\_size较大或非序列的场景；LayerNorm与之相反，此外LayerNorm保留一个样本内不同特征之间的大小关系或者说时序关系，这对NLP任务是至关重要的，所以RNN或Transformer用的是LayerNorm。

#### 致谢：

本博客仅做记录使用，无任何商业用途，参考内容如下：  
[【AI基础】图解手算BatchNorm、LayerNorm和GroupNorm](https://blog.csdn.net/qq_43426908/article/details/123119919?spm=1001.2014.3001.5506)  
[深入理解Pytorch的BatchNorm操作（含部分源码）](https://zhuanlan.zhihu.com/p/439116200)  
[BatchNorm和LayerNorm——通俗易懂的理解](https://blog.csdn.net/Little_White_9/article/details/123345062)  
[深入理解Batch Normalization原理与作用](https://blog.csdn.net/litt1e/article/details/105817224)  
[Layer Normalization解析](https://blog.csdn.net/qq_37541097/article/details/117653177?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164570699716780366593622%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164570699716780366593622&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-117653177.nonecase&utm_term=Layernorm&spm=1018.2226.3001.4450)

 

文章知识点与官方知识档案匹配，可进一步学习相关知识

[Python入门技能树](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)[人工智能](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)[深度学习](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)462525 人正在系统学习中

本文转自 <https://blog.csdn.net/beginner1207/article/details/137428146>，如有侵权，请联系删除。