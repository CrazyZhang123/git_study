---
created: 2024-12-16T22:39
updated: 2024-12-16T22:39
---
 

[PyTorch](https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020)中的torch.nn.Parameter() 详解
---------------------------------------------------------------------------------------------------

今天来聊一下PyTorch中的[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).nn.Parameter()这个函数，笔者第一次见的时候也是大概能理解函数的用途，但是具体实现原理细节也是云里雾里，在参考了几篇博文，做过几个实验之后算是清晰了，本文在记录的同时希望给后来人一个参考，欢迎留言讨论。

### 分析

先看其名，parameter，中文意为参数。我们知道，使用PyTorch训练神经网络时，本质上就是训练一个函数，这个函数输入一个数据（如CV中输入一张图像），输出一个预测（如输出这张图像中的物体是属于什么类别）。而在我们给定这个函数的结构（如卷积、全连接等）之后，能学习的就是这个函数的参数了，我们设计一个[损失函数](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&spm=1001.2101.3001.7020)，配合梯度下降法，使得我们学习到的函数（神经网络）能够尽量准确地完成预测任务。

通常，我们的参数都是一些常见的结构（卷积、全连接等）里面的计算参数。而当我们的网络有一些其他的设计时，会需要一些额外的参数同样很着整个网络的训练进行学习更新，最后得到最优的值，经典的例子有注意力机制中的权重参数、Vision Transformer中的class token和positional embedding等。

而这里的torch.nn.Parameter()就可以很好地适应这种应用场景。

下面是[这篇博客](https://www.jianshu.com/p/d8b77cc02410)的一个总结，笔者认为讲的比较明白，在这里引用一下：

> 首先可以把这个函数理解为类型[转换函数](https://edu.csdn.net/cloud/houjie?utm_source=highword&spm=1001.2101.3001.7020)，将一个不可训练的类型`Tensor`转换成可以训练的类型`parameter`并将这个`parameter`绑定到这个`module`里面(`net.parameter()`中就有这个绑定的`parameter`，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个`self.v`变成了[模型](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&spm=1001.2101.3001.7020)的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

### ViT中nn.Parameter()的实验

看过这个分析后，我们再看一下Vision Transformer中的用法：

```python
...

self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
...
```

我们知道在ViT中，positonal embedding和class token是两个需要随着网络训练学习的参数，但是它们又不属于FC、MLP、MSA等运算的参数，在这时，就可以用nn.Parameter()来将这个随机初始化的Tensor注册为可学习的参数Parameter。

为了确定这两个参数确实是被添加到了net.Parameters()内，笔者稍微改动源码，显式地指定这两个参数的初始数值为0.98，并打印迭代器net.Parameters()。

```python
...

self.pos_embedding = nn.Parameter(torch.ones(1, num_patches+1, dim) * 0.98)
self.cls_token = nn.Parameter(torch.ones(1, 1, dim) * 0.98)
...
```

实例化一个ViT模型并打印net.Parameters()：

```python
net_vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

for para in net_vit.parameters():
        print(para.data)
```

输出结果中可以看到，最前两行就是我们显式指定为0.98的两个参数pos\_embedding和cls\_token：

```
tensor([[[0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800],
         [0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800],
         [0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800],
         ...,
         [0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800],
         [0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800],
         [0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800]]])
tensor([[[0.9800, 0.9800, 0.9800,  ..., 0.9800, 0.9800, 0.9800]]])
tensor([[-0.0026, -0.0064,  0.0111,  ...,  0.0091, -0.0041, -0.0060],
        [ 0.0003,  0.0115,  0.0059,  ..., -0.0052, -0.0056,  0.0010],
        [ 0.0079,  0.0016, -0.0094,  ...,  0.0174,  0.0065,  0.0001],
        ...,
        [-0.0110, -0.0137,  0.0102,  ...,  0.0145, -0.0105, -0.0167],
        [-0.0116, -0.0147,  0.0030,  ...,  0.0087,  0.0022,  0.0108],
        [-0.0079,  0.0033, -0.0087,  ..., -0.0174,  0.0103,  0.0021]])
...
...
```

这就可以确定nn.Parameter()添加的参数确实是被添加到了Parameters列表中，会被送入优化器中随训练一起学习更新。

```python
from torch.optim import Adam
opt = Adam(net_vit.parameters(), learning_rate=0.001)
```

### 其他解释

以下是国外StackOverflow的一个大佬的解读，笔者自行翻译并放在这里供大家参考，想查看原文的同学请戳[这里](https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter)。

我们知道Tensor相当于是一个高维度的矩阵，它是Variable类的子类。Variable和Parameter之间的差异体现在与Module关联时。当Parameter作为model的属性与module相关联时，它会被自动添加到Parameters列表中，并且可以使用net.Parameters()迭代器进行访问。  
最初在Torch中，一个Variable（例如可以是某个中间state）也会在赋值时被添加为模型的Parameter。在某些实例中，需要缓存变量，而不是将它们添加到Parameters列表中。  
文档中提到的一种情况是RNN，在这种情况下，您需要保存最后一个hidden state，这样就不必一次又一次地传递它。需要缓存一个Variable，而不是让它自动注册为模型的Parameter，这就是为什么我们有一个显式的方法将参数注册到我们的模型，即nn.Parameter类。

举个例子：

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        self.linear2 = nn.Linear(hid,out_dim)
        self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred

in_d = 5
hidn = 2
out_d = 3
net = NN_Network(in_d, hidn, out_d)
```

然后检查一下这个模型的Parameters列表：

```python
for param in net.parameters():
    print(type(param.data), param.size())

""" Output
<class 'torch.FloatTensor'> torch.Size([5, 2])
<class 'torch.FloatTensor'> torch.Size([2])
<class 'torch.FloatTensor'> torch.Size([5, 2])
<class 'torch.FloatTensor'> torch.Size([2])
"""
```

可以轻易地送入到优化器中：

```python
opt = Adam(net.parameters(), learning_rate=0.001)
```

另外，请注意Parameter的require\_grad会自动设定。

各位读者有疑惑或异议的地方，欢迎留言讨论。

参考：

https://www.jianshu.com/p/d8b77cc02410

https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter

本文转自 <https://blog.csdn.net/weixin_44966641/article/details/118730730>，如有侵权，请联系删除。