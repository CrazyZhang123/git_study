 

#### [PyTorch](https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020)实现MLP的两种方法，以及nn.Conv1d, kernel\_size=1和nn.Linear的区别

*   [MLP（Multi-layer perceptron，多层感知机）实现](#MLPMultilayer_perceptron_2)
*   *   [MLP结构](#MLP_7)
    *   [方法1:nn.Linear](#1nnLinear_36)
    *   [方法2:nn.Conv1d & kernel\_size=1](#2nnConv1d__kernel_size1_97)
*   [nn.Conv1d, kernel\_size=1与nn.Linear不同](#nnConv1d_kernel_size1nnLinear_166)

MLP（Multi-layer perceptron，多层感知机）实现
-----------------------------------

最近在看`PointNet`论文，其主要思想为利用`MLP`结构学习[点云](https://so.csdn.net/so/search?q=%E7%82%B9%E4%BA%91&spm=1001.2101.3001.7020)特征，并进行全局池化（构造一个对称函数，symmetric function），实现无序点集输入时特征提取的不变性。

转到代码实现时，原以为用`nn.Linear`（`PyTorch`）这个方法创建网络结构（因为结构上`CNN`中的全连层`FC Layer`就是一个`MLP`，而其实现用的就是`nn.Linear`），但实际上用的是`nn.Conv1d`（注意`kernel_size=1`）实现的，一下就有些疑问了，`nn.Conv1d`也能实现`MLP`结构？

> **答案是`肯定`的，但`输入数据形式存在不同`**

### MLP结构

`MLP`应该是最简单的神经网络结构，下图（由[NN-SVG](http://alexlenail.me/NN-SVG/index.html)生成）所示为一个`输入层4节点`、`隐含层8节点`、`输出层3节点`的`MLP`：  
![MLP结构 ](https://i-blog.csdnimg.cn/blog_migrate/68d1d8b8fbb90922574bfd21e789f27a.png#pic_center)  
每一层的每个节点与前一层的所有节点进行连接（也即CNN中全连接的来由），节点的个数表示该层的特征维度，通过设置网络层数和节点个数，学习到输入数据的不同维度特征信息。

具体到数据处理形式上，MLP计算如下：  
X = \[ x 1 , x 2 , . . . , x m \] T X = \[x\_{1}, x\_{2}, ..., x\_{m}\]^{T} X\=\[x1​,x2​,...,xm​\]T  
Y = \[ y 1 , y 2 , . . . , y n \] T Y=\[y\_{1}, y\_{2}, ..., y\_{n}\]^{T} Y\=\[y1​,y2​,...,yn​\]T  
h j = ∑ i = 1 m w i j x i h\_{j}=\\sum\\limits\_{i=1}^{m}w\_{ij}x\_{i} hj​\=i\=1∑m​wij​xi​  
y j = g ( h j ) = g ( ∑ i = 1 m w i j x i ) y\_{j}=g(h\_j)=g(\\sum\\limits\_{i=1}^{m}w\_{ij}x\_{i}) yj​\=g(hj​)\=g(i\=1∑m​wij​xi​)  
其中：

*   X X X：输入层向量， m m m个维度/节点， Y Y Y：输出层向量， n n n个维度/节点，注意：**此处输入层输出层指的是相邻两层`前一层为输入层`，`后一层为输出层`，与MLP的输入层和输出层概念不同**
*   w w w：权重系数， w i j w\_{ij} wij​：输入层第 i i i个节点至输出层第 j j j个节点的权重
*   h j h\_{j} hj​：输出层第 j j j个节点的所有输入层节点加权之和
*   g ( ) g() g()：激活函数
*   i = 1 , 2 , . . . , m i=1, 2, ..., m i\=1,2,...,m， j = 1 , 2 , . . . , n j=1, 2, ..., n j\=1,2,...,n

> 需要注意的是，上述表示的是以`向量`（**Tensor维度为1**）作为输入的计算过程，对于由多个向量构成的`多维矩阵`（**Tensor维度大于等于2**），计算过程类似，保持向量的组合尺寸，只对向量的不同特征维度进行加权计算

例如，对于一个长度为100的点云（`100×3，tensor`）进行MLP处理，经过一个`3输入-10输出`的`Layer`计算后，输出结果仍为一个二维tensor（`100×10，tensor`）；同样，对于一个batch size为4，长度为100的点云数据包（`4×100×3，tensor`），经过同样的`Layer`计算，输出为一个三维tensor（`4×100×10，tensor`），如下图所示  
![计算数据流](https://i-blog.csdnimg.cn/blog_migrate/91e00ae62b1bf123d96a178b92a99543.png#pic_center)

### 方法1:nn.Linear

PyTorch官方文档中[nn.Linear](https://pytorch.org/docs/stable/nn.html?highlight=nn%20linear#torch.nn.Linear)的描述如下：  
![nn.Linear介绍](https://i-blog.csdnimg.cn/blog_migrate/12f11008b8ac89da85ba31621b1d0503.png#pic_center)  
对输入数据 x x x进行一个线性变化，与上文中 h h h的计算方式一致，具体含义：

*   _in\_features_：每个输入样本的大小，对应MLP中当前层的输入节点数/特征维度
*   _out\_features_：每个输出样本的大小，对应MLP中当前层的输出节点数/特征维度
*   输入数据形式：形状为\[N, \*, _in\_features_\]的tensor，N为batch size，这个参数是PyTorch各个数据操作中都具备的，相似的，输出数据形式为\[N, \*, _out\_features_\]

> 需要注意的是输入输出数据形式中的`*`参数，其表示为任意维度，对于单个向量，`*`为空

_代码A：利用`nn.Linear`对单个点云数据（向量）进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 3)		# 创建batch_size=1的单个点云
layer = nn.Linear(3, 10)	# 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		# 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([1, 3])
>>>torch.Size([1, 10])
'''
```

_代码B：利用`nn.Linear`对点云集进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 100, 3)		# 创建一个batch_size=1的点云，长度100
layer = nn.Linear(3, 10)	    # 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		    # 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([1, 100, 3])
>>>torch.Size([1, 100, 10])
'''
```

_代码C：利用`nn.Linear`对多批次点云集进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 100, 3)		# 创建一个batch_size=4的点云，长度100
layer = nn.Linear(3, 10)	    # 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		    # 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([4, 100, 3])
>>>torch.Size([4, 100, 10])
'''
```

> 通过上述代码可以看出，**`nn.Linear`作用在输入数据的最后一个维度上**，这一点不同于以下的`nn.Conv1d`

### 方法2:nn.Conv1d & kernel\_size=1

Pytorch官方文档中[nn.Conv1d](https://pytorch.org/docs/stable/nn.html?highlight=nn%20conv1d#torch.nn.Conv1d)的描述如下：  
![nn.Conv1d介绍](https://i-blog.csdnimg.cn/blog_migrate/dca9f71e43a13db7db36c9bb50f4b132.png#pic_center)  
关键参数：

*   _in\_channels_：输入通道，MLP中决定Layer输入的节点
*   _out\_channels_：输出通道，MLP中决定Layer输出的节点
*   _kernel\_size_：卷积核的宽度，应用在MLP中必须为1
*   _stride_：每次卷积移动的步长，应用在MLP中必须为1
*   _padding_：序列两端补0的个数，应用在MLP中必须为0

与图像的二维卷积（可参考该[博客中gif介绍](https://blog.csdn.net/l1076604169/article/details/92747124)）类似，一维卷积表示对序列数据进行卷积，如下图所示：  
![一维卷积示意](https://i-blog.csdnimg.cn/blog_migrate/00a9663e78e2b96908be8d4a5e774e23.png#pic_center)  
每个卷积核沿着数据长度方向对核内的数据进行卷积（根据卷积核权重累加），每移动一个步长获取一个值，所有的值构成输出的一个通道/特征维度；每个卷积核计算获得一个通道/特征维度

由nn.Conv1d的输出长度计算方式和上图示意可知：

> 当`kernel_size=1`，`stride=1`，`padding=0`时，每个卷积核计算后输出数据和输入数据的长度相同，并且一一对应，即 h o j = ∑ s = 1 i c k s x j s h\_{oj}=\\sum\\limits\_{s=1}^{ic}k\_{s}x\_{js} hoj​\=s\=1∑ic​ks​xjs​， o j oj oj为第 o o o个卷积核第 j j j个输出值， i c ic ic为输入数据的通道/特征维度， j s js js为输入数据第 j j j个中通道 s s s的位置，**这与MLP的节点计算方式一样**，因此可以用`nn.Conv1d`进行MLP计算

_代码D：利用`nn.Conv1d`对单个点云数据（向量）进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 3, 1)		            # 创建batch_size=1的单个点云
layer = nn.Conv1d(3, 10, kernel_size=1)	    # 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		                # 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([1, 3, 1])
>>>torch.Size([1, 10, 1])
'''
```

_代码E：利用`nn.Conv1d`对点云集进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 3, 100)		            # 创建一个batch_size=1的点云，长度100
layer = nn.Conv1d(3, 10, kernel_size=1)	    # 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		                 # 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([1, 3, 100])
>>>torch.Size([1, 10, 100])
'''
```

_代码F：利用`nn.Conv1d`对多批次点云集进行Layer计算_

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 3, 100)		            # 创建一个batch_size=4的点云，长度100
layer = nn.Conv1d(3, 10, kernel_size=1)	    # 构造一个输入节点为3，输出节点为10的网络层
y = F.sigmoid(layer(x))		                 # 计算y，sigmoid激活函数

print(x.size())
print(y.size())
'''
>>>torch.Size([4, 3, 100])
>>>torch.Size([4, 10, 100])
'''
```

> 通过上述代码可以看出，`nn.Conv1d`的输入数据格式只能一个三维tensor`[batch, channel, length]`，与`nn.Linear`输入数据格式不同；并且，`nn.Conv1d`的数据作用位置也不同，`nn.Conv1d`作用在第二个维度`channel`上

nn.Conv1d, kernel\_size=1与nn.Linear不同
-------------------------------------

从上述方法1和方法2可以看出，两者可以实现同样结构的MLP计算，但计算形式不同，具体为：

> *   `nn.Conv1d`输入的是一个`[batch, channel, length]`，3维tensor，而`nn.Linear`输入的是一个`[batch, *, in_features]`，可变形状tensor，在进行等价计算时务必保证`nn.Linear`输入tensor为三维
> *   `nn.Conv1d`作用在第二个维度位置`channel`，`nn.Linear`作用在第三个维度位置`in_features`，对于一个 X X X，若要在两者之间进行等价计算，需要进行`tensor.permute`，重新排列维度轴秩序

代码G：验证`nn.Conv1d, kernel_size=1`与`nn.Linear`计算结果相同，代码[**来自stack overflow**](https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-interpretation)

```python
import torch

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])

conv = torch.nn.Conv1d(8,32,1)
print(count_parameters(conv))
# 288

linear = torch.nn.Linear(8,32)
print(count_parameters(linear))
# 288

print(conv.weight.shape)
# torch.Size([32, 8, 1])
print(linear.weight.shape)
# torch.Size([32, 8])

# use same initialization
linear.weight = torch.nn.Parameter(conv.weight.squeeze(2))
linear.bias = torch.nn.Parameter(conv.bias)

tensor = torch.randn(128,256,8)
permuted_tensor = tensor.permute(0,2,1).clone().contiguous()	# 注意此处进行了维度重新排列

out_linear = linear(tensor)
print(out_linear.mean())
# tensor(0.0067, grad_fn=<MeanBackward0>)

out_conv = conv(permuted_tensor)
print(out_conv.mean())
# tensor(0.0067, grad_fn=<MeanBackward0>)
```

本文转自 <https://blog.csdn.net/l1076604169/article/details/107170146>，如有侵权，请联系删除。