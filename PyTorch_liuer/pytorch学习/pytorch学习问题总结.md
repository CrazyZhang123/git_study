---
created: 2024-09-30T12:12
updated: 2025-01-23T20:55
---
### 1、torch.nn.Linear()讲解

^a35883
nn.Linear 是 PyTorch 框架中的一个模块，用于实现线性层，也就是全连接层。线性层是神经网络中的基本构件，它执行一个基于矩阵乘法的线性变换，通常用于将输入数据转换为输出数据。

参数介绍：

**in_features**：输入特征的数量。
**out_features**：输出特征的数量。
**bias**：一个布尔值，指示是否使用偏置项（默认为True）。
```python
import torch
import torch.nn as nn
 
# 定义输入特征的尺寸
input_height, input_width = 4, 4
# 定义输入通道数
input_channels = 3
# 定义输出节点数
output_nodes = 5
 
# 创建一个随机的输入特征图，维度为[2，3，4，4]
input_data = torch.randn(2, input_channels, input_height, input_width)
 
# 创建一个全连接层，4 -> 5
linear_layer = nn.Linear(input_data.size(-1), output_nodes)
 
# 应用全连接层
output = linear_layer(input_data)
 
# 输出的尺寸将是 [2，3，4，5]
print("Output shape:", output.shape)
```

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241123155904.png)
可以看到[2,3,4,4]维度的数据经过`nn.Linear得到了`[2,3,4,5]的数据，确实可以计算多维度。
首先通过公式可以看到nn.Linear是通过一个权重矩阵来实现维度的变化的。x是输入，A是权重矩阵，x与经过转置的权重矩阵A进行矩阵乘法，最后加上偏置项。
其次nn.Linear的输入是不限制维度的，可以看到括号中的*，其中 * 表示任意数量的附加维度，包括为空（即常见的数据拉平后只剩一个维度）。
权重矩阵维度为（out,in）,但是nn.Linear函数的用法是nn.Linear(in,out)。
最终输出的结果是（*,out）。

我画了个计算维度变换图，如下：

![image.png|504](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241123160008.png)



假设输入的数据维度为[32，3，4]，通过nn.Linear(4,2)得到[32，3，2]。这里取消偏置项。
由于在手册中权重矩阵的维度是（out,in），那么而经过转置之后就是（in,out)也就是图中的（4，2）。|
最终得到（1，2）形状的输出，准确的来说，是将（4，）形状变为（2，）。


torch.nn.Linear(1,1) 也就是给了输入1，输出1

[Linear()官网讲解](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930122340.png)

#### 权重矩阵A和偏置b
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930122457.png)

- A的维度是(out_features,input_features)
- A和b都服从均匀分布$u(-\sqrt{ k},\sqrt{ k }),k= \frac{1}{in_features}$
- 例如k= 1，那么u(-1,1)也就是A和b的元素都是从-1到1之间随便取一个值。


```python
#    每个输入样本的大小 20，输出样本大小 30
m = torch.nn.Linear(20, 30)
# 正态分布 维度128 * 20
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930123031.png)
输出维度是128 * 30

y = xA^T + b
    x: 128 * 20
     A^T:   20 * 30
     b: 30
因此 y维度是 128 * 30



### 2、Tensor常见方法

.item ()  
把 tensor 转为 python 的 float 类型

.numpy ()  
将张量转换为与其共享底层存储的 n 维 numpy 数组

对Tensor维度的理解：
```python
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

print(x_data,y_data)
print(x_data.size(),y_data.size())

# 结果
tensor([[1.],
        [2.],
        [3.]]) torch.Size([3, 1])
tensor([[0.],
        [0.],
        [1.]]) torch.Size([3, 1])

```

### 3、torch.max,full,ones,zeros
#### 一、一个参数时的 [torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).max()
返回Tensor的最大值

#### 二、增加指定维度时的 torch.max()

###### 1. 函数介绍

```python
torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)
```

返回张量 input 在压缩指定维度 dim 时的最大值及其下标。

###### 2. 实例

```python
import torch

# 返回张量在压缩指定维度时的最大值及其下标
b = torch.randn(4, 4)
print(b)
print(torch.max(b, 0))  # 指定0维，压缩0维，0维消失，也就是行消失，返回列最大值及其下标
print(torch.max(b, 1))  # 指定1维，压缩1维，1维消失，也就是列消失，返回行最大值及其下标
```

输出结果：

```python
tensor([[-0.8862,  0.3502,  0.0223,  0.6035],
        [-2.0135, -0.1346,  2.0575,  1.4203],
        [ 1.0107,  0.9302, -0.1321,  0.0704],
        [-1.4540, -0.4780,  0.7016,  0.3029]])
torch.return_types.max(
values=tensor([1.0107, 0.9302, 2.0575, 1.4203]),
indices=tensor([2, 2, 1, 1]))
torch.return_types.max(
values=tensor([0.6035, 2.0575, 1.0107, 0.7016]),
indices=tensor([3, 2, 0, 2]))
```
#### 三、两个输入张量时的 torch.max()

###### 1. 函数介绍

```python
torch.max(input, other_input, out=None) → Tensor
```

返回两张量 input 和 other_input 在对应位置上的最大值形成的新张量。
#### 一. [torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).full()函数解析

##### 1. 官网链接

**[torch.full()](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=full#torch.full)**，如下图所示： ![torch.full()](https://i-blog.csdnimg.cn/blog_migrate/f36712507c5bc58a8c31c67cfb834b9c.png)

##### 2. torch.full()函数解析

```
 torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

**返回创建size大小的维度，里面元素全部填充为fill_value**

##### 3.代码举例

**输出形状为(2,3)的二维张量，里面元素全部填充为5**

```
 x = torch.full(size=(2,3),fill_value=5)  
 x
```

 输出结果如下：  
```
 tensor([[5, 5, 5],  
         [5, 5, 5]])
```

```
input_ids = torch.full(
    (batch_size*num_beams, 1),  
    bos_token_id,
    dtype=torch.long,
    device=next(self.parameters()).device,
)
```
解释：
```ad-note
- 使用 torch.full 函数创建一个形状为 (batch_size * num_beams, 1) 的张量。
- 张量的每个元素都初始化为 bos_token_id，即目标序列的开始标记。
- 数据类型设置为 torch.long，表示张量的元素类型为长整型。
- 设备设置为模型参数所在的设备，确保张量与模型在同一设备上（CPU 或 GPU）。
**size**：输出张量的形状，这里为 (batch_size * num_beams, 1)。
**fill_value**：填充值，这里为 bos_token_id。
**dtype**：数据类型，这里为 torch.long。
**device**：设备，这里为 next(self.parameters()).device，表示模型参数所在的设备。
next(self.parameters()).device：
self.parameters() 返回模型的所有参数的迭代器。
next(self.parameters()) 获取模型的第一个参数。
.device 获取该参数所在的设备（CPU 或 GPU）。

```

#### 二. torch.ones()函数解析

##### 1. 官网链接

**[torch.ones()](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones)**，如下图所示： ![torch.ones()](https://i-blog.csdnimg.cn/blog_migrate/d4142e8c8bdc5bf4ac46c7986ee49b98.png)

##### 2. torch.ones()函数解析

```
 torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

**返回创建size大小的维度，里面元素全部填充为1**

##### 3. 代码举例

```
 a = torch.ones(2, 3)  
 b = torch.ones(5)  
 c = torch.ones(size=(3,4))  
 a,b,c

 输出结果如下：  
 (tensor([[1., 1., 1.],  
          [1., 1., 1.]]),  
  tensor([1., 1., 1., 1., 1.]),  
  tensor([[1., 1., 1., 1.],  
          [1., 1., 1., 1.],  
          [1., 1., 1., 1.]]))
```

#### 三. torch.zeros()函数解析

##### 1.官网链接

**[torch.zeros()](https://pytorch.org/docs/stable/generated/torch.zeros.html?highlight=zero)**，如下图所示： ![torch.zeros()](https://i-blog.csdnimg.cn/blog_migrate/220c12ff2166ab296edb471710bbb931.png)

##### 2. torch.zeros()函数解析

```
 torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

**返回创建size大小的维度，里面元素全部填充为0**

##### 3. 代码举例

```
 a = torch.zeros(2, 3)  
 b = torch.zeros(5)  
 c = torch.zeros(size=(3,4))  
 a,b,c

 输出结果如下：  
 (tensor([[0., 0., 0.],  
          [0., 0., 0.]]),  
  tensor([0., 0., 0., 0., 0.]),  
  tensor([[0., 0., 0., 0.],  
          [0., 0., 0., 0.],  
          [0., 0., 0., 0.]]))
```

本文转自 <[https://blog.csdn.net/flyingluohaipeng/article/details/125103847](https://blog.csdn.net/flyingluohaipeng/article/details/125103847)，如有侵权，请联系删除。
## 4、transpose函数
transpose 是 PyTorch 中用于交换张量维度的函数，特别是用于二维张量（矩阵）的转置操作，常用于线性代数运算、深度学习模型的输入和输出处理等

### 基本知识如下

**功能**：交换张量的两个维度
**输入**：一个张量和==两个要交换的维度的索引==
**输出**：具有新维度顺序的张量
**原理分析如下**：
transpose 的核心原理是通过交换指定维度的方式改变张量的形状
例如，对于一个二维张量 (m, n)，调用 transpose(0, 1) 会返回一个形状为 (n, m) 的新张量，其元素顺序经过了调整

**高维张量**： 对于高维张量，transpose 只会影响指定的两个维度，而其他维度保持不变
**内存视图**：与 permute 类似，transpose 返回的是原始张量的一个视图，不会进行数据复制

[[详细分析Pytorch中的transpose基本知识（附Demo）_ 对比 permute]]

## 5、F.softmax
#### 1.函数语法格式和作用

==F.softmax作用==:
按照行或者列来做归一化的
F.softmax函数语言格式:

``` python
# 0是对列做归一化，1是对行做归一化
F.softmax(x,dim=1) 或者 F.softmax(x,dim=0)
```

==F.log_softmax作用==:
在softmax的结果上再做多一次log运算
F.log_softmax函数语言格式:

F.log_softmax(x,dim=1) 或者 F.log_softmax(x,dim=0)

#### 2.参数解释
- x指的是输入矩阵。
- dim指的是归一化的方式，==如果为0是对列做归一化，1是对行做归一化==。

#### 3.具体代码
```python
import torch
import torch.nn.functional as F
logits = torch.rand(2,2)
pred = F.softmax(logits, dim=1)
pred1 = F.log_softmax(logits, dim=1)
print(logits)
print(pred)
print(pred1)

```
结果

————————————————
https://blog.csdn.net/m0_51004308/article/details/118001835

# 6、Drop层
Dropout 层是深度学习中一种常用的正则化技术，主要用于缓解神经网络过拟合问题，以下是关于它的详细介绍：

### 基本原理
- 在神经网络训练过程中，Dropout 层会按照一定的概率（这个概率通常是人为设定的，比如 0.5 等）随机地 “丢弃”（也就是让其输出为 0）一些神经元，使得网络在每次训练迭代时结构都有所不同。形象地说，就好比训练一个团队，每次训练时都随机让一部分成员 “休息”，整个团队的构成处于动态变化中。
- 例如，对于一个全连接神经网络的某一层有 100 个神经元，设定 Dropout 概率为 0.3，那么在每次前向传播训练时，大约会有 30 个神经元被随机置为 0，它们不参与此次的计算和后续的梯度更新等操作。

### 作用机制
- **减少神经元之间的复杂共适应关系**：如果没有 Dropout，神经元之间可能会逐渐形成非常固定、复杂的相互依赖关系来拟合训练数据，这容易导致过拟合。而 Dropout 通过随机丢弃神经元，使得每个神经元不能过度依赖其他特定的神经元，迫使它们学习更具鲁棒性、更普遍的特征，因为任何一个神经元都有可能在某次训练中被 “抛弃”，所以它们需要独立地对各种输入情况做出合理反应。
- **增加模型的泛化能力**：由于训练时网络结构不断变化，模型相当于在训练多个不同结构的 “子网络”，到了测试阶段（此时 Dropout 层通常是关闭的，即所有神经元都参与计算），模型可以综合这些不同 “子网络” 学习到的特征来对新的数据进行更好的预测，从而使得模型对未见过的数据（也就是测试数据或者实际应用中的新数据）有更好的泛化表现，避免只对训练数据拟合得很好，但对新数据效果很差的过拟合情况。

### 应用场景及示例
- **图像分类任务**：在卷积神经网络（CNN）用于图像分类时，比如经典的 VGG、ResNet 等网络结构中，可以在全连接层中间添加 Dropout 层。例如在一个简单的手写数字识别的 CNN 模型中，在最后几个全连接层分别添加 Dropout 层，概率设置为 0.5 左右，能有效提升模型在测试集上的准确率，让模型不至于在训练集上准确率很高但在实际的手写数字图像测试时准确率大幅下降。
- **自然语言处理任务**：在循环神经网络（RNN）或者基于 Transformer 的模型（如 BERT 等用于文本分类、机器翻译等任务时），在词嵌入层之后或者中间隐藏层添加 Dropout 层也较为常见。例如在一个情感分析的双向 LSTM 模型中，在 LSTM 层之后添加 Dropout 层，可避免模型对训练文本中的特定词汇组合过度拟合，从而使模型在面对新的评论文本时能更准确地判断其情感倾向。

### 使用方式及参数选择
- 在常见的深度学习框架（如 PyTorch、TensorFlow 等）中使用起来都比较方便。以 PyTorch 为例，代码如下：

```python
import torch
import torch.nn as nn

# 定义一个包含Dropout层的简单神经网络层序列
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(p=0.5)  # 这里设置Dropout概率为0.5
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

- **参数选择**：Dropout 的概率参数 `p` 的选择通常需要根据具体的任务、模型结构等来进行调试。一般来说，取值范围常在 0.2 到 0.5 之间比较常见，但也不是绝对的。如果概率设置得过低，可能起不到足够的正则化效果，无法很好地防止过拟合；而如果概率设置得过高，比如接近 1，那模型可能因为丢失太多信息而难以学习到有效的特征，导致训练效果不佳，欠拟合等问题。

# 7、size函数
## 一、torch.size与torch.shape的区别

torch.size()是一个方法，而torch.shape是tensor的一个属性，例如：

```python
a = torch.arange(24).view(4,-1,2,3)

a.shape   ## 返回torch.Size([4,1,2,3])

a.size()    ## 返回torch.Size([4,1,2,3])
```

```python
import torch
zl=torch.ones(3,20,28,28)
b,c,h,w=zl.size()
print('张量的batch维度是{}'.format(b))
print('张量的通道维度是{}'.format(c))
```
- 可以发现size取出来的就是定义的函数，可以通过索引直接取
```
sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
```
## 二、numpy中size与shape的区别

np.size,np.shape均是数组类的属性，不同的是size**返回整个数组的数字个数，而shape返回数组维度大小**，例如：

```python
a = np.arange(24).reshape(4,-1,2,3)

a.shape  ## 返回 (4,1,2,3)

a.size      ## 返回24
```

# 8、torch.squeeze()和torch.unsqueeze()
## 一. torch.squeeze()函数解析

### 1. 官网链接

[torch.squeeze()](https://pytorch.org/docs/stable/generated/torch.squeeze.html?highlight=squeeze)，如下图所示： ![torch.squeeze()](https://gitee.com/zhang-junjie123/picture/raw/master/image/6fe0b584bdab10db7091c40a7afe8ed9.png)

### 2. torch.squeeze()函数解析

 torch.squeeze(input, dim=None, out=None) 

**squeeze()函数的功能是维度压缩。返回一个tensor（张量），其中 input 中维度大小为1的所有维都已删除。** 举个例子：如果 input 的形状为 (A×1×B×C×1×D)，那么返回的tensor的形状则为 (A×B×C×D) **当给定 dim 时，那么只在给定的维度（dimension）上进行压缩操作，注意给定的维度大小必须是1，否则不能进行压缩。** 举个例子：如果 input 的形状为 (A×1×B)，squeeze(input, dim=0)后，返回的tensor不变，因为第0维的大小为A，不是1；squeeze(input, 1)后，返回的tensor将被压缩为 (A×B)。

### 3. 代码举例

**3.1 输入size=(2, 1, 2, 1, 2)的张量**

```
 x = torch.randn(size=(2, 1, 2, 1, 2))  
 x.shape

 输出结果如下：  
 torch.Size([2, 1, 2, 1, 2])
```

**3.2 把x中维度大小为1的所有维都已删除**

```
 y = torch.squeeze(x)#表示把x中维度大小为1的所有维都已删除  
 y.shape
```

 输出结果如下：  
```
 torch.Size([2, 2, 2])
```

**3.3 把x中第一维删除，但是第一维大小为2，不为1，因此结果删除不掉**

```
 y = torch.squeeze(x,0)#表示把x中第一维删除，但是第一维大小为2，不为1，因此结果删除不掉  
 y.shape
```

 输出结果如下：  
```
 torch.Size([2, 1, 2, 1, 2])
```

**3.4 把x中第二维删除，因为第二维大小是1，因此可以删掉**

```
 y = torch.squeeze(x,1)#表示把x中第二维删除，因为第二维大小是1，因此可以删掉  
 y.shape
```

 输出结果如下：  
```
 torch.Size([2, 2, 1, 2])
```

**3.5 把x中最后一维删除，但是最后一维大小为2，不为1，因此结果删除不掉**

```
 y = torch.squeeze(x,dim=-1)#表示把x中最后一维删除，但是最后一维大小为2，不为1，因此结果删除不掉  
 y.shape
```

 输出结果如下：  
```
 torch.Size([2, 1, 2, 1, 2])
```

## 二.torch.[unsqueeze](https://so.csdn.net/so/search?q=unsqueeze&spm=1001.2101.3001.7020)()函数解析

### 1. 官网链接

[torch.unsqueeze()](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)，如下图所示： ![torch.unsqueeze()](https://gitee.com/zhang-junjie123/picture/raw/master/image/64a4c4d77cffb3c6228baf556459f5ef.png)

### 2. torch.unsqueeze()函数解析

 torch.unsqueeze(input, dim) → Tensor

**unsqueeze()函数起升维的作用,参数dim表示在哪个地方加一个维度，注意dim范围在:[-input.dim() - 1, input.dim() + 1]之间，比如输入input是一维，则dim=0时数据为行方向扩，dim=1时为列方向扩，再大错误。**

#### 3. 代码举例

**3.1 输入一维张量，在第0维(行)扩展，第0维大小为1**

```
 x = torch.tensor([1, 2, 3, 4])  
 y = torch.unsqueeze(x, 0)#在第0维扩展，第0维大小为1  
 y,y.shape

 输出结果如下：  
 (tensor([[1, 2, 3, 4]]), torch.Size([1, 4]))
```

**3.2 在第1维(列)扩展，第1维大小为1**

```
 y = torch.unsqueeze(x, 1)#在第1维扩展，第1维大小为1  
 y,y.shape
```

 输出结果如下：  
```
 (tensor([[1],  
          [2],  
          [3],  
          [4]]),  
  torch.Size([4, 1]))
```

**3.3 在第最后一维（也就是倒数第一维进行）扩展，最后一维大小为1**

```
 y = torch.unsqueeze(x, -1)#在第最后一维扩展，最后一维大小为1  
 y,y.shape

 输出结果如下：  
 (tensor([[1],  
          [2],  
          [3],  
          [4]]),  
  torch.Size([4, 1]))
```

# 9、forward函数

### （1）forward被调用的时机是  
```python
 encoder_layer = EncoderLayer()  
 output, attention_weights = encoder_layer(input_data, mask)
```
**类被初始化后的对象，就可以当作forward函数了**

# 10、torch.contiguous()
torch.contiguous()方法语义上是“连续的”，经常与torch.permute()、torch.transpose()、torch.view()方法一起使用，要理解这样使用的缘由，得从[pytorch](https://so.csdn.net/so/search?q=pytorch&spm=1001.2101.3001.7020)多维数组的低层存储开始说起：

touch.view()方法**对张量改变“形状”其实并没有改变张量在内存中真正的形状**，可以理解为：

1. view方法没有拷贝新的张量，没有开辟新内存，与原张量共享内存；
   
2. view方法只是**重新定义了访问张量的规则，使得取出的张量按照我们希望的形状展现**。
   

pytorch与numpy在存储MxN的数组时，均是按照行优先将数组拉伸至一维存储，比如对于一个二维张量

 // An highlighted blockt = torch.tensor([[2, 1, 3], [4, 5, 9]])

在内存中实际上是

> [2, 1, 3, 4, 5, 9]

按照行优先原则，数字在语义和在内存中都是连续的，**当我们使用torch.transpose()方法或者torch.permute()方法对张量翻转后，改变了张量的形状**

 // An highlighted blockt2 = t.transpose(0, 1)t2

> tensor([[2,4],
> 
> [1,5],
> 
> [3,9])

此时如果对t2使用view方法，会报错：

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/9549649aaec0a194a7732410beb96e9f.png)

	原因是：改变了形状的t2语义上是3行2列的，在内存中还是跟t一样，没有改变，导致如果按照语义的形状进行view拉伸，数字不连续，此时torch.contiguous()方法就派上用场了

 // An highlighted blockt3 = t2.contiguous()t3

> tensor([[2,4],
> 
> [1,5],
> 
> [3,9])

可以看到t3与t2一样，都是3行2列的张量，此时再对t3使用view方法：

 // An highlighted blockt3.view(-1)

> tensor([2, 4, 1, 5, 3, 9])

t3已经按照语义的形状展开了，t2与t共享内存是怎样的呢？

 // An highlighted blockt.view(-1)

> tensor([2, 1, 3, 4, 5, 9])

可以看出contiguous方法改变了**多维数组在内存中的存储顺序，以便配合view方法使用**

torch.contiguous()方法<mark style="background: FFFF00;">首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。</mark>

# 11、Embedding层
## 简单了解
nn.Embedding((num_embeddings,embedding_dim)

其中，num_embeddings代表词典大小尺寸，比如训练时所可能出现的词语一共5000个词，那么就有num_embedding=5000，而embedding_dim表示嵌入向量的维度，即用多少来表示一个符号。提到embedding_dim，就不得先从one_hot向量说起。

最初的时候，人们将word转换位vector是利用one_hot向量来实现的。简单来讲，现在词典里一共5个字，[‘我’,‘是’,‘中’,‘国’,‘人’]，即num_embedding=5，而现在有一句话‘我是人’，one_hot则利用一个长度为5的01向量来代表这句话中的每个字（假设词典按顺序编码），有

- 我：[1 0 0 0 0 ]
- 是：[0 1 0 0 0 ]
- 人：[0 0 0 0 1 ]
显然，这种方法简单快捷，但是当词典的字很多，比如50000个字的时候，这种方法会造成极大的稀疏性，不便于计算。而且one_hot方法无法处理原来标签的序列信息，比如“我是人”这句话中，“我”和“人”的距离与“我”和“是”的距离一样，这显然是不合理的。

因此，为了改进这些缺点，embedding算是它的一个升级版（没有说谁好和谁不好的意思，现在one hot向量也依旧在很多地方运用，选择特征时要选择自己合适的才行。）

embedding翻译word是这样操作的，首先，先准备一本词典，这个词典将原来句子中的每个字映射到更低的维度上去。比如，字典中有50000个字，那按照One-hot方法，我们要为每个字建立一个50000长度的vector,对于embedding来说，我们只需要指定一个embedding_dim，这个embedding_dim<50000即可。

![在这里插入图片描述|520](https://gitee.com/zhang-junjie123/picture/raw/master/image/665b89c8c3af9000ad96eaac8fa3b15b.jpeg) 见上图，也就是说，原来one-hot处理一句话（这句话有length个字），那我们需要一个（length，50000）的矩阵代表这句话，现在只需要（length，embedding_dim）的矩阵就可以代表这句话。 从数学的角度出发就是（length，50000）*（50000,embedding），做了个矩阵运算。

![在这里插入图片描述|520](https://gitee.com/zhang-junjie123/picture/raw/master/image/59cf24f39fd1fe7315333b6e0136354f.jpeg) 上面这张图是计算示意图，为了方便计算，我们将句子的最大长度设置为max_length,也就是说，输入模型的所有语句不可能超过这个长度。原来用one_hot向量表示的话，如果浓缩起来就是上面的那个长条，如果展开则是下方的那个矩阵。 **也就是说，当整个输入数据X只有一句话时** X（1, max_length, num_embeddings） 字典为（num_embeddings, embedding_dim） 则经过翻译之后，这句话变成（1，max_length，embedding_dim）

**当输入数据X有多句话时，即Batch_size不等于1**,有 X（batch_size, max_length, num_embeddings） 字典为（num_embeddings, embedding_dim） 则经过翻译之后，输入数据X变成（batch_size，max_length，embedding_dim）

![在这里插入图片描述|520](https://gitee.com/zhang-junjie123/picture/raw/master/image/52a19047db1d9b9d9b5ad3058fca831d.jpeg) 因此，nn.embedding（num_embeddings,embedding_dim）的作用就是将输入数据降维到embedding_dim的表示层上，得到了输入数据的另一种表现形式。

**代码应用如下**

```python
 import torch  
 import numpy as np  
 ​  
 batch_size=3  
 seq_length=4  
 ​  
 input_data=np.random.uniform(0,19,size=(batch_size,seq_length))#shape(3,4)  
 ​  
 input_data=torch.from_numpy(input_data).long()   
 ​  
 embedding_layer=torch.nn.Embedding(vocab_size,embedding_dim)   
 ​  
 lstm_input=embedding_layer(input_data)#shape(3,4,6)
```

注意，输入进embedding层的数据并不是经过词典映射的，而是原始数据，因此张量内部有超出embedding层合法范围的数，这会导致embedding层报错，所以一开始input_data要约束在0,19之间。 且**在输入数据的时候，不需要自己手动的将vocab_num给设置出来，这个embedding会自己完成映射**。

 [torch.nn.Embedding官方页面](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)

### 1. torch.nn.Embedding介绍

#### （1）[词嵌入](https://so.csdn.net/so/search?q=%E8%AF%8D%E5%B5%8C%E5%85%A5&spm=1001.2101.3001.7020)简介

  关于词嵌入，[这篇文章](https://blog.csdn.net/qq_41477675/article/details/114645012)讲的挺清楚的，相比于One-hot编码，Embedding方式更方便计算，例如在“就在江湖之上”整个词典中，要编码“江湖”两个字，One-hot编码需要大小的张量，其中${word\_count} $为词典中所有词的总数，而Embedding方式的嵌入维度${embedding\_dim} $ 可远远小于 ${word\_count} $。在运用Embedding方式编码的词典时，只需要词的索引，下图例子中： “江湖”——>[2, 3]

![在这里插入图片描述](https://gitee.com/zhang-junjie123/picture/raw/master/image/f1137cd85b74b625b73005c4684b7a60.png)

#### （2）重要参数介绍

  nn.embedding就相当于一个词典嵌入表：

 torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)

常用参数：   **① num_embeddings (int)：** 词典中词的总数

  **② embedding_dim (int)：** 词典中每个词的嵌入维度

  **③ padding_idx (int, optional)：** 填充索引，在padding_idx处的嵌入向量在训练过程中没有更新，即它是一个固定的“pad”。对于新构造的Embedding，在padding_idx处的嵌入向量将默认为全零，但可以更新为另一个值以用作填充向量。

输入：$ {Input(∗)} $: IntTensor 或者 LongTensor，为任意size的张量，包含要提取的所有词索引。 输出： ${Output(∗, H)} $: $ {∗} $ 为输入张量的size， ${H} $ = embedding_dim

### 2. torch.nn.Embedding用法

#### （1）基本用法

官方例子如下：

```
 import torch  
 import torch.nn as nn  
 ​  
 embedding = nn.Embedding(10, 3)  
 x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])  # (2,4)
 ​  
 y = embedding(x)  
 ​  
 print('权重:\n', embedding.weight)  
 print('输出:')  
 print(y) # 维度 (2,)
```

查看权重与输出，打印如下：

 权重:  
```
  Parameter containing:  
 tensor([[ 1.4212,  0.6127, -1.1126],  
         [ 0.4294, -1.0121, -1.8348],  
         [-0.0315, -1.2234, -0.4589],  
         [ 0.6131, -0.4381,  0.1253],  
         [-1.0621, -0.1466,  1.7412],  
         [ 1.0708, -0.7888, -0.0177],  
         [-0.5979,  0.6465,  0.6508],  
         [-0.5608, -0.3802, -0.4206],  
         [ 1.1516,  0.4091,  1.2477],  
         [-0.5753,  0.1394,  2.3447]], requires_grad=True)  
```
 输出:  
```
 tensor([[[ 0.4294, -1.0121, -1.8348],  
          [-0.0315, -1.2234, -0.4589],  
          [-1.0621, -0.1466,  1.7412],  
          [ 1.0708, -0.7888, -0.0177]],  
 ​  
         [[-1.0621, -0.1466,  1.7412],  
          [ 0.6131, -0.4381,  0.1253],  
          [-0.0315, -1.2234, -0.4589],  
          [-0.5753,  0.1394,  2.3447]]], grad_fn=<EmbeddingBackward0>)
```

  家人们，发现了什么，输入x 的size 大小为 [ 2 , 4 ] {[2, 4]} [2,4] ，输出 y 的 size 大小为 [ 2 , 4 , 3 ] {[2, 4, 3]} [2,4,3] ，下图清晰的展示出nn.Embedding干了个什么事儿： ![在这里插入图片描述](https://gitee.com/zhang-junjie123/picture/raw/master/image/9d83e25e00756d924b7bb392b1e7053f.png)

  nn.Embedding相当于是一本词典，本例中，词典中一共有10个词 X 0 {X_0} X0​~ X 9 {X_9} X9​，每个词的嵌入维度为3，输入 x {x} x 中记录词在词典中的索引，输出 y {y} y 为输入 x {x} x 经词典编码后的映射。

  **注意：此时存在一个问题，词索引是不能超出词典的最大容量的，即本例中，输入 x {x} x 中的数值取值范围为 [ 0 , 9 ] {[0, 9]} [0,9]。**

#### （2）自定义词典权重

  如上所示，在未定义时，nn.Embedding的自动初始化权重满足 N ( 0 , 1 ) {N(0,1)} N(0,1) 分布，此外，nn.Embedding的权重也可以通过from_pretrained来自定义：
```

 import torch  
 import torch.nn as nn  
 ​  
 weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])  
 embedding = nn.Embedding.from_pretrained(weight)  
 x = torch.LongTensor([1, 0, 0])  
 y = embedding(x)  
 print(y)
```

输出为：

```
 tensor([[4.0000, 5.1000, 6.3000],  
         [1.0000, 2.3000, 3.0000],  
         [1.0000, 2.3000, 3.0000]])
```

#### （3）padding_idx用法

  padding_idx可用于指定词典中哪一个索引的词填充为0。

```
 import torch  
 import torch.nn as nn  
 ​  
 embedding = nn.Embedding(10, 3, padding_idx=5)  
 x = torch.LongTensor([[5, 2, 0, 5]])  
 y = embedding(x)  
 print('权重:\n', embedding.weight)  
 print('输出:')  
 print(y)
```

输出为：

```
 权重:  
  Parameter containing:  
 tensor([[ 0.1831, -0.0200,  0.7023],  
         [ 0.2751, -0.1189, -0.3325],  
         [-0.5242, -0.2230, -1.1677],  
         [-0.4078, -1.2141,  1.3185],  
         [ 0.8973, -0.9650,  0.5420],  
         [ 0.0000,  0.0000,  0.0000],  
         [ 0.0597,  0.6810, -0.2595],  
         [ 0.6543, -0.6242,  0.2337],  
         [-0.0780, -0.9607, -0.0618],  
         [ 0.2801, -0.6041, -1.4143]], requires_grad=True)  
 输出:  
 tensor([[[ 0.0000,  0.0000,  0.0000],  
          [-0.5242, -0.2230, -1.1677],  
          [ 0.1831, -0.0200,  0.7023],  
          [ 0.0000,  0.0000,  0.0000]]], grad_fn=<EmbeddingBackward0>)

```
  词典中，被padding_idx[标定](https://so.csdn.net/so/search?q=%E6%A0%87%E5%AE%9A&spm=1001.2101.3001.7020)后的词嵌入向量可被重新定义：

```
 import torch  
 import torch.nn as nn  
 ​  
 padding_idx=2  
 embedding = nn.Embedding(3, 3, padding_idx=padding_idx)  
 print('权重:\n', embedding.weight)  
 ​  
 with torch.no_grad():  
     embedding.weight[padding_idx] = torch.tensor([1.1, 2.2, 3.3])  
 print('权重:\n', embedding.weight)
```

输出为：

```
 权重:  
  Parameter containing:  
 tensor([[ 0.7247,  0.7553, -1.8226],  
         [-1.3304, -0.5025,  0.5237],  
         [ 0.0000,  0.0000,  0.0000]], requires_grad=True)  
 权重:  
  Parameter containing:  
 tensor([[ 0.7247,  0.7553, -1.8226],  
         [-1.3304, -0.5025,  0.5237],  
         [ 1.1000,  2.2000,  3.3000]], requires_grad=True)
```

# 12、Layer norm
说明
LayerNorm中不会像BatchNorm那样跟踪统计全局的均值方差，因此train()和eval()对LayerNorm没有影响。

### LayerNorm参数
```
torch.nn.LayerNorm(
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-05,
        elementwise_affine: bool = True)
```
### normalized_shape
如果传入整数，比如4，则被看做只有一个整数的list，此时LayerNorm会对输入的最后一维进行归一化，这个int值需要和输入的最后一维一样大。

- 假设此时输入的数据维度是[3, 4]，则对3个长度为4的向量求均值方差，得到3个均值和3个方差，分别对这3行进行归一化（每一行的4个数字都是均值为0，方差为1）；LayerNorm中的weight和bias也分别包含4个数字，重复使用3次，对每一行进行仿射变换（仿射变换即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
- 如果输入的是个list或者torch.Size，比如[3, 4]或torch.Size([3, 4])，则会对网络最后的两维进行归一化，且要求输入数据的最后两维尺寸也是[3, 4]。

- 假设此时输入的数据维度也是[3, 4]，首先对这12个数字求均值和方差，然后归一化这个12个数字；weight和bias也分别包含12个数字，分别对12个归一化后的数字进行仿射变换（仿射变换即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
- 假设此时输入的数据维度是[N, 3, 4]，则对着N个[3,4]做和上述一样的操作，只是此时做仿射变换时，weight和bias被重复用了N次。
假设此时输入的数据维度是[N, T, 3, 4]，也是一样的，维度可以更多。
注意：显然LayerNorm中weight和bias的shape就是传入的normalized_shape。

### eps
归一化时加在分母上防止除零。

### elementwise_affine
如果设为False，则LayerNorm层不含有任何可学习参数。

如果设为True（默认是True）则会包含可学习参数weight和bias，用于仿射变换，即对输入数据归一化到均值0方差1后，乘以weight，即bias。



### LayerNorm前向传播（以normalized_shape为一个int举例）
1. 如下所示输入数据的shape是(3, 4)，此时normalized_shape传入4（输入维度最后一维的size），则沿着最后一维（沿着最后一维的意思就是对最后一维的数据进行操作）求E(x)和Var(x)，并用这两个结果把batch沿着最后一维归一化，使其均值为0，方差为1。归一化公式用到了eps($\epsilon$)，即 $y=\frac{x-E(x)}{\sqrt{ Var(x)+\epsilon }}$。
```
tensor = torch.FloatTensor([[1, 2, 4, 1],
                            [6, 3, 2, 4],
                            [2, 4, 6, 1]])
```
此时E[x]=[2.0,3.75,3.25]$Var[y]biased$=[1.5000,2.1875,3.6875]，（有偏样本方差），归一化后的值如下，举例说明：第0行第2列的数字4，减去第0行的均值2.0等于2，然后除以$\sqrt{ 1.5+\epsilon }$即2/1.224749≈1.6330。
```
[[-0.8165,  0.0000,  1.6330, -0.8165],
 [ 1.5213, -0.5071, -1.1832,  0.1690],
 [-0.6509,  0.3906,  1.4321, -1.1717]]
```
2. 如果**elementwise_affine**\==True，则对归一化后的batch进行仿射变换，即乘以模块内部的weight（初值是[1., 1., 1., 1.]）然后加上模块内部的bias（初值是[0., 0., 0., 0.]），这两个变量会在反向传播时得到更新。
3. 如果**elementwise_affine**\==False，则LayerNorm中不含有**weight和bias**两个变量，只做归一化，不会进行仿射变换。
### 总结
在使用LayerNorm时，通常只需要指定normalized_shape就可以了。

# 13、分布
## 1、xavier分布
其目的是使得每层网络的输入和输出的方差保持一致，从而有效地避免梯度消失或爆炸问题。

## 2、kaiming 分布
[[pytorch学习：xavier分布和kaiming分布]]

## 3、multinomial 多项分布

[[Pytorch中的多项分布multinomial.Multinomial().sample()解析]]

# 14、LogitsProcessorList和StoppingCriteriaList

## 1、LogitsProcessorList
**功能**：LogitsProcessorList 是一个**用于处理模型输出的 logits 的列表。logits 是模型在输出层的原始预测值，通常是一个未归一化的概率分布。**
**用途**：在生成任务中，logits 处理器可以用于对模型的输出进行各种操作，例如**温度采样、top-k 采样、top-p 采样**等，以提高生成文本的质量。
### logits
logits 是机器学习和深度学习中常用的一个术语，特别是在分类任务中。logits 是**模型在输出层的原始预测值**，通常是一个未归一化的概率分布。具体来说，logits 具有以下**特点**：
- **未归一化**：logits 是模型输出的原始值，没有经过归一化处理（如 softmax 函数）。因此，它们可能不是概率值，而是一些任意的实数值。
- **输入给激活函数**：在分类任务中，logits 通常作为激活函数（如 softmax 或 sigmoid）的输入，以转换为概率分布。
- **用于损失计算**：在训练过程中，logits 通常直接用于计算损失函数（如交叉熵损失），因为这样可以更稳定地进行梯度计算。
#### 示例
假设有一个二分类任务，模型的输出层有两个神经元，分别对应两个类别。模型的输出可能是 [2.0, 1.0]，这就是 logits。如果我们将这些 logits 传递给 softmax 函数，可以得到归一化的概率分布：
```
import numpy as np

logits = [2.0, 1.0]
probabilities = np.exp(logits) / np.sum(np.exp(logits))
print(probabilities)

```
输出可能是 \[0.73105858, 0.26894142]，这表示第一个类别的概率约为 73.1%，第二个类别的概率约为 26.9%。
## 2、StoppingCriteriaList
**功能**：StoppingCriteriaList 是一个用于**定义生成过程停止条件的列表。生成过程通常是一个迭代过程，直到满足某个停止条件才会终止**。
**用途**：在生成任务中，停止条件可以包括生成的序列长度达到最大限制、生成的序列中出现了特定的结束标记等。这些停止条件可以帮助控制生成过程，防止生成过长或不合适的序列。

# 15、张量乘法

**引 言**  torch中的[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)张量之间乘法操作分为向量乘法和[矩阵乘法](https://edu.csdn.net/course/detail/40020?utm_source=glcblog&spm=1001.2101.3001.7020)。向量乘法分为内积运算和外积运算，矩阵乘法又分为元素级乘法(Hadamard积)和传统矩阵乘法（第一矩阵列数等于第二矩阵行数），向量和[矩阵乘法运算](https://so.csdn.net/so/search?q=%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E8%BF%90%E7%AE%97&spm=1001.2101.3001.7020)对于初学者而言很容易混淆和错误使用。结合本人在实践操作中的经验，将pytorch中常用torch.dot()、torch.outer()、torch.mul()、torch.mm()和torch.matmul()等[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)的用法进行详细介绍和举例说明。

**目录**

[一、 向量运算](#t0)

[1.1 内积运算](#t1)

[1.2 外积运算](#t2)

[二、torch.mul()矩阵元素级乘法函数](#t3)

[三、torch.mm()二维矩阵乘法函数](#t4)

[四、torch.matmul()矩阵乘法函数](#t5)

[4.1 1维向量×1维向量的内积运算](#t6)

[4.2 1维向量×2维矩阵或2维矩阵×1维向量](#t7)

[4.3 2维矩阵×2维矩阵](#t8)

[4.4 三维矩阵相乘](#t9)

[五、torch.mv()矩阵向量乘法函数](#t10)

[六、@运算符的矩阵乘法](#t11)

[七、总结与注意](#t12)

---

## 一、 向量运算

### 1.1 内积运算

内积运算(inner product)是两个向量各元素相乘相加，结果是一个标量scalar。对于n维向量![\vec{a} = (a_{1},a_{2},...,a_{n}) \in R^{^{n}}](https://latex.csdn.net/eq?\vec{a} %3D (a_{1}%2Ca_{2}%2C...%2Ca_{n}) \in R^{^{n}})，![\vec{b} = (b_{1},b_{2},...,b_{n})\in R^{^{n}}](https://latex.csdn.net/eq?\vec{b} %3D (b_{1}%2Cb_{2}%2C...%2Cb_{n})\in R^{^{n}})，![\vec{a}\cdot \vec{b} = (a_{1}b_{1}+a_{2}b_{2}+...+a_{n}b_{n})](https://latex.csdn.net/eq?\vec{a}\cdot \vec{b} %3D (a_{1}b_{1}&plus;a_{2}b_{2}&plus;...&plus;a_{n}b_{n})) 。在阅读文献时内积表达式为 ![a^{T}b](https://latex.csdn.net/eq?a%5E%7BT%7Db), where ![a^{T}\in R^{1\ast n}](https://latex.csdn.net/eq?a%5E%7BT%7D%5Cin%20R%5E%7B1%5Cast%20n%7D), ![b\in R^{n\ast 1}](https://latex.csdn.net/eq?b%5Cin%20R%5E%7Bn%5Cast%201%7D) ，（向量不特殊说明一般指代列向量）。

在torch中使用torch.dot()函数或者torch.matmul()函数(在matmul部分会对**向量内积运算**进行详细介绍)，示例代码。

```
 x = torch.tensor([2,3])
 y = torch.tensor([2,2]) 
 out = torch.dot(x,y)    
 # out = tensor(10)
```

对于numpy [array](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020) 调用函数 numpy.inner(x,y)计算两个向量的内积。

### 1.2 外积运算

两个向量的外积运算(outer product)是向量中的每一个元素与另外一个向量中的每一个元素相乘，结果不是一个标量，而是一个矩阵。对于m维向量![\vec{a} = (a_{1},a_{2},...,a_{m}) \in R^{^{m}}](https://latex.csdn.net/eq?\vec{a} %3D (a_{1}%2Ca_{2}%2C...%2Ca_{m}) \in R^{^{m}})，n维向量![\vec{b} = (b_{1},b_{2},...,b_{n})\in R^{^{n}}](https://latex.csdn.net/eq?\vec{b} %3D (b_{1}%2Cb_{2}%2C...%2Cb_{n})\in R^{^{n}}),![\vec{a}\odot\vec{b} \in R^{m\ast n}](https://latex.csdn.net/eq?%5Cvec%7Ba%7D%5Codot%5Cvec%7Bb%7D%20%5Cin%20R%5E%7Bm%5Cast%20n%7D)。在阅读外文文献时向量外积表达式为 ![ab^{T}](https://latex.csdn.net/eq?ab%5E%7BT%7D), where ![a\in R^{ m\ast 1}](https://latex.csdn.net/eq?a%5Cin%20R%5E%7B%20m%5Cast%201%7D), ![b^{T}\in R^{1\ast n}](https://latex.csdn.net/eq?b%5E%7BT%7D%5Cin%20R%5E%7B1%5Cast%20n%7D) 。

![\vec{a}\odot\vec{b} = \begin{bmatrix} a_{1}b_{1}& a_{1} b_{2}&\cdots &a_{1} b_{n}& \\ a_{2}b_{1}& a_{2} b_{2}&\cdots &a_{2} b_{n}& \\ \vdots &\vdots & \ddots &\vdots \\ a_{m}b_{1}& a_{m} b_{2}&\cdots &a_{m} b_{n}& \end{bmatrix}](https://latex.csdn.net/eq?\vec{a}\odot\vec{b} %3D \begin{bmatrix} a_{1}b_{1}%26 a_{1} b_{2}%26\cdots %26a_{1} b_{n}%26 \\ a_{2}b_{1}%26 a_{2} b_{2}%26\cdots %26a_{2} b_{n}%26 \\ \vdots %26\vdots %26 \ddots %26\vdots \\ a_{m}b_{1}%26 a_{m} b_{2}%26\cdots %26a_{m} b_{n}%26 \end{bmatrix})

**两个向量的外积运算其核心本质是Hadamard积运算**，需先对向量a和![b^{T}](https://latex.csdn.net/eq?b%5E%7BT%7D)采用广播机制变成m*n维度的矩阵，然后每个对应位置元素相乘。两个向量外积运算可以直接使用torch.outer()函数，也可以使用torch.mul()函数（在torch.mul()函数部分会展示Hadamard积的示例代码），但是要注意对向量进行增加维度、转置和广播，保证参与torch.mul()运算的两个向量变形后的矩阵维度相同。

示例代码

```
1.直接使用torch.outer函数
x = torch.tensor([2,3,4,5])
y = torch.tensor([2,2,2]) 
out = torch.outer(x,y)print(out) 
# 2.对向量进行增加维度和转置操作，随后使用torch.mul函数计算Hadamard积
x = torch.tensor([2,3,4,5])
y = torch.tensor([2,2,2]) 
x = x.view(-1,1)    
#(4,1)y = y[None,:]   
#transpose  (1,3)
print(torch.mul(x,y))
```

对于numpy array 调用np.outer(x,y)函数，计算两个向量的外积。（**注：两个向量的元素数量可以不同**）

**如果两个矩阵进行外积运算其核心算法是克罗内克积（Kronecker积）。![A\epsilon R^{m\times n}](https://latex.csdn.net/eq?A%5Cepsilon%20R%5E%7Bm%5Ctimes%20n%7D)，![B\epsilon R^{p\times q}](https://latex.csdn.net/eq?B%5Cepsilon%20R%5E%7Bp%5Ctimes%20q%7D),其Kronecker积为![A\otimes B\in R^{mp\ast nq}](https://latex.csdn.net/eq?A%5Cotimes%20B%5Cin%20R%5E%7Bmp%5Cast%20nq%7D)。**

示例：

矩阵![A = \begin{bmatrix} a_{1} & a_{2}\\ a_{3}& a_{4} \end{bmatrix}](https://latex.csdn.net/eq?A %3D \begin{bmatrix} a_{1} %26 a_{2}\\ a_{3}%26 a_{4} \end{bmatrix})，矩阵![B = \begin{bmatrix} b_{1} & b_{2}& b_{3}\\ b_{4}& b_{5}& b_{6} \end{bmatrix}](https://latex.csdn.net/eq?B %3D \begin{bmatrix} b_{1} %26 b_{2}%26 b_{3}\\ b_{4}%26 b_{5}%26 b_{6} \end{bmatrix}),

![A\odot B = \begin{bmatrix} a_{1}B & a_{2}B\\ a_{3}B&a_{4}B \end{bmatrix} = \begin{bmatrix} a_{1}b_{1}& a_{1}b_{2} & a_{1}b_{3}& \vdots & a_{2}b_{1} &a_{2}b_{2}& a_{2}b_{3} \\ a_{1}b_{4}& a_{1}b_{5} & a_{1}b_{6}& \vdots&a_{2}b_{4} &a_{2}b_{5}& a_{2}b_{6} \\ \cdots &\cdots &\cdots&\cdots&\cdots&\cdots&\cdots\\a_{3}b_{1}&a_{3}b_{2} & a_{3}b_{3}& \vdots&a_{4}b_{1} &a_{4}b_{2}& a_{4}b_{3} \\ a_{3}b_{4} & a_{3}b_{5} & a_{3}b_{6}& \vdots&a_{4}b_{4} & a_{4}b_{5} &a_{4}b_{6}\end{bmatrix}](https://latex.csdn.net/eq?A\odot B %3D \begin{bmatrix} a_{1}B %26 a_{2}B\\ a_{3}B%26a_{4}B \end{bmatrix} %3D \begin{bmatrix} a_{1}b_{1}%26 a_{1}b_{2} %26 a_{1}b_{3}%26 \vdots %26 a_{2}b_{1} %26a_{2}b_{2}%26 a_{2}b_{3} \\ a_{1}b_{4}%26 a_{1}b_{5} %26 a_{1}b_{6}%26 \vdots%26a_{2}b_{4} %26a_{2}b_{5}%26 a_{2}b_{6} \\ \cdots %26\cdots %26\cdots%26\cdots%26\cdots%26\cdots%26\cdots\\a_{3}b_{1}%26a_{3}b_{2} %26 a_{3}b_{3}%26 \vdots%26a_{4}b_{1} %26a_{4}b_{2}%26 a_{4}b_{3} \\ a_{3}b_{4} %26 a_{3}b_{5} %26 a_{3}b_{6}%26 \vdots%26a_{4}b_{4} %26 a_{4}b_{5} %26a_{4}b_{6}\end{bmatrix}) 

可以直接调用**torch.kron(x,y)**函数，也可以自行编写函数。

**注意：在中文文献中exterior product 也翻译成外积，但指的是空间解析几何中的向量积，结果是一个向量。通常也称为矢量积或者叉积(cross product)。在英文文献中严格区分区分cross product和outer product，所以阅读文献时要特别注意。**

## 二、torch.mul()矩阵元素级乘法函数

torch.mul()函数主要对矩阵中的元素实施Hadamard积运算，该运算属于元素级乘法操作。可以直接使用**“ * ”**替换torch.mul()函数。在矩阵运算中，要求两个矩阵的维度相同，矩阵![A\epsilon R^{m\times n}](https://latex.csdn.net/eq?A%5Cepsilon%20R%5E%7Bm%5Ctimes%20n%7D),![B\epsilon R^{m\times n}](https://latex.csdn.net/eq?B%5Cepsilon%20R%5E%7Bm%5Ctimes%20n%7D)，矩阵A和B的Hadamard积![A\odot B\in R^{m*n}](https://latex.csdn.net/eq?A%5Codot%20B%5Cin%20R%5E%7Bm*n%7D)。

矩阵元素级乘法也可以用于【向量×矩阵】的情况，此时要求**向量的长度与矩阵最后一个维度相同**，采用广播机制将向量变成与矩阵相同的形状，随后进行逐元素相乘操作。

```
 x = torch.tensor([[1,1],[3,3],[4,4]])

y = torch.tensor([2,2])
out1 = torch.mul(x,y)    #等价于out1 = x*y

  #结果tensor([[2, 2],        [6, 6],        [8, 8]])
```



## 三、torch.mm()二维矩阵乘法函数

torch.mm()只适合于二维矩阵乘法运算，如果矩阵维度超过两个维度则会报错。**二维矩阵乘法运算要求第一个矩阵的列数与第二个矩阵的行数相同**。

 import torch A = torch.randint(1,5,size=(2,3))B = torch.randint(1,5,(3,2))print('A: \n',A)print('B: \n',B)result = torch.mm(A,B)print('result: \n {}'.format(result)) ##结果##A:  tensor([[2, 3, 2],        [1, 4, 4]])B:  tensor([[2, 2],        [4, 4],        [2, 3]])result:  tensor([[20, 22],        [26, 30]])

## 四、torch.matmul()矩阵乘法函数

torch.matmul()属于广义矩阵乘法函数操作，适用形式有：1维向量×1维向量，1维向量×2维矩阵，2维矩阵×1维向量，任意维度矩阵相乘等。每种情况的具体使用会结合示例代码逐一介绍。

### 4.1 1维向量×1维向量的内积运算

torch.matmul()函数作用于两个1维向量运算时，两个向量长度相同，主要对两个1维向量进行内积运算（**结果为标量scalar**）。功能与**torch.dot()**函数相同(**torch.dot()函数只适用于1维向量运算**)。

```
 x = torch.tensor([2,3,4])

y = torch.tensor([2,2,2])

out1 = torch.matmul(x,y)  

#out1 : tensor(18)out2 = torch.dot(x,y)     #out2 : tensor(18)
```

### 4.2 1维向量×2维矩阵或2维矩阵×1维向量

向量与矩阵做矩阵乘法运算时，需对向量进行增维操作，将其变成2维矩阵，矩阵相乘结束后，结果中增加的维度需要被删除。


1）向量$a\in R^{m}$与矩阵$B\in R^{m*n}$相乘，需先将向量变成矩阵$A\in R^{1*m}$，矩阵乘法维度变化：(1×m)×(m×n)->(1×n)，乘法运算结果矩阵$R^{1*n}$需删除新增维度，删除后的结果变成长度为n的1维向量$R^{n}$。

```
 x = torch.tensor([2,3])

y = torch.tensor([[1,1,1],[2,2,2]])

out = torch.matmul(x,y) 

 #out:tensor([8, 8, 8])print(out.shape)         #torch.Size([3])
```

2）矩阵![B\in R^{m*n}](https://latex.csdn.net/eq?B%5Cin%20R%5E%7Bm*n%7D)与向量![a\in R^{n}](https://latex.csdn.net/eq?a%5Cin%20R%5E%7Bn%7D)相乘，则将向量a增维成矩阵![A\in R^{n*1}](https://latex.csdn.net/eq?A%5Cin%20R%5E%7Bn*1%7D)，矩阵乘法维度变化：(m×n)×(n×1)->(m×1),运算结果![R^{m*1}](https://latex.csdn.net/eq?R%5E%7Bm*1%7D)需删除新增维度，[降维](https://ml-summit.org/cloud-member?uid=c1041&spm=1001.2101.3001.7020)成1维向量![R^{m}](https://latex.csdn.net/eq?R%5E%7Bm%7D)。

```
x = torch.tensor([[3,3,3],[4,4,4]])

y = torch.tensor([2,2,2])

out = torch.matmul(x,y) 

 #out:tensor([18, 24])print(out.shape)         #out.shape:torch.Size([2])
```



### 4.3 2维矩阵×2维矩阵

两个矩阵相乘时，torch.matmul()函数等价于torch.mm()函数：(m,n)×(n,t)->(m,t)

```
x = torch.tensor([[1,1],[3,3],[4,4]])

y = torch.tensor([[2,2,2],[5,5,5]])

 out1 = torch.matmul(x,y)

print(f"out1: {out1}")

 out2 = torch.mm(x,y)print(f"out2: {out2}")

 ##结果##out1: tensor([[ 7,  7,  7],        [21, 21, 21],        [28, 28, 28]])

out2: tensor([[ 7,  7,  7],        [21, 21, 21],        [28, 28, 28]])
```



### 4.4 三维矩阵相乘

对于高于二维的矩阵，第一个矩阵最后一个维度必须和第二个矩阵的倒数第二维度相同。如果是两个三维矩阵相乘，也可以使用torch.bmm()。

```
x = torch.randn(3,4,5)

y = torch.randn(3,5,2)

result = torch.matmul(x,y)

print(result.shape)   

 #shape: torch.Size([3, 4, 2]) 
```



## 五、torch.mv()矩阵向量乘法函数

torch.mv()用于执行2维矩阵×1维向量操作，**矩阵的最后一个维度与向量长度必须相同**。内部运算机理是先对向量末尾进行增维操作变成矩阵，执行矩阵乘法操作后，删除结果的最后一个维度。也可以采用上文提到的torch.matmul()。

```
x = torch.randint(1,4,(3,5))

y = torch.randint(1,4,(5,))

print(f"x: {x}")

print(f"y: {y}")

result = torch.mv(x,y)

print("result: {}".format(result))

print(result.shape) 

##结果
##x: tensor([[2, 1, 1, 1, 2],        [1, 1, 1, 2, 3],        [1, 2, 1, 3, 2]])

y: tensor([1, 1, 3, 2, 2])

result: tensor([12, 15, 16])torch.Size([3])
```



## 六、@运算符的矩阵乘法

- 若mat1和mat2都是两个一维向量，那么对应操作就是torch.dot()
  
- 若mat1是二维矩阵，mat2是一维向量，那么对应操作就是torch.mv()
  
- 若mat1和mat2都是两个二维矩阵，那么对应操作就是torch.mm()
  

## 七、总结与注意

1.向量的运算分为内积和外积运算，内积运算结果为标量，外积运算结果为矩阵(**Hadamard积**)，如果是矩阵的外积运算其实质就是**克罗内克积（Kronecker积）。**在使用外积运算时，注意区分**cross product和outer product。**

2.torch.mul()属于元素级操作，参与运算的矩阵要求形状相同，如果是向量与矩阵相乘，要求向量的长度与矩阵最后一个维度相同。torch.mm()只能执行二维矩阵运算，torch.matmul()适用于多维度矩阵乘法运算。

本文转自 [https://blog.csdn.net/li1784506/article/details/129756233?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-129756233-blog-105783610.235^v43^control&spm=1001.2101.3001.4242.2&utm_relevant_index=4](https://blog.csdn.net/li1784506/article/details/129756233?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-129756233-blog-105783610.235%5Ev43%5Econtrol&spm=1001.2101.3001.4242.2&utm_relevant_index=4)，如有侵权，请联系删除。

# 16、深入PyTorch：PyTorch张量和NumPy[数组](https://edu.csdn.net/course/detail/40020?utm_source=glcblog&spm=1001.2101.3001.7020)相互转换

#### 文章目录

*   [深入PyTorch：PyTorch张量和NumPy数组相互转换](#PyTorchPyTorchNumPy_2)
*   *   [一、torch.from\_numpy()](#torchfrom_numpy_7)
    *   *   [用法](#_12)
        *   [注意事项](#_25)
    *   [二、numpy()](#numpy_31)
    *   *   [用法](#_36)
        *   [注意事项](#_49)
    *   [三、示例代码与性能分析](#_56)
    *   *   [示例代码：使用torch.from\_numpy()和numpy()进行转换](#torchfrom_numpynumpy_61)
*   [结束语](#_87)



在PyTorch和NumPy的交互中，`torch.from_numpy()`和`numpy()`是两个重要的函数，它们允许我们在PyTorch张量和[NumPy数组](https://so.csdn.net/so/search?q=NumPy%E6%95%B0%E7%BB%84&spm=1001.2101.3001.7020)之间进行转换。了解这两种方法及其工作原理对于充分利用这两个库是非常重要的。本文将深入探讨这两个函数的使用和它们背后的机制，并通过示例代码展示其应用。

### 一、torch.from\_numpy()

`torch.from_numpy()`是PyTorch提供的一个便捷函数，用于将NumPy数组转换为PyTorch张量。该函数在内部使用了NumPy的C接口，所以它保留了NumPy数组的形状和数据类型。

#### 用法

```python
import numpy as np
import torch

# 创建一个NumPy数组
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])

# 使用torch.from_numpy()转换为PyTorch张量
torch_tensor = torch.from_numpy(numpy_array)
```

#### 注意事项

1.  **数据类型**：`torch.from_numpy()`会保留NumPy数组的数据类型。如果NumPy数组是浮点数类型，转换后的张量也将是浮点数类型。
2.  **可变性**：通过`torch.from_numpy()`创建的张量默认是不可变的，这意味着你不能直接修改其内容。如果你需要修改张量，可以通过`.clone()`方法创建一个副本。
3.  **内存共享**：`torch.from_numpy()`创建的张量和原始NumPy数组共享相同的内存。这意味着对张量的修改将影响原始数组，反之亦然。如果你不希望共享内存，可以使用`.clone()`方法。

### 二、numpy()

与`torch.from_numpy()`相反，`numpy()`函数用于将PyTorch张量转换为NumPy数组。这个函数在内部使用了PyTorch的C++ API，确保了转换的准确性和效率。

#### 用法

```python
import numpy as np
import torch

# 创建一个PyTorch张量
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 使用numpy()转换为NumPy数组
numpy_array = torch_tensor.numpy()
```

#### 注意事项

1.  **数据类型**：`numpy()`函数将张量转换为与原始张量相同的数据类型。如果张量是浮点数类型，转换后的数组也将是浮点数类型。
2.  **可变性**：通过`numpy()`创建的数组默认是不可变的，这意味着你不能直接修改其内容。如果你需要修改数组，可以通过`.clone()`方法创建一个副本。
3.  **内存共享**：与`torch.from_numpy()`类似，`numpy()`创建的数组和原始张量共享相同的内存。这意味着对数组的修改将影响原始张量，反之亦然。如果你不希望共享内存，可以使用`.clone()`方法。
4.  **类型转换**：需要注意的是，在某些情况下，将PyTorch张量转换为NumPy数组可能会触发类型转换。例如，如果你的PyTorch张量包含整数类型的数据，而你希望将其转换为浮点数类型的NumPy数组，那么转换可能会发生数据类型的自动转换。

### 三、示例代码与[性能分析](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)

下面是一个简单的示例代码，展示了如何使用`torch.from_numpy()`和`numpy()`进行转换，并比较了它们的[性能](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)。

#### 示例代码：使用torch.from\_numpy()和numpy()进行转换

```python
import numpy as np
import torch
import time

# 创建一个大的NumPy数组和相应的PyTorch张量作为示例
numpy_array = np.random.rand(10000, 10000)  # 大数组，用于测试性能
torch_tensor = torch.tensor(numpy_array)  # 将NumPy数组转换为PyTorch张量

# 使用torch.from_numpy()进行转换
start_time = time.time()
torch_from_np = torch.from_numpy(numpy_array)  # 使用torch.from_numpy()进行转换
end_time = time.time() - start_time  # 计算时间差
print(f"torch.from_numpy() time: {end_time} seconds")

# 使用numpy()进行转换
start_time = time.time()
np_from_torch = torch_tensor.numpy()  # 使用numpy()进行转换
end_time = time.time() - start_time
print(f"numpy() time: {end_time} seconds")
```

结束语
---

*   亲爱的读者，感谢您花时间阅读我们的博客。我们非常重视您的反馈和意见，因此在这里鼓励您对我们的博客进行评论。
*   您的建议和看法对我们来说非常重要，这有助于我们更好地了解您的需求，并提供更高质量的内容和服务。
*   无论您是喜欢我们的博客还是对其有任何疑问或建议，我们都非常期待您的留言。让我们一起互动，共同进步！谢谢您的支持和参与！
*   我会坚持不懈地创作，并持续优化博文质量，为您提供更好的阅读体验。
*   谢谢您的阅读！

本文转自 <https://blog.csdn.net/qq_41813454/article/details/129838551>，如有侵权，请联系删除。

# 17、Tensordataset and DataLoader                 
#### TensorDataset 详解
 ` TensorDataset ` 主要用于将多个 ` Tensor ` 组合在一起，方便对数据进行统一处理。它可以用于简单地将特征和标签配对，也可以将多个特征张量组合在一起。
##### 1. 将特征和标签组合

假设我们有一组图像数据（特征）和对应的标签，我们可以将它们组合成一个 ` TensorDataset ` 。

```python
import torch
from torch.utils.data 
import TensorDataset 
# 创建输入数据（图像）和标签
images = torch.randn(100, 3, 28, 28)  # 100张图像，每张图像3通道，28x28像素
labels = torch.randint(0, 10, (100,))  # 100个标签，范围在0到9之间 
# 创建 TensorDataset
dataset = TensorDataset(images, labels) # 访问数据集中的特定样本
sample_image, sample_label = dataset[0]
print(f"Sample Image Shape: {sample_image.shape}")  
# 输出: Sample Image Shape: torch.Size([3, 28, 28])
print(f"Sample Label: {sample_label}") 
# 输出: Sample Label: 3
```

在这个例子中，我们创建了一个包含100张图像和对应标签的 ` TensorDataset ` 。通过 ` dataset[0] ` ，我们可以访问第一个样本的图像和标签。

##### 2. 组合多个特征张量

除了将特征和标签组合， ` TensorDataset ` 还可以将多个特征张量组合在一起。例如，假设我们有两个[不同的](https://so.csdn.net/so/search?q=%E4%B8%8D%E5%90%8C%E7%9A%84&amp;spm=1001.2101.3001.7020 )特征张量，我们可以将它们组合成一个 ` TensorDataset ` 。

```python
# 创建两个特征张量
feature1 = torch.randn(100, 50)  # 100个样本，每个样本50维
feature2 = torch.randn(100, 30)  # 100个样本，每个样本30维 
# 创建 TensorDataset
dataset = TensorDataset(feature1, feature2) # 访问数据集中的特定样本
sample_feature1, sample_feature2 = dataset[0]
print(f"Sample Feature1 Shape: {sample_feature1.shape}")  # 输出: Sample Feature1 Shape: torch.Size([50])
print(f"Sample Feature2 Shape: {sample_feature2.shape}")  # 输出: Sample Feature2 Shape: torch.Size([30])
```

在这个例子中，我们创建了一个包含两个特征张量的 ` TensorDataset ` ，并通过 ` dataset[0] ` 访问第一个样本的两个特征。
#### DataLoader 详解

 ` DataLoader ` 主要用于批量加载数据，并支持多种[数据处理](https://so.csdn.net/so/search?q=%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86&amp;spm=1001.2101.3001.7020 )功能，如随机打乱、多线程加载等。
##### 1. [批量处理](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&amp;spm=1001.2101.3001.7020 )数据
 ` DataLoader ` 可以将数据集划分为多个批次（batch），便于[模型训练](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&amp;spm=1001.2101.3001.7020 )。


```python
from torch.utils.data import DataLoader 
# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
# 遍历 DataLoader
for batch_features, batch_labels in train_loader:    
	print(f"Batch Features Shape: {batch_features.shape}")  
	# 输出: Batch Features Shape: torch.Size([32, 3, 28, 28]) 
	print(f"Batch Labels Shape: {batch_labels.shape}")  # 输出: Batch Labels Shape: torch.Size([32])    
	# 这里可以进行训练操作，如前向传播、反向传播等
```

在这个例子中， ` train_loader ` 将数据集划分为大小为32的批次。通过遍历 ` train_loader ` ，我们可以轻松地获取每个批次的特征和标签。

##### 2. 数据打乱

 ` DataLoader ` 可以通过设置 ` shuffle=True ` 来在每个 epoch 开始时随机打乱数据，避免模型学习到数据的顺序。

```python
# 创建 DataLoader，并设置 shuffle=True
train_loader = DataLoader(dataset, batch_size=32, shuffle=True) 
# 遍历 DataLoader
for epoch in range(2):  
	# 假设我们要训练两个 epoch    
	for batch_features, batch_labels in train_loader:   
		print(f"Epoch {epoch}, Batch Features Shape: {batch_features.shape}")        
		# 这里可以进行训练操作
```

在这个例子中，每次 epoch 开始时，数据都会被随机打乱，确保模型不会受到数据顺序的影响。

##### 3. 多线程加载

 ` DataLoader ` 支持通过设置 ` num_workers ` 参数来使用多线程并行加载数据，加快数据读取速度。

```python
# 创建 DataLoader，并设置 num_workers=4
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
# 遍历 DataLoader
for batch_features, batch_labels in train_loader:    
	print(f"Batch Features Shape: {batch_features.shape}")    
	# 这里可以进行训练操作
```

在这个例子中，我们设置了 ` num_workers=4 ` ，表示使用4个线程来并行加载数据，从而加快数据读取速度。

#### 结合使用 TensorDataset 和 DataLoader

以下是一个完整的示例，展示了如何结合使用 ` TensorDataset ` 和 ` DataLoader ` 进行数据加载和训练。

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
# 创建输入数据和标签
images = torch.randn(1000, 3, 28, 28)  # 1000张图像，每张图像3通道，28x28像素
labels = torch.randint(0, 10, (1000,))  # 1000个标签，范围在0到9之间 
# 创建 TensorDataset
dataset = TensorDataset(images, labels) # 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4) 
# 遍历 DataLoader 进行训练
for epoch in range(2):    
	for batch_images, batch_labels in train_loader:        
		print(f"Epoch {epoch}, Batch Images Shape: {batch_images.shape}")        
		print(f"Epoch {epoch}, Batch Labels Shape: {batch_labels.shape}")        # 这里可以进行训练操作，如前向传播、反向传播等
```

在这个例子中，我们首先使用 ` TensorDataset ` 将图像和标签组合在一起，然后通过 ` DataLoader ` 进行批量加载和训练。通过设置 ` shuffle=True ` 和 ` num_workers=4 ` ，我们实现了数据的随机打乱和多线程加载。

#### 总结

 - **TensorDataset**用于将多个 ` Tensor ` 组合在一起，方便对数据进行统一处理。 
 - 可以组合特征和标签。
 - 可以组合多个特征张量。
 - **DataLoader**用于批量加载数据，支持多种数据处理功能。 
 - 支持批量处理数据。
 - 支持数据打乱。
 - 支持多线程加载。

# 18、pytorch中的transforms.[ToTensor](https://so.csdn.net/so/search?q=ToTensor&spm=1001.2101.3001.7020)和transforms.Normalize理解、transforms.resize
#### 文章目录

- [pytorch中的transforms.ToTensor和transforms.Normalize理解🌴](#pytorchtransformsToTensortransformsNormalize_10)
  
- - [transforms.ToTensor🌵](#transformsToTensor_12)
        
    - [transforms.Normalize🌵](#transformsNormalize_75)
      


### transforms.ToTensor🌵

  最近看pytorch时，遇到了对图像数据的归一化，如下图所示：

![image-20220416115017669](https://gitee.com/zhang-junjie123/picture/raw/master/image/0beba8d999bd216f19542257a0e7a8f4.png)

  该怎么理解这串代码呢？我们一句一句的来看，先看`transforms.ToTensor()`，我们可以先转到官方给的定义，如下图所示：

![image-20220416115331930](https://gitee.com/zhang-junjie123/picture/raw/master/image/5292fa11ee9daafaf8c9c77324ccbba6.png)

  大概的意思就是说，`transforms.ToTensor()`可以将PIL和numpy格式的数据从[0,255]范围转换到[0,1] ，具体做法其实就是将原始数据除以255。另外原始数据的shape是（H x W x C），通过`transforms.ToTensor()`后shape会变为（C x H x W）。这样说我觉得大家应该也是能理解的，这部分并不难，但想着还是用一些例子来加深大家的映像🌽🌽🌽

- 先导入一些包
  
```
 import cv2  
 import numpy as np  
 import torch  
 from torchvision import transforms
```

- 定义一个数组[模型](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&spm=1001.2101.3001.7020)图片，注意数组数据类型需要时np.uint8【官方图示中给出】
  
```
data = np.array([  
			 [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],  
			 [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],  
			 [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],  
			 [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],  
			 [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]  
	 ],dtype='uint8')
```

这是可以看看data的shape，注意现在为（W H C）。
    
![image-20220416120518895](https://gitee.com/zhang-junjie123/picture/raw/master/image/37da4fd4becceba642e1b122a63019cb.png)
    
- 使用`transforms.ToTensor()`将data进行转换
  
```
data = transforms.ToTensor()(data)
```

这时候我们来看看data中的数据及shape。
    
![image-20220416120811156](https://gitee.com/zhang-junjie123/picture/raw/master/image/4d5b0f134386cf8b247a87e76cc569cf.png)
    
很明显，数据现在都映射到了[0, 1]之间，并且data的shape发生了变换。
    

> **注意：不知道大家是如何理解三维数组的，这里提供我的一个方法。**🥝🥝🥝
> 
> 🌼**原始的data的shape为（5，5，3），则其表示有5个（5 ， 3）的二维数组，即我们把最外层的[]去掉就得到了5个五行三列的数据。**
> 
> 🌼**同样的，变换后data的shape为（3，5，5），则其表示有3个（5 ， 5）的二维数组，即我们把最外层的[]去掉就得到了3个五行五列的数据。**

---
### transforms.Normalize🌵

相信通过前面的叙述大家应该对`transforms.ToTensor`有了一定的了解，下面将来说说这个`transforms.Normalize`🍹🍹🍹同样的，我们先给出官方的定义，如下图所示：

![image-20220416195418909](https://gitee.com/zhang-junjie123/picture/raw/master/image/5eba422a7423faba96cda38f17750238.png)

可以看到这个[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)的输出`output[channel] = (input[channel] - mean[channel]) / std[channel]`。这里[channel]的意思是指对特征图的每个通道都进行这样的操作。【mean为均值，std为标准差】接下来我们看第一张图片中的代码，即

![image-20220416200305159](https://gitee.com/zhang-junjie123/picture/raw/master/image/74b85022f1ca118a06d0debf1174c9d7.png)

 这里的第一个参数（0.5，0.5，0.5）表示每个通道的均值都是0.5，第二个参数（0.5，0.5，0.5）表示每个通道的方差都为0.5。【因为图像一般是三个通道，所以这里的向量都是1x3的🍵🍵🍵】有了这两个参数后，当我们传入一个图像时，就会按照上面的公式对图像进行变换。【**注意：这里说图像其实也不够准确，因为这个函数传入的格式不能为PIL Image，我们应该先将其转换为Tensor格式**】

 说了这么多，那么这个函数到底有什么用呢？我们通过前面的ToTensor已经将数据归一化到了0-1之间，现在又接上了一个Normalize函数有什么用呢？其实Normalize函数做的是将数据变换到了[-1,1]之间。之前的数据为0-1，当取0时，`output =（0 - 0.5）/ 0.5 = -1`；当取1时，`output =（1 - 0.5）/ 0.5 = 1`。这样就把数据统一到了[-1，1]之间了🌱🌱🌱那么问题又来了，数据统一到[-1，1]有什么好处呢？数据如果分布在(0,1)之间，可能实际的bias，就是神经网络的输入b会比较大，而模型初始化时b=0的，这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。【这句话是再网络上找到最多的解释，自己也不确定其正确性】

 读到这里大家是不是以为就完了呢？这里还想和大家唠上一唠🍓🍓🍓上面的两个参数（0.5，0.5，0.5）是怎么得来的呢？这是根据数据集中的数据计算出的均值和标准差，所以往往不同的数据集这两个值是不同的🍏🍏🍏这里再举一个例子帮助大家理解其计算过程。同样采用上文例子中提到的数据。

- 上文已经得到了经ToTensor转换后的数据，现需要求出该数据每个通道的mean和std。【这一部分建议大家自己运行看看每一步的结果🌵🌵🌵】
  
```
 # 需要对数据进行扩维，增加batch维度  
 data = torch.unsqueeze(data,0)    #在pytorch中一般都是（batch,C,H,W）  
 nb_samples = 0.  
 #创建3维的空列表  
 channel_mean = torch.zeros(3)  
 channel_std = torch.zeros(3)  
 N, C, H, W = data.shape[:4]  
 data = data.view(N, C, -1)  #将数据的H,W合并  
 #展平后，w,h属于第2维度，对他们求平均，sum(0)为将同一纬度的数据累加  
 channel_mean += data.mean(2).sum(0)    
 #展平后，w,h属于第2维度，对他们求标准差，sum(0)为将同一纬度的数据累加  
 channel_std += data.std(2).sum(0)  
 #获取所有batch的数据，这里为1  
 nb_samples += N  
 #获取同一batch的均值和标准差  
 channel_mean /= nb_samples  
 channel_std /= nb_samples  
 print(channel_mean, channel_std)   #结果为tensor([0.0118, 0.0118, 0.0118]) tensor([0.0057, 0.0057, 0.0057])  
```
     ​

- 将上述得到的mean和std带入公式，计算输出。
  
```
 for i in range(3):  
	 data[i] = (data[i] - channel_mean[i]) / channel_std[i]  
 print(data)
    
```
输出结果：

![image-20220416205341050](https://gitee.com/zhang-junjie123/picture/raw/master/image/b7ad0fbb1a108976422d86f7f9c5ae19.png)

从结果可以看出，我们计算的mean和std并不是0.5，且最后的结果也没有在[-1，1]之间。
    

最后我们再来看一个有意思的例子，我们得到了最终的结果，要是我们想要变回去怎么办，其实很简单啦，就是一个逆运算，即`input = std*output + mean`,然后再乘上255就可以得到原始的结果了。很多人获取吐槽了，这也叫有趣！！？？哈哈哈这里我想说的是另外的一个事，**如果我们对一张图像进行了归一化，这时候你用归一化后的数据显示这张图像的时候，会发现同样会是原图。**感兴趣的大家可以去试试🥗🥗🥗🥗这里给出一个参考链接：[https://blog.csdn.net/xjp_xujiping/article/details/102981117](https://blog.csdn.net/xjp_xujiping/article/details/102981117)

> 参考链接1：[https://zhuanlan.zhihu.com/p/414242338](https://zhuanlan.zhihu.com/p/414242338)
> 
> 参考链接2：[https://blog.csdn.net/peacefairy/article/details/108020179](https://blog.csdn.net/peacefairy/article/details/108020179?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_antiscanv2&utm_relevant_index=2)


### transforms.Resize(resize)
 transforms.Resize(resize) 预定义的转换函数，**它的作用是将图像调整为指定的大小（resize），可以是一个整数或一个元组。**

如若文章对你有所帮助，那就🛴🛴🛴
本文转自 [https://blog.csdn.net/qq_47233366/article/details/124225860?ops_request_misc=%257B%2522request%255Fid%2522%253A%252292ed14f42f0f1ac78cb4e511cbb8a796%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=92ed14f42f0f1ac78cb4e511cbb8a796&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124225860-null-null.142^v100^pc_search_result_base5&utm_term=transforms.ToTensor%28%29&spm=1018.2226.3001.4187](https://blog.csdn.net/qq_47233366/article/details/124225860?ops_request_misc=%257B%2522request%255Fid%2522%253A%252292ed14f42f0f1ac78cb4e511cbb8a796%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=92ed14f42f0f1ac78cb4e511cbb8a796&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124225860-null-null.142^v100^pc_search_result_base5&utm_term=transforms.ToTensor%28%29&spm=1018.2226.3001.4187)，如有侵权，请联系删除。

 

# 19、auto_grad 自动微分

 - https://blog.csdn.net/sinat_28731575/article/details/90342082

- https://zhuanlan.zhihu.com/p/29923090

- https://zhuanlan.zhihu.com/p/65609544

# 20、torch.nn.Parameter()

[[PyTorch中的torch.nn.Parameter() 详解-CSDN博客]]

**简单总结：**
首先可以把这个函数理解为类型[转换函数](https://edu.csdn.net/cloud/houjie?utm_source=highword&spm=1001.2101.3001.7020)，将一个不可训练的类型`Tensor`转换成可以训练的类型`parameter`并将这个`parameter`绑定到这个`module`里面(`net.parameter()`中就有这个绑定的`parameter`，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个`self.v`变成了[模型](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&spm=1001.2101.3001.7020)的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。


# 21、torch.view和torch.reshape
[[Pytorch_ view()和reshape()的区别？他们与continues()的关系是什么？]]
正常用，基本没区别



# 22、model.apply(fn)或net.apply(fn)
[[model.apply(fn)或net.apply(fn)-CSDN博客]]

pytorch中的`model.apply(fn)`会递归地将函数`fn`应用到父模块的每个子模块`submodule`，也包括`model`这个父模块自身。 
fn的参数就是每个模块。

# 23、torch.nn.MSELoss损失函数

MSE: Mean Squared Error（**均方误差**） 含义：**均方误差**，是预测值与真实值之差的平方和的平均值，即： 
$$
\begin{aligned} MSE =\cfrac {1}{N}\sum_{i=1}^n(x_i-y_i)^2 \end{aligned} 
$$
但是，在具体的应用中跟定义稍有不同。主要差别是参数的设置，在[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).nn.MSELoss中有一个reduction参数。reduction是维度要不要缩减以及如何缩减主要有三个选项：

- **‘none’**:no reduction will be applied.
  
- **‘mean’**: the sum of the output will be divided by the number of elements in the output.
  
- **‘sum’**: the output will be summed.
  

  如果不设置reduction参数，**默认是’mean’**。 下面看个例子：

```python
 import torch  
 import torch.nn as nn  
    
 a = torch.tensor([[1, 2],   
                   [3, 4]], dtype=torch.float)  
                     
 b = torch.tensor([[3, 5],   
                   [8, 6]], dtype=torch.float)  
    
 loss_fn1 = torch.nn.MSELoss(reduction='none')  
 loss1 = loss_fn1(a.float(), b.float())  
 print(loss1)   # 输出结果：tensor([[ 4.,  9.],  
                #                 [25.,  4.]])  
    
 loss_fn2 = torch.nn.MSELoss(reduction='sum')  
 loss2 = loss_fn2(a.float(), b.float())  
 print(loss2)   # 输出结果：tensor(42.)  
    
    
 loss_fn3 = torch.nn.MSELoss(reduction='mean')  
 loss3 = loss_fn3(a.float(), b.float())  
 print(loss3)   # 输出结果：tensor(10.5000)
```

在loss1中是按照原始维度输出，即对应位置的元素相减然后求平方；loss2中是对应位置求和；loss3中是对应位置求和后取平均。   
除此之外，torch.nn.MSELoss还有一个妙用，**求矩阵的F范数**[（F范数详解）](https://blog.csdn.net/zfhsfdhdfajhsr/article/details/115639274?spm=1001.2014.3001.5501)当然对于所求出来的结果还需要开方。

### 参考文献

[[1]pytorch的nn.MSELoss损失函数]([https://www.cnblogs.com/picassooo/p/13591663.html](https://www.cnblogs.com/picassooo/p/13591663.html)) [[2]状态估计的基本概念（3）最小均方估计和最小均方误差估计]([https://zhuanlan.zhihu.com/p/119432387](https://zhuanlan.zhihu.com/p/119432387))

本文转自 [https://blog.csdn.net/zfhsfdhdfajhsr/article/details/115637954](https://blog.csdn.net/zfhsfdhdfajhsr/article/details/115637954)，如有侵权，请联系删除。

 
# 24、torch.expand和repeat函数
[[【Pytorch】对比expand和repeat函数]]
​                

​        

