---
created: 2024-10-06T21:29
updated: 2024-10-07T15:29
---

RNN 适用于处理序列数据，比如天气数据，自然语言(我爱中国)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006214444.png)

#### What is RNN ?
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006215517.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006215634.png)
- 循环神经网络RNN 通常用**tanh**
- RNN Cell 实际只做了一次线性层
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006220112.png)
- ==h和x都要进行 线性运算，但是权重不同。==

#### RNN Cell in PyTorch
- 通过上面的示意图，可以看到，内部的权重维度需要 input_size 和 hidden_size来确定所有的W,输出的维度也是一样，因此参数就是这两个。
- <mark style="background: FFFF00;">batch就是模型训练的最小单位</mark>
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006220417.png)

#### How to use RNN Cell

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006220708.png)
- seq_len = 3,从下面的可以理解，总共3个大的对象，for每次取一个。
- seq_len是数据序列的长度，就是单个时间单元，但一个时间单元里面可能的batch_size大于1，有这么多个的输入特征向量。
![image.png|517](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007124200.png)
在PyTorch中，处理序列数据时，数据的维度通常遵循`(seq_len, batch_size, *)`的格式，其中`seq_len`是序列的长度，`batch_size`是批处理大小，`*`表示其他可能的维度（如`input_size`表示输入特征的维度）。这种格式允许PyTorch高效地处理批量序列数据。

==然而，当你使用`torch.nn.RNNCell`时，需要注意一点：`RNNCell`是处理单个时间步的RNN单元。这意味着你需要为每个时间步单独调用`RNNCell`，并传递当前时间步的输入和上一个时间步的隐藏状态。==

在你的例子中，`dataset`被设置为三维张量`(seq_len, batch_size, input_size)`，这是因为你可能原本打算使用`torch.nn.RNN`（它接受这种维度的输入）而不是`torch.nn.RNNCell`。==`torch.nn.RNN`可以一次性处理整个序列，而`torch.nn.RNNCell`则需要你手动迭代序列的每个时间步。==

为了与`RNNCell`兼容，你需要在循环中迭代`dataset`的每个时间步，并将每个时间步的输入（形状为`(batch_size, input_size)`）传递给`RNNCell`。你的代码已经正确地做到了这一点，通过`enumerate(dataset)`来迭代`dataset`，并在每次迭代中取出当前时间步的输入`input_`（注意，这里`input_`的形状实际上是`(batch_size, input_size)`，因为`dataset`是一个三维张量，当你通过索引访问它时，它会返回一个减少了一个维度的张量）。

#### How to use RNN
- RNN 运算很耗时
- numLayers 隐层的层数
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006230708.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006230820.png)
- 尤其注意h0,hn的维度都是 numLayers,batchsize,hiddensize；
numLayers解释
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006230923.png)

- 左侧的h个数就是numLayers的个数，表示有多少层；
- 上面的是output,右侧的是输出的hidden


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006231504.png)
结果：
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007152746.png)


- <mark style="background: #ADCCFFA6;">设置batch_first = True</mark>，所有的第一个维度就是batch_size了，比如input batch_size,seq_len,input_size
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006231732.png)

##### Example 12-1 : Using RNN Cell
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006232144.png)
- 字母进行向量化 —— 独热编码 <mark style="background: FFFF00;">one hot</mark>
 ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006232326.png)<mark style="background: FFFF00;">input_size = 4 </mark>,因为编码完helo只需要4个数字，转换成独热向量后，只有对应索引位置的值为1，其他都是0.
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233240.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233334.png)

#### Code

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233353.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233523.png)
labels 维度应该是 (seqLen , 1)

这里是==1-gram方法==，只用一个词预测下一个词，所以seqLen中每一个seq拥有一个batchsize也就是1的batch，来计算损失。
**交叉熵损失的核心是 “输入是类别概率分布（对数 ），目标是类别索引”**，维度必须严格对齐 `(batch_size, num_classes)` 和 `(batch_size)` 。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233717.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233717.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006233823.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234011.png)
所有的损失都要计算，构成计算图。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234059.png)
- hidden.max用来查找4种字母哪个概率高，输出预测结果 用idx2char
##### Result 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234302.png)


#### Example RNN Module
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234743.png)
![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234343.png)


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234654.png)



##### Result 

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234814.png)

#### Question

##### one-hot encoding
- 维度高
- 分散，稀疏
- 硬编码
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006234948.png)
解决：EMBEDDING 嵌入层
##### One-hot VS EMBEDDING
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235024.png)

##### Embedding in PyTorch
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235247.png)
通过矩阵乘法，取出来对应的权重进行反向传播

#### Example 12-3 Using embedding and linear layer
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235358.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235431.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235513.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235549.png)

#### Coding 

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235621.png)
emb
![image.png|295](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235634.png)

batch_first
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235708.png)
FC
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235723.png)
view 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235738.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235802.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235811.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235816.png)

##### Result 
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235847.png)


#### Exercise 12-1 Using LSTM
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006235935.png)
神经网络的万能逼近性

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007000210.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007000329.png)

##### GRU

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007000408.png)

LSTM计算复杂，性能高；GRU计算快。
