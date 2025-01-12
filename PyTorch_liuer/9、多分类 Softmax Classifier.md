---
created: 2024-10-05T17:33
updated: 2024-10-05T21:29
---
#### Design 10 outputs using Sigmoid

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005204041.png)

多分类希望：
- 概率大于等于0
- 求和为1
- <mark style="background: FFFF00;">输出多种类别的概率是一个分布。</mark> 

### Output a Distribution of prediction with Softmax function
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005204326.png)
-<mark style="background: #FF0000;"> softmax可以保证上面的需求。</mark>
#### Softmax Layer
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005204640.png)

#### Example
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005204833.png)
#### Cross Entropy in Numpy
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005205244.png)
#### Cross Entropy in Pytorch
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005205323.png)
- 注意：最后一层不用激活了
- 交叉熵损失函数也做了softmax和log.
- 注意 Y生成独热向量，需要长整型的张量

Mini-Batch: batch_size = 3
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005205735.png)
- 从线性层出来的Y_pred就可以看出来 loss1更小。

#### Exercise
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005205830.png)

#### In MNIST Dataset
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005210041.png)
- 0到255，映射到 0到1之间。

#### Implementation of classifier to MNIST Dataset
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005210150.png)

##### 0 . Import package
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005210243.png)
- relu更流行

##### 1 .Prepare Dataset

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005210703.png)

c是channel 通道

- 为了更高效的处理，比如卷积操作，做如下操作：
- 通过cv读取图像进来是 w * H * c,转到pytorch 里面会转为 c * w * h

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005210822.png)
- 从PIL 到 Tensor 会将图像转为张量 c * w * h,同时范围在0到1。
- normalize: 就是标准化 减去均值 除以标准差；第一个是mean,第二个是std。

##### 2 .Design Model

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005211738.png)

- 将张量(展开）转为二阶的矩阵，第一个数-1会自动计算维度是多少，显然是N，784是28 * 28，也就是一个图像的所有数值点的个数.
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005211756.png)

##### 3 .Construct Loss and Optimizer
- 选择交叉熵
- 模型比较大——SGD 添加了 冲量来训练。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005211924.png)
##### 4 .Train and Test
- Train函数的每一轮作为一个函数，将epoch作为输入。
- 每300轮，输出一次。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005212044.png)
- Test函数，和上面一样

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005212247.png)
- no_grad()不会计算梯度。
- torch.max dim是维度找最大值，返回的值是最大值和对应下标
- labels.size(0) 每个batch的元素个数，因为是列向量所以是N * 1 ，size(0)就是第一个值。
- correct是猜对的量。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005212807.png)
- 问题是没有区别抽象的数据点和其他数据的权重区别，会导致训练结果不太好。
#### Exercise
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005212929.png)
