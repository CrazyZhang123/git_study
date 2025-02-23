---
created: 2024-09-30T17:44
updated: 2024-09-30T18:45
---


- 分类问题不能使用回归的方法输出y值，因为各个类别之间不存在数值关系，数值大小没有意义。
- torchvision包可以下载一些公共数据集，如。

##### MINSIT数据集
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930175539.png)

##### CIFAR-10数据集
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930175645.png)

### 分类问题

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930180006.png)
我们需要计算属于不同类别的概率，所以输出结果应该为0到1之间，因此需要映射到0和1之间。

#### How to map :R->\[0,1]

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930180246.png)

#### Sigmod functions

<mark style="background: FFFF00;">Sigmod函数最出名的就是Logistic Function</mark>，所以上面的函数可以叫Sigmod函数。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930180514.png)

#### Logistic Regression Model

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930180835.png)

#### Binary Classification

从之间数值的差异，到现在不同类别的差异，有KL散度，cross-entropy交叉熵。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930181117.png)

确保真实值向y去接近。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930183704.png)

[交叉熵及实现讲解](https://zhuanlan.zhihu.com/p/98785902?ivk_sa=1024320u)

[Pytorch常用的交叉熵损失函数CrossEntropyLoss()详解](../其他资料/Pytorch常用的交叉熵损失函数CrossEntropyLoss()详解.md)

softmax的目的是归一化，并且使得相加和为1.

##### Mini-Batch 公式
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930183830.png)

#### Implementation of Logistic Regression

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930183903.png)

- torch.nn.functional as F ，F里面存着许多函数比如sigmod函数(Logistic函数)

##### new Loss——BCE Loss
新损失——BCELoss
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930184117.png)

#### Logistic Regression Implementation
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930184228.png)


#### Result of Logistic Regression

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930184539.png)
