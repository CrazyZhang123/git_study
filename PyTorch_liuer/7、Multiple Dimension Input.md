---
created: 2024-09-30T18:46
updated: 2024-10-01T22:45
---

#### Diabetes Dataset   糖尿病数据集
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930212415.png)


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930212934.png)

#### Mini-Batch (N samples)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930213233.png)

##### 代码修改 
全连接层  Linear(输入特征大小，输出特征大小)

多输入就把第一个变成大于1的

![image.png|571](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930213322.png)


矩阵是空间变换的函数。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930213805.png)

A是一个将N维空间转为M维的空间变换的函数。

神经网络是寻找一个非线性变化的函数。


##### Linear Layer
![image.png|603](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930214037.png)

#### Neural Network
一般中间层数越多，非线性变化越强，效果更好。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930214513.png)

必须要有泛化能力，不然容易过拟合。
<mark style="background: #FF0000;">可以通过超参数搜索进行。</mark>

##### Example 1 : Artifical Neural Network

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930214635.png)

##### Example 2 : Diabetes Prediction

![image.png|572](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930214711.png)

Y是评估糖尿病下年是否会家中。

##### (1) Prepare Dataset

一般普通的显卡只支持float32,更高级的显卡才支持double

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930214937.png)
- x_data 不要最后一列
- y_data 最后一列
- torch.from_numpy 会根据ndarray创建Tensor


##### (2) Define Model

![image.png|578](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930215410.png)

- forward里面输出全用x，这样才能保证每层以上一层的输出作为输入，如果用不同的变量可以会遗漏。
- 多种Sigmoid函数的区别:
- ![[三种Sigmoid的区别#^200f74]]
##### (3) Construct Loss and Optimizer

还是二分类用BCELoss
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930215534.png)

##### (4) Training Cycle
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930215656.png)

#### Exercise :Try different activate function
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930215736.png)


##### 修改位置
![image.png|499](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930215832.png)
