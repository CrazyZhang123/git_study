---
created: 2024-09-26T17:16
updated: 2024-09-27T11:40
---

### 1、深度学习的一般流程：
- dataSet
- Model  模型
- Train   训练
- infer    推断

### 2、提出问题
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172042.png)

#### 2.1机器学习流程：
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172131.png)
- 输入dataset,训练模型，后续只需要输入数据就可以推断出结果。
- 有明确的标签的训练和测试集  属于 supervised Learning 监督学习
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172439.png)
- kaggle是机器学习数据分析领域的一个竞赛，会给参赛者一个数据集，等参赛者提交一份模型代码，使用未给的额外数据去测试模型。


<mark style="background: #ADCCFFA6;">模型的泛化能力</mark>：对于训练好的模型，在没有参与训练的数据集中，也能展示出很好的性能。

#### 2.2 模型设计 Model design

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926173926.png)
#### 线性回归
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926174334.png)
先找一个随机值w,然后来评估目前的模型。


#### 计算损失 Compute Loss

单个样本的损失->求所有样本的平均损失->调整w来降低样本的平均损失。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926174924.png)

样本的损失  
$$
loss = (\hat y - y)^2 = (x * \omega - y)^2
$$
数据集的损失  cost fuction  
均方误差mean squared error
$$
cost = \frac{1}{N}\sum_{n=1}^{N} (\hat y_n - y_n)^2
$$

可以通过枚举的方法来绘制关于w的损失曲线
![image.png|700](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926181406.png)
- 但是后续画的图 横坐标是训练轮数(epoch)  纵坐标是损失
- 可视化工具 visdom


练习
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926183113.png)
