---
created: 2024-09-28T11:52
updated: 2024-09-29T19:14
---
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240928175016.png)

- 观察法：如果要搜索的权重比较多，随着搜索维度的增加，搜索的数据量会急剧增加。
- 分治法：先去4(n)个点,看哪个loss比较低，然后在loss小的附近详细搜索。
	- 可能会出现局部最优，但不是全局最优。
- 梯度下降算法(Gradient Descent)： 
	- 如下图，右侧的导数(梯度)为正，寻找loss减小的方向应该向左，所以更新的方向是负的梯度。
	- 左侧的导数(梯度)为负数，寻找loss减小的方向应该向右，所以更新的方向是负的梯度。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240928180013.png)
- 更新公式updating formula:
$$
\omega = \omega - \alpha \frac{\partial cost}{\partial \omega }
$$

-   $\alpha$是学习率，学习率一般比较小。

```ad-note
图形判断（直观方法）

通过观察函数的图形，也可以直观地判断其凸凹性。<mark style="background: #ADCCFFA6;">凸函数的图形在其上任意两点之间的连线总是位于图形之上，而凹函数的图形则在其上任意两点之间的连线之下。</mark>


```


##### $\hat{y} = \omega * x$的梯度下降算法


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929183817.png)

##### Implement 实现

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929184739.png)

- cost曲线如果不够光滑，用[指数加权平滑](https://blog.csdn.net/sinat_18127633/article/details/88371916)的方法。
- cost曲线经过低点后整体回升，说明训练发散了，训练失败了。

	- 可能的原因：①学习率$\alpha$太大 

随机梯度下降 Stochastic Gradient Descent
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929190012.png)

- 原因：用于解决cost梯度下降可能有鞍点，无法推动训练，使用随机梯度下降，可以使用随机一个单个样本的梯度进行梯度下降，来跳出鞍点。

##### 随机梯度下降实现
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929190520.png)


- 各个预测值$\hat{y}_{i}$受到前一个$\hat{y}_{i-1}$的影响，因为是把每一个数据点进行更新权重w了，所以会影响后面的值，时间复杂度高。

##### 梯度下降和随机梯度下降对比
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929190646.png)

- 正常Batch指的是所以的训练集，但是Mini-Batch比较流行，现在说的Batch都是Mini-Batch，把样本分成很多的小的Batch，这样兼顾的性能和时间。

##### 优化失败的原因 Optimization Fail

也就是梯度 Gradient 为0的情况
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929172246.png)
 
(1)<mark style="background: #FF0000;">不一定能找到全局最优，可以找局部最优。</mark> 但实际情况能找到全局最优，局部最优没有这么普遍。
(2)鞍点问题
	在鞍点梯度g = 0,就导致w=w-ag无法更新了
	注意鞍点 saddle point 在上面正视从左到右是最低点，从前往后是最高点，这个点就是saddle point


学习

[[区分鞍点和局部最优]](李宏毅老师)
[[深度学习详解：局部极小值与鞍点]](代码部分讲解)



