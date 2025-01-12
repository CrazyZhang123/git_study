---
created: 2024-10-06T18:24
updated: 2024-10-06T21:28
---

### 1、GoogleLeNet

减少代码冗余： 函数，类。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006201648.png)
- 红色的就是Inception

#### Inception Module
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006202113.png)
- 1 * 1卷积  需要的个数和输入的channel一样。
- 计算方式：
- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006202409.png)
- 多通道m，输出就会变成m个channel了
- 原因：降低通道数，加快训练速度和减少计算次数
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006203039.png)

##### Implementation of Inception Module
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006203715.png)
- padding 是为了保证图像卷积池化后大小不变。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006203821.png)
- torch.cat(dim = 1)  要按照通道拼接起来。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006204144.png)
- mp池化改变的是w,h  不会改变通道数。

##### Results
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006204640.png)


### 2、ResetNet
 ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006205119.png)
- 避免梯度消失
- 当梯度减小到0附近，由于+x后，梯度就会靠近1附近，然后和全连接层处理后，就可以避免梯度消失。
- 注意 x和F(x)的通道，宽高等等都一样。

##### Reset Network
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006211759.png)


#### Implementation of Residual Block
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006211915.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006211953.png)
- 第二个relu激活的对象是 y+x；是卷积后的结果和输入的加法

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006212122.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006212248.png)

#### Exercise 11-1
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006212341.png)

##### Exercise  11-2
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006212438.png)

#### 后续学习
1、理论角度 《深度学习》花书
2、阅读Pytorch文档
	api,
3、复现经典工作
  学习过程
  读代码 <-> 写代码
4、查看论文，扩充视野

