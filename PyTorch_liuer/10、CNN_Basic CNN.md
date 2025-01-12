---
created: 2024-10-05T21:30
updated: 2024-11-15T22:23
---
- CNN: convolutional(卷积) Neural Network
- Linear Layer全连接层(Fully connected neural network)：输入和输出之间都有权重。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005213357.png)

#### Convolutional Neural Network
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241005214416.png)

##### 光敏电阻示例
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006162925.png)
- 将一个图像通过透镜照射到对应的光敏电阻上面，不同光锥的光强就会影响不同位置的光敏电阻，这样就会使不同位置的光敏电阻的值不同，对应电流和电压发生改变，根据光敏电阻的特性曲线，由阻值可以转换到光强，这样就完成了图像的矩阵化。
 ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006163229.png)
- 彩色照片
	- 使用RGB的三种光敏电阻，将电阻值最后组合起来，转成0-255之间，得到图像。

#### Connvolution 卷积
- 栅格图像就是上述光明电阻得到的像素矩阵。
- 矢量图，是程序生成的，每次都是现场画出来的。
	- 举例：圆 —— 有一个圆，边的颜色，填充什么。
- <mark style="background: #FF0000;">卷积的对象是patch</mark>，输出的channel，w,h都可能变，，但是包含了原始图像对应patch的所有信息。
- 坐标系原点 在左上角

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006164242.png)
##### Single Input Channel 单通道卷积
- 在对应的块和kernel做数乘(点积)，得到对应的输出。
- 移动规则：每次向右，向下移动一个像素点
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006164512.png)

##### 3 input Channels 卷积
- 每个输入channel都要设置对应的卷积核，得到对应的输出矩阵应该对应输出矩阵相加。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006164703.png)

##### N input channel
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006164955.png)
#####  多卷积核
- 每个卷积核的channel数和输入的channel数一样
- 输出的channel数和输入的卷积核的个数一样
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006165226.png)
###### Convolutional layer
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006165425.png)
#### 1\*1 卷积的作用
###### 作用
以图片为例，只改变数据特征的通道数，图片数据的高（H）与宽（W）维持，不发生改变。

###### 介绍
      使用 1x1 的卷积操作可以实现通道数的改变，这是因为 1x1 卷积的作用是在每个空间位置进行逐元素的线性变换。虽然它的卷积核大小很小，但它能够改变数据的通道数。
下面是对这个过程的解释： 

1.输入数据的形状为 \[B, C, H, W]，其中 B 是批次大小（batch size），C 是输入数据的通道数（channels），H 和 W 是输入数据的高度和宽度。
2.对输入数据应用 1x1 的卷积操作，卷积核的形状为 \[C', C]，其中 C' 是输出数据的通道数。
3.在每个空间位置，1x1 卷积会对输入数据的通道维度进行线性变换，将输入通道的每个元素与对应的卷积核参数进行点乘并相加，产生输出通道的一个元素。
4.经过 1x1 卷积操作后，输出数据的形状变为 \[B, C', H, W]，其中 B 是批次大小，C' 是输出数据的通道数，H 和 W 保持不变。

- 通过这种方式，1x1 卷积操作可以改变数据的通道数，从而在深度学习模型中起到调整和控制特征通道维度的作用。它通常被用于改变特征的维度、调整网络的参数量和计算量，并且可以在不改变特征图空间尺寸的情况下引入非线性变换。
- 由于 1x1 卷积操作的计算量较小，它在很多深度学习模型中被广泛使用，例如用于降低通道数、增加非线性表达能力以及实现特征的维度变换等。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006170550.png)
- input , output第一个是batch_size 1 
- input n * w * h    5 * 100 * 100
- output m * w * h    10 * (width - kernel_size + 1) * (height - kernel_size + 1)   也就是 10 * 98 * 98
- conv_layer m * n* k_w* k_h  10* 5* 3* 3

##### padding 操作
- 往输入矩阵外面添加0.
- 为了保证输入和输出效果大小一样，向外添加的行数就是 <mark style="background: #FF0000;">kernel_size / 2</mark>

padding = 1
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006171547.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006171639.png)

##### stride 步长
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006171745.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006171807.png)
降低输出维度。


##### Max Pooling layer 最大池化层
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006173741.png)

每个通道去处理，找最大值；通道大小不变。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006173944.png)
- k_size是2，默认步长是2
- 默认步长和k_size一样

#### A Simple Convolutional N N
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006174352.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006174527.png)

Implementation
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006181304.png)

- 用交叉熵损失最后一层不需要激活。

#### How to use GPU ——
##### 1.Move Model
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006181909.png)
- 使用device和to进行迁移。
##### 2.Move Tensors to GPU
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006182128.png)

#### Results
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006182206.png)
- 3%->2%   错误率降低 1/3 ,哈哈哈

##### Exercise 10
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241006182437.png)

