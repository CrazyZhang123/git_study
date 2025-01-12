---
created: 2024-09-30T22:00
updated: 2024-12-06T21:29
---

#### 之前的训练方法——使用所有的数据集计算损失
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930220336.png)
- from_numpy 将numpy转为tensor
#### Terminology : Epoch ,Batch-size,Iterations

- Iterations = Total-size / Batch-size
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930220548.png)

#### DataLoader: batch_size = 2, shuffle = True(打乱顺序)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930220753.png)

##### How to define Dataset
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930220901.png)

- Dataset是抽象类，创建的新类需要继承它。
- 魔法方法 getitem方法：实现 dataset\[index]
- 魔法方法 len方法 可以使用len()返回dataset长度
- (1) init 中加载所有数据（数据量小）
	- (2) 定义读写的方法（数据量大，图像，语音），数据打包
- DataLoader类
	- dataset,hatch_size,shuffle,num_workers=2(2个并行进程)
- num_workers的问题(pytorch 0.4)
	- 由于linux和windows创建进程的底层api不同，linux(fork),win(spawn)。
	- 解决办法：用一个if或者其他的把整个训练过程封装起来，就不会报错了
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930222212.png)

#### Complete Implement
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930222528.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930222632.png)
- 每个i就是 对应的batch size
- inputs ,Labels都是张量
- enumerate(iterable,start) start开始索引


#### Available Dataset
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930223422.png)


##### MNIST Dataset

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930223605.png)
- transform变换，转换成张量 torch.ToTensor()
- 训练需要shuffle设置为True


##### Exercise :

![image.png|546](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930223913.png)
