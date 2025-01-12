---
created: 2024-10-07T00:05
updated: 2024-10-07T21:52
---

### Name Classification
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007200546.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007200514.png)

改进模型
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201005.png)

### Our Model
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201147.png)
- 输入序列的长度不一致

#### Implementation - 1、Preparing Data
- 处理人名Name
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201741.png)
- 将字符串转换为ascii序列
- 解决输入不一致的问题，用padding填充0
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201847.png)

- 国家类别——构造字典
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201928.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007202017.png)
- gz的读取  用包gz,和np
- rows 格式是  name,language
- 将国家转换为词典 getCountryDict(),具体在getitem里面实现。
![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007202017.png)
- getitem函数 先拿出来国家(countries\[index]) 再通过字典取到对应的类别值(country_dict\[index])
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007202804.png)
- 函数
	- getCountryDict构造国家字典
	- idx2country 通过索引取到国家字符串
	- getCountriesNum 获取总共有多少国家
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007203030.png)

#### Implementation - 2、Model Design
![image.png|603](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007204548.png)
- hidden_size , n_layers 和 GRU有关。
- <mark style="background: FFFF00;">n_directions 含义</mark>
	- 双向和单向
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007203938.png)
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007204045.png)
	- 最终输出 output是上面的，hidden = $[h_{N}^f,h_{N}^b]$

	

- embeddiing的输入和输出维度
	- ![image.png|619](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201318.png)
- GRU层维度解析
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007203320.png)
- fc层 因为GRU设置是双向的，所有输出的hidden是前面维度拼接一次的，所有GRU的输出应该是hidden_size * n_directions。
![image.png|569](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007204548.png)
- input shape转置了，之前是b * s，现在转置为 s * b; ？
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007210329.png)
- embedding 维度
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007210753.png)
- pack_padded_sequence
	- 输入 embedding,seq_lengths(每个样本的序列长度)
	- 为了解决填充完0后，不让这些进入计算过程，加快计算速度。
![image.png|573](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007211317.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007211510.png)
- 横着的是batch_size。这也是GRU可以接收的输入 packedSequence对象。
- 如果是双向的，需要把hidden拼接
	- ![image.png|514](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007211809.png)

#### Implementation - Convert name to tensor
- 从name到字符列表->ascii列表->填充->转置->按seqlen排序
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007212151.png)
- 序列和长度
	- name2list函数
	- ![image.png|381](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007212203.png)
- country之前是整数，现在变成long类型。
- 填充0
	- 先构造全是0的向量
	- 然后把原始向量赋值过去(按照长度)
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007212637.png)
- 按照序列长度排序——国家和向量
	- sort排序返回排序结果，和对应的原始的索引
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007212722.png)
- 创建tensor
	- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007212936.png)

#### Implementation - One epoch Train

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213047.png)


#### Implementation - Testing
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213124.png)

#### Implementation - Main Cycle
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201318.png)
- N_CHARS是字符的数量；N_COUNTRY类别,N_LAYER几层模型。
- 使用GPU
- 交叉熵和Adam优化器
- start 计时
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201520.png)
- 训练和测试封装成函数，训练结果添加到acc里面,可以绘图
	- trainModel(),testModel()
	- ![image.png|344](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007201626.png)

#### Implementation - Result
![image.png|529](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213249.png)
- 可以边训练，边看是否是目前的acc的最大值，如果是保存模型。

#### Exercise 13-1 sentiment Analysis on Movie Reviewer
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213413.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213504.png)


### 做古诗
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007213938.png)
训练：每次输出都是下一个字。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007214130.png)
![image.png|521](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007214249.png)
- 希望每次输出不一样；可以采用<mark style="background: FFFF00;">重要性采样</mark>——轮盘赌法
- ![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241007214612.png)
- 这就是隐含层输出不是每次找概率最大的，而是概率越大，被选中的概率越大，反之亦然。



RNN 处理序列问题
CNN 处理 图像空间问题
- 引入attention注意力机制。

图卷积，图神经网络

```ad-note
先看pytorch文档
多读文献
动手实现
```
