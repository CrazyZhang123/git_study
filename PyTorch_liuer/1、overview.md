---
created: 2024-09-25T19:21
updated: 2024-09-26T14:58
---

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925215024.png)

四种学习系统

基于规则的学习 rule-based systems
- 比如求积分的原函数
	- 给出对应的base原则，比如常见的函数的积分原函数，积分的法则。

机器学习
- 最火的是SVM family，受限之后转向深度学习。

感知学习 representation learning
- 提取features 
	- 维度太高不利于训练，所以要降维。

深度学习
- 提取 simple features(图片的像素点，语音的波形)
- 抽象特征


注意：
- 端到端的学习过程
	- feature是单独训练的
	- 提取feature也是单独的
	- 映射函数（学习器）单独的
-  基于规则的系统是根据人工去设计的，而表示学习是从数据训练得到算法的过程。

传统机器学习的策略
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925215639.png)

新挑战：
①手动提取特征的局限性
②SVM处理大数据集不太好
③越来越多的营员需要处理无结构化数据，比如语音，图像，视频等等。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925221245.png)


神经网络历史

- 最早来源于神经科学，现在是数学和工程学
- Cambrian Period 寒武纪

感知机的出现
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925221959.png)
- 核心是 <mark style="background: #D2B3FFA6;">反向传播</mark> Back Propagation
	- 核心是计算图
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925222544.png)

不同神经网络的架构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925222804.png)


good news:
- 深度学习并不难
- 有很多现成的深度学习框架
	- Starting <mark style="background: #ADCCFFA6;">from scratch</mark> (从头开始)do not be required
	- Enabled efficient and convenient use of GPU
	- 大量的神经网络构件
- 流行的深度学习框架
	- Theano (University of Montreal) / TensorFlow (Google) --- TFboys
	- kears已经合并到TF里面了
	- Caffe (UC Berkeley)/ Caffe 2 (Facebook)
	- Torch (NYU & Facebook)/ PyTorch (Facebook) 这两个已经合并了

what is PyTorch:

PyTorch is a python package that provides two o high-level features:
- Tensor computation (like numpy) with strong GPU acceleration. 
	- 张量计算用强大的GPU加速
- Deep Neural Networks built on a tape-based autodiff system

### 2024最新Pytorch安装教程（Anaconda+GPU）
https://blog.csdn.net/m0_65788436/article/details/139641734



$$
$$
### CUDA安装
$$
$$
https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925232915.png)

自定义路径，没有装VS
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925233003.png)

安装成功，查看CUDA环境变量
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925233039.png)


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240925232822.png)


### [CUDNN](https://so.csdn.net/so/search?q=CUDNN&spm=1001.2101.3001.7020)安装

- 创建账户后安装
- 解压到随便的一个地方，然后修改文件夹的权限为ZJJ完全控制
- 复制所有的文件夹到cuda下面



安装命令就用pip安装
网站 https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- 安装速度慢可以用梯子加速。

测试
```
import torch
 
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装成功
```

安装成功
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926145849.png)
