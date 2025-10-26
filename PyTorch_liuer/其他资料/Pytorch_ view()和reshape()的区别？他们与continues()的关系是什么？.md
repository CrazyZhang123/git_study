---
created: 2024-12-16T22:44
updated: 2024-12-16T22:44
---
 

#### 一、概要

##### 1\. 两者相同之处

view()和[reshape](https://so.csdn.net/so/search?q=reshape&spm=1001.2101.3001.7020)()在pytorch中都可以用来重新调整[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)的形状。

##### 2\. 两者不同之处

1). view()产生的tensor总是和原来的tensor共享一份相同的数据，而reshape()在新形状满足一定条件时会共享相同一份数据，否则会复制一份新的数据。

2). 两者对于原始tensor的连续性要求不同。reshape()不管tensor是否是连续的，都能成功改变形状。而view()对于不连续的tensor()，需要新形状[shape](https://so.csdn.net/so/search?q=shape&spm=1001.2101.3001.7020)满足一定条件才能成功改变形状，否则会报错。 transpose, permute 等操作会改变 tensor的连续性，在新形状shape不满足一定的情况下会报错。(注：有的人说view()[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)只用于连续的tensor，事实上这是不准确的，可看下面解释)

##### 3\. 使用指南

如果你想要新的tensor和旧的tensor始终共享一份数据，使用view()

若只是单纯的改变形状，不要求共享数据，reshape()不容易出错

大部分情况下，了解以上即足够了，如果想要知道：

> view(shape)在新形状shape满足哪些情况下才能成功改变形状  
> view和reshape的工作机制

你可以接着看下面的介绍。

#### 二、Tensor的连续性

tensor的连续性是指逻辑上相邻的元素在内存中是否是连续存储的，如是，则称其是连续的，反之则是不连续的。我们可以调用tensor.is\_contiguous()判断该tensor是否是连续的。下面通过举例说明。

当我们构造一个tensor如下：

a = torch.arange(12).reshape(3,4)

**在没有任何transpose的情况下**

逻辑上，它看起来是这样的：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9350f57d983e95bb175e0d17655a39f2.png#pic_center)  
实际上，在电脑内存中，a的存储是下面这样的：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/08054fb75000cc1d2e31513ad840cb47.png#pic_center)  
由于逻辑上相邻的元素, 在内存上也是相邻的，我们称这个tensor是连续的(contiguous tensor)，此时a.is\_contiguous() 为True.

**在调用b=a.T之后**

逻辑上，他看起来是这样的：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d2fa9c185227478fe08e076ced11690.png#pic_center)  
但实际上，在电脑内存中，b的存储仍然是下面这样的：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/08054fb75000cc1d2e31513ad840cb47.png#pic_center)  
为什么存储跟a是一样的呢？这是由转置函数本身的机制决定的，  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eff5ce16ece7d370ad3eb784587a5a47.png)  
也就是说转置之后的tensor b和原来的tensor a共享一份数据，只是访问的方式改变了。由于此时b逻辑上相邻的两个元素在内存中并不连续存储，于是我们称tensor b是不连续的，此时b.is\_contiguous() 为False.

#### 三、 view()和reshape()的工作机制

##### 1\. view()的工作机制

这里的view和数据库中的视图(view)概念上十分类似，其本质就是不会复制一份新的数据，而是与原来的tensor或原来的数据库表共享一份相同的数据。

所以b=a.view(shape)中，tensor b与tensor a共享一份数据，修改b中的数据，a的相应元素也会改变。

上面我们说到view()对于不连续的tensor，需要新形状shape满足一定条件才能成功改变形状。那这里的条件是什么呢？

首先我们需要知道view()改变形状改的是什么，我们知道它与原tensor共享一份数据，所以数据存放顺序并没改变，它改变的是tensor的步幅(stride)，步幅的改变使得新tensor有他自己的访问方式和访问顺序。

如下例所示，a = torch.tensor(\[1,2,3,4\]), 当访问下一个数据的时候，指针每次移动1，我们将这个移动距离定义为stride，而在view之后，这个步幅会变为(2,1)，从而支持在原始内存数据上用新的访问方式b\[i,j\]而不是a\[i\]来访问数据。

```python
>>>a = torch.tensor([1,2,3,4])
>>>a.stride()
(1,)
>>>b = a.view((2,2))
>>>b.stride()
(2,1)
```

这里核心就在于b = a.view(shape)中的新形状shape基于不变的内存数据仍能以**固定**的步幅访问下一个元素，这样才能成功改变形状，否则若没有固定的步幅，我们无法实现元素的顺序访问，这种情况下程序会报错。

举例：比如说在第二部分中经过转置得到的tensor b，  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d2fa9c185227478fe08e076ced11690.png#pic_center)

我们知道它是不连续的，当我们执行b.view((12，))时就会出错

```python
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

这是因为假设我们希望通过b.view((12，))得到逻辑上连续的1维tensor(\[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11\])  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/132b3a92870b6f0b4185ecbaa12914b8.png#pic_center)  
而其内存上的排列顺序仍为：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/08054fb75000cc1d2e31513ad840cb47.png#pic_center)  
此时假设我们逻辑上顺序访问[数组](https://edu.csdn.net/course/detail/40020?utm_source=glcblog&spm=1001.2101.3001.7020)，那么内存指针应当这样移动  
4, 4, -7, 4, 4, -7, 4, 4, 7, 4, 4，然而此时计算机只能支持固定的步幅，无法记忆-7，7，故这样的逻辑数组是不被支持的，无法做到基于不变的内存数据仍以**固定**的步幅访问下一个元素，遂报错。

在[pytorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/#viewargs-tensor)说到一个tensor必须是连续的，才能使用view函数。事实上是不对的，根据最新的英文文档和笔者的实际实践，非连续的tensor也可以使用view函数。比如：

```python
>>>a = torch.arange(12).reshape(3,4)
>>>a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>>b = a.T
>>>b
tensor([[ 0,  4,  8],
       	[ 1,  5,  9],
       	[ 2,  6, 10],
        [ 3,  7, 11]])
>>>c = b.view((2,2,3))
>>>c
tensor([[[ 0,  4,  8],
         [ 1,  5,  9]],
        [[ 2,  6, 10],
         [ 3,  7, 11]]])
>>>b.shape, b.stride()
(torch.Size([4, 3]), (1, 4))
>>>c.shape,c.stride()
(torch.Size([2, 2, 3]), (2, 1, 4))
```

在英文文档中，一个tensor b，能否成功执行b.view(shape), 它归纳出以下条件：

新的shape中的各个维度的值，

1.  要么就是原始tensor的shape中的值(如上面的3)
2.  要么满足下面的条件，剩下的维度 d 0 , d 1 . . . d k d\_0,d\_1...d\_k d0​,d1​...dk​，∀i=0,…,k−1, 相应的步幅满足：  
    s t r i d e \[ i \] = s t r i d e \[ i + 1 \] × s i z e \[ i + 1 \] stride\[i\]=stride\[i+1\]×size\[i+1\] stride\[i\]\=stride\[i+1\]×size\[i+1\]  
    比如上面c的shape，3排除后，剩下\[2,2\]，只需要第一个2对应的stride满足stride\[i\]=stride\[i+1\]×size\[i+1\]即可。

所以非连续的tensor能不能进行view，要看在新的shape的条件下，我们能不能求得新的步幅支持顺序访问目标形状的tensor。在某些情况下非连续的tensor b不能进行view的症结就在于目前的tensor b已经是逻辑顺序和物理顺序不匹配，在目前的逻辑顺序上再做一次逻辑抽象，就可能会得不到一个线性映射。

##### 2\. reshape()的工作机制

reshape()本着尽量节省内存的原件进行形状的调整。

如果新形状满足view函数所要求的条件(即基于不变的内存数据仍能以固定的新步幅访问该数据)，那么这时候reshape()跟view()等价，不会复制一份新的数据。

如果新的形状不满足view函数所要求的条件(即无法求得满足条件的新步幅)，这时候reshape也能工作，这时候它会将原来非连续性的tensor按逻辑顺序copy到新的内存空间(即使得要使用view函数的tensor b其逻辑数据顺序和物理数据顺序一致)，然后再改变tensor b形状。

#### 四、参考资料

\[1\] [pytorch document](https://pytorch.org/docs/stable/torch.html)  
\[2\] [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)  
\[3\] [What is the difference between contiguous and non-contiguous arrays?](https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays/26999092#26999092)  
\[4\] [What’s the difference between reshape and view in pytorch?](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch)  
\[5\] [PyTorch view和reshape的区别](https://blog.csdn.net/HuanCaoO/article/details/104794075/)

本文转自 <https://blog.csdn.net/qq_40765537/article/details/112471341>，如有侵权，请联系删除。