在[pytorch](https://zhida.zhihu.com/search?content_id=166836229&content_type=Article&match_order=1&q=pytorch&zhida_source=entity)中，常见的拼接函数主要是两个，分别是：

1.  [stack()](https://zhida.zhihu.com/search?content_id=166836229&content_type=Article&match_order=1&q=stack%28%29&zhida_source=entity)
2.  `cat()`

实际使用中，这两个函数互相辅助：关于`cat()`参考[torch.cat()](https://link.zhihu.com/?target=https%3A//blog.csdn.net/xinjieyuan/article/details/105208352)，但是本文主要说`stack()`。

**函数的意义**：使用`stack`可以保留两个信息：\[1. 序列\] 和 \[2. 张量矩阵\] 信息，属于【**扩张**再拼接】的函数；可以认为把一个个矩阵按时间序列压紧成一个矩阵。 常出现在[自然语言处理](https://zhida.zhihu.com/search?content_id=166836229&content_type=Article&match_order=1&q=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&zhida_source=entity)（`NLP`）和[图像卷积神经网络](https://zhida.zhihu.com/search?content_id=166836229&content_type=Article&match_order=1&q=%E5%9B%BE%E5%83%8F%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)(`CV`)中。

### 1\. `stack()`

官方解释：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。

浅显说法：把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在**增加新的维度进行堆叠**。

### `outputs = torch.stack(inputs, dim=0) → Tensor`

**参数**

*   inputs : 待连接的张量序列。 注：`python`的序列数据只有`list`和`tuple`。
*   dim : 新的维度， 必须在`0`到`len(outputs)`之间。 注：`len(outputs)`是生成数据的维度大小，也就是`outputs`的维度值。

### 2\. 重点

1.  函数中的输入`inputs`只允许是序列；且序列内部的张量元素，必须`shape`相等

\----举例：`[tensor_1, tensor_2,..]`或者`(tensor_1, tensor_2,..)`，且必须`tensor_1.shape == tensor_2.shape`

1.  `dim`是选择生成的维度，必须满足`0<=dim<len(outputs)`；`len(outputs)`是输出后的`tensor`的维度大小

不懂的看例子，再回过头看就懂了。

### 3\. 例子

1.准备2个`tensor`数据，每个的`shape`都是`[3,3]`

```python
 # 假设是时间步T1
 T1 = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
 # 假设是时间步T2
 T2 = torch.tensor([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])
```

2.测试stack函数

```python
print(torch.stack((T1,T2),dim=0).shape)
print(torch.stack((T1,T2),dim=0))
print(torch.stack((T1,T2),dim=1).shape)
print(torch.stack((T1,T2),dim=1))
print(torch.stack((T1,T2),dim=2).shape)
print(torch.stack((T1,T2),dim=2))
# print(torch.stack((T1,T2),dim=3).shape)
 '选择的dim>len(outputs)，所以报错'
 IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

可以复制代码运行试试：拼接后的`tensor`形状，会根据不同的`dim`发生变化。

```python
torch.Size([2, 3, 3])
tensor([[[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9]],

        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]]])
torch.Size([3, 2, 3])
tensor([[[ 1,  2,  3],
         [10, 20, 30]],

        [[ 4,  5,  6],
         [40, 50, 60]],

        [[ 7,  8,  9],
         [70, 80, 90]]])
torch.Size([3, 3, 2])
tensor([[[ 1, 10],
         [ 2, 20],
         [ 3, 30]],

        [[ 4, 40],
         [ 5, 50],
         [ 6, 60]],

        [[ 7, 70],
         [ 8, 80],
         [ 9, 90]]])
```

| dim | shape |
| --- | --- |

### 4\. 总结

1.  函数作用： 函数`stack()`对**序列数据**内部的张量进行扩维拼接，指定维度由程序员选择、大小是生成后数据的维度区间。
2.  存在意义： 在自然语言处理和卷及神经网络中， **通常为了保留–\[序列(先后)信息\] 和 \[张量的矩阵信息\]** 才会使用`stack`。

本文转自 <https://zhuanlan.zhihu.com/p/354177500>，如有侵权，请联系删除。