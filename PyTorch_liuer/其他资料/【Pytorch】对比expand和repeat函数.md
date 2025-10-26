---
created: 2025-01-22T23:42
updated: 2025-01-22T23:42
---
 

`expand`和`repeat`函数是pytorch中常用于进行张量数据复制和维度扩展的函数，但其工作机制差别很大，本文对这两个函数进行对比。

##### 1\. expand

```python
tensor.expand(*sizes)
```

`expand`函数用于将张量中**单数维**的数据扩展到指定的size。

首先解释下什么叫**单数维**（[singleton](https://so.csdn.net/so/search?q=singleton&spm=1001.2101.3001.7020) dimensions），张量在某个维度上的size为1，则称为**单数维**。比如`zeros(2,3,4)`不存在单数维，而`zeros(2,1,4)`在第二个维度（即维度1）上为**单数维**。`expand`函数仅仅能作用于这些**单数维**的维度上。

参数`*sizes`用于逐个指定各个维度扩展后的大小（也可以理解为拓展的次数），对于不需要或者无法（即非单数维）进行扩展的维度，对应位置可写上原始维度大小或直接写作-1。

`expand`函数可能导致原始张量的升维，其作用在张量前面的维度上，因此通过`expand`函数可将张量数据复制多份（可理解为沿着第一个batch的维度上）。

另一个值得注意的点是：`expand`函数并不会重新分配内存，返回结果仅仅是原始张量上的一个视图。

下面为几个简单的示例：

```python
import torch
a = tensor([1, 0, 2])
b = a.expand(2, -1)   # 第一个维度为升维，第二个维度保持原阳
# b为   tensor([[1, 0, 2],  [1, 0, 2]])

a = torch.tensor([[1], [0], [2]])
b = a.expand(-1, 2)   # 保持第一个维度，第二个维度只有一个元素，可扩展
# b为  tensor([[1, 1],
#              [0, 0],
#              [2, 2]])
```

##### 2\. expand\_as

`expand_as`函数可视为`expand`的另一种表达，其`size`通过函数传递的目标张量的`size`来定义。

```python
import torch
a = torch.tensor([1, 0, 2])
b = torch.zeros(2, 3)
c = a.expand_as(b)  # a照着b的维度大小进行拓展
# c为 tensor([[1, 0, 2],
#        [1, 0, 2]])
```

##### 3\. [repeat](https://so.csdn.net/so/search?q=repeat&spm=1001.2101.3001.7020)

前文提及`expand`仅能作用于**单数维**，那对于非单数维的拓展，那就需要借助于`repeat`函数了。

```python
tensor.repeat(*sizes)
```

参数`*sizes`指定了原始张量在各维度上复制的次数。整个原始张量作为一个整体进行复制，这与`Numpy`中的`repeat`函数截然不同，而更接近于`tile`函数的效果。

与`expand`不同，`repeat`函数会真正的复制数据并存放于内存中。

下面是一个简单的例子：

```python
import torch
a = torch.tensor([1, 0, 2])
b = a.repeat(3,2)  # 在轴0上复制3份，在轴1上复制2份
# b为 tensor([[1, 0, 2, 1, 0, 2],
#        [1, 0, 2, 1, 0, 2],
#        [1, 0, 2, 1, 0, 2]])
```

###### 4\. repeat\_intertile

Pytorch中，与`Numpy`的`repeat`函数相类似的函数为`torch.repeat_interleave`：

```python
torch.repeat_interleave(input, repeats, dim=None)
```

参数`input`为原始张量，`repeats`为指定轴上的复制次数，而`dim`为复制的操作轴，若取值为`None`则默认将所有元素进行复制，并会返回一个flatten之后一维张量。

与`repeat`将整个原始张量作为整体不同，`repeat_interleave`操作是逐元素的。

下面是一个简单的例子：

```python
a = torch.tensor([[1], [0], [2]])
b = torch.repeat_interleave(a, repeats=3)   # 结果flatten
# b为tensor([1, 1, 1, 0, 0, 0, 2, 2, 2])

c = torch.repeat_interleave(a, repeats=3, dim=1)  # 沿着axis=1逐元素复制
#　ｃ为tensor([[1, 1, 1],
#        [0, 0, 0],
#        [2, 2, 2]])
```

本文转自 <https://blog.csdn.net/guofei_fly/article/details/104467138>，如有侵权，请联系删除。