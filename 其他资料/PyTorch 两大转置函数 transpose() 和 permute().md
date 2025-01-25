---
created: 2025-01-24T19:55
updated: 2025-01-26T00:13
---
 

### 关心差别的可以直接看\[3.不同点\]和\[4.连续性问题\]

##### 前言

在`pytorch`中转置用的函数就只有这两个

1.  `transpose()`
2.  `permute()`

注意只有`transpose()`有后缀格式：`transpose_()`：后缀函数的作用是简化如下代码：

```python
x = x.transpose(0,1)
等价于
x.transpose_()
# 相当于x = x + 1 简化为 x+=1
```

这两个函数功能相同，有一些在内存占用上的细微区别，但是不影响编程、可以忽略

##### 1\. 官方文档

###### `transpose()`

```python
torch.transpose(input, dim0, dim1, out=None) → Tensor
```

函数返回输入矩阵`input`的转置。交换维度`dim0`和`dim1`

参数:

*   input (Tensor) – 输入张量，必填
*   dim0 (int) – 转置的第一维，默认0，可选
*   dim1 (int) – 转置的第二维，默认1，可选

###### `permute()`

```python
permute(dims) → Tensor
```

将[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)的维度换位。

参数：

*   **dims** (int…\*)-换位顺序，必填

##### 2\. 相同点

1.  都是返回转置后矩阵。
2.  都可以操作高纬矩阵，`permute`在高维的功能性更强。

##### 3.不同点

先定义我们后面用的数据如下

```python
# 创造二维数据x，dim=0时候2，dim=1时候3
x = torch.randn(2,3)       'x.shape  →  [2,3]'
# 创造三维数据y，dim=0时候2，dim=1时候3，dim=2时候4
y = torch.randn(2,3,4)   'y.shape  →  [2,3,4]'
```

1.  合法性不同

`torch.transpose(x)`合法， `x.transpose()`合法。  
`tensor.permute(x)`**不**合法，`x.permute()`合法。

参考第二点的举例

2.  操作`dim`不同：

`transpose()`只能一次操作两个维度；`permute()`可以一次操作多维数据，且必须传入所有维度数，因为`permute()`的参数是`int*`。

**举例**

```python
# 对于transpose
x.transpose(0,1)     'shape→[3,2] '  
x.transpose(1,0)     'shape→[3,2] '  
y.transpose(0,1)     'shape→[3,2,4]' 
y.transpose(0,2,1)  'error，操作不了多维'

# 对于permute()
x.permute(0,1)     'shape→[2,3]'
x.permute(1,0)     'shape→[3,2], 注意返回的shape不同于x.transpose(1,0) '
y.permute(0,1)     "error 没有传入所有维度数"
y.permute(1,0,2)  'shape→[3,2,4]'
```

3.  `transpose()`中的`dim`没有数的大小区分；`permute()`中的`dim`有数的大小区分

举例，注意后面的`shape`：

```python
# 对于transpose，不区分dim大小
x1 = x.transpose(0,1)   'shape→[3,2] '  
x2 = x.transpose(1,0)   '也变换了，shape→[3,2] '  
print(torch.equal(x1,x2))
' True ，value和shape都一样'

# 对于permute()
x1 = x.permute(0,1)     '不同transpose，shape→[2,3] '  
x2 = x.permute(1,0)     'shape→[3,2] '  
print(torch.equal(x1,x2))
'False，和transpose不同'

y1 = y.permute(0,1,2)     '保持不变，shape→[2,3,4] '  
y2 = y.permute(1,0,2)     'shape→[3,2,4] '  
y3 = y.permute(1,2,0)     'shape→[3,4,2] '  
```

##### 4.关于连续contiguous()

经常有人用`view()`函数改变通过转置后的数据结构，导致报错  
`RuntimeError: invalid argument 2: view size is not compatible with input tensor's....`

这是因为tensor经过转置后**数据的内存地址不连续**导致的,也就是`tensor . is_contiguous()==False`  
这时候`reshape()`可以改变该tensor结构，但是`view()`不可以，具体不同可以看[view和reshape的区别](https://blog.csdn.net/xinjieyuan/article/details/107966712)  
例子如下：

```python
x = torch.rand(3,4)
x = x.transpose(0,1)
print(x.is_contiguous()) # 是否连续
'False'
# 再view会发现报错
x.view(3,4)
'''报错
RuntimeError: invalid argument 2: view size is not compatible with input tensor's....
'''

# 但是下面这样是不会报错。
x = x.contiguous()
x.view(3,4)
```

我们再看看`reshape()`

```python
x = torch.rand(3,4)
x = x.permute(1,0) # 等价x = x.transpose(0,1)
x.reshape(3,4)
'''这就不报错了
说明x.reshape(3,4) 这个操作
等于x = x.contiguous().view()
尽管如此，但是torch文档中还是不推荐使用reshape
理由是除非为了获取完全不同但是数据相同的克隆体
'''
```

调用`contiguous()`时，会强制拷贝一份`tensor`，让它的布局和从头创建的一毛一样。  
（这一段看文字你肯定不理解，你也可以不用理解，有空我会画图补上）

只需要记住了，每次在使用`view()`之前，该`tensor`只要使用了`transpose()`和`permute()`这两个函数一定要`contiguous()`.

##### 5.总结

最重要的区别应该是上面的第三点和第四个。

另外，简单的数据用`transpose()`就可以了，但是个人觉得不够直观，指向性弱了点；复杂维度的可以用`permute()`，对于维度的改变，一般更加精准。

本文转自 <https://blog.csdn.net/xinjieyuan/article/details/105232802>，如有侵权，请联系删除。