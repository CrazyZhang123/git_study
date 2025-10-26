# 1、meshgrid理解

[[numpy.meshgrid的理解以及3D曲面图绘制（梯度下降法实现过程）]]

meshgrid后的W, B 都是(len(b_range), len(w_range)) 形状的

#### (1)Z值通过双层for循环得到，需要进行reshape注意事项
详细见
[[2、Linear Model#方法一]]
```python
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)
# print('w_range=',w_range,',b_range=',b_range)

W, B = np.meshgrid(w_range, b_range) 
print(W.shape, B.shape)
# [length(b_range), length(w_range)]
mse_array = np.array(mse_list)
# 外层是 w，内层是 b
# 所以 mse_list 的顺序是：w 变得慢，b 变得快
# 因为我们上面迭代是外层w,内层b，所以Reshape只能先(len(w), len(b))
# 而 reshape((len(w), len(b))) 会生成一个 (w_steps, b_steps) 的矩阵
# 但 np.meshgrid(w_range, b_range) 默认是 indexing='xy'，返回 (len(b_range), len(w_range)) 形状
# 也就是说，meshgrid的第一个维度是b，第二个维度是w 点，空间中的点应该是(b,w,mse)这种维度
# 而实际我们需要的是(w,b,mse)这种维度，所以我们需要转置一下

mse_array = mse_array.reshape(W.shape).T
# print('mse_array',mse_array)
```

# 2、数据最好使用numpy数组



# 3、numpy范数

np.linalg.norm(A, ord=2) 计算2范数

详见文档：[[numpy范数]]

# 4、排序

## (1)np.argsort()

### 功能说明

`numpy.argsort()`函数返回一个数组，该数组包含原数组元素按升序排列时的索引位置。换句话说，它不会直接对数组进行排序，而是返回排序后元素在原数组中的位置索引。

### 语法

```
numpy.argsort(a, axis=-1, kind=None, order=None)
```

### 参数说明

- `a`: 要排序的数组
- `axis`: 排序的轴向，-1表示最后一个轴（默认值）
- `kind`: 排序算法，如'quicksort'、'mergesort'等
- `order`: 当数组元素是字段时，指定排序的字段顺序

```python
import numpy as np

# 一维数组示例
arr = np.array([3, 1, 2])
indices = np.argsort(arr)
print(indices)  # 输出: [1 2 0]
# 这表示最小元素在索引1位置，次小元素在索引2位置，最大元素在索引0位置

# 使用索引获取排序后的数组
sorted_arr = arr[indices]
print(sorted_arr)  # 输出: [1 2 3]

# 二维数组示例
arr2d = np.array([[3, 1, 2], [6, 5, 4]])
indices2d = np.argsort(arr2d, axis=1)
print(indices2d)
# 输出: [[1 2 0]
#       [2 1 0]]
```

# 5、重组np向量
### np.concatenate

`numpy.concatenate()` 函数用于将多个数组沿着指定的轴合并成一个数组。这个函数在处理大型数据集或需要合并来自不同来源的数据时特别有用。与其他数组连接函数（如 `numpy.vstack()` 和 `numpy.hstack()`，它们只能垂直或水平连接数组）不同，`numpy.concatenate()` 通过允许你指定沿哪个轴连接数组，提供了更大的灵活性

```python
 >>> a = np.array([[1, 2], [3, 4]])  
 >>> b = np.array([[5, 6]])  
 >>> np.concatenate((a, b), axis=0)  
 array([[1, 2],  
        [3, 4],  
        [5, 6]])  
 >>> np.concatenate((a, b.T), axis=1)  
 array([[1, 2, 5],  
        [3, 4, 6]])  
 >>> np.concatenate((a, b), axis=None)  
 array([1, 2, 3, 4, 5, 6])
```

### 向量堆叠

### np.vstack

numpy.vstack() 是 NumPy 中的一个函数，用于垂直（按行）堆叠数组。它接收一系列数组序列作为输入，并通过沿垂直轴（轴 0）堆叠它们，返回一个单一的数组。

示例：使用 numpy.vstack() 垂直堆叠一维数组

```python
import numpy as geek

a = geek.array([1, 2, 3])
print("第一个输入数组：\n", a)

b = geek.array([4, 5, 6])
print("第二个输入数组：\n", b)

res = geek.vstack((a, b))
print("垂直堆叠后的输出数组：\n", res)
```

输出：
```
第一个输入数组：
 [1 2 3]
第二个输入数组：
 [4 5 6]
垂直堆叠后的输出数组：
  [[1 2 3]
 [4 5 6]]
```

两个一维数组 a 和 b 通过 np.vstack() 进行垂直堆叠，组合成一个二维数组，其中每个输入数组构成一行。

语法：
numpy.vstack(tup)

参数：
tup：[ndarray 序列] 包含要堆叠的数组的元组。这些数组在除第一个轴之外的所有轴上必须具有相同的形状。

返回值：[堆叠的 ndarray] 输入数组堆叠后的数组。

使用 numpy.vstack() 垂直堆叠二维数组

以下代码展示了如何使用 numpy.vstack() 垂直堆叠两个二维数组，得到一个组合的二维数组。

```python
import numpy as geek

a = geek.array([[1, 2, 3], [-1, -2, -3]])
print("第一个输入数组：\n", a)

b = geek.array([[4, 5, 6], [-4, -5, -6]])
print("第二个输入数组：\n", b)

res = geek.vstack((a, b))
print("堆叠后的输出数组：\n", res)
```

输出：
```
第一个输入数组：
 [[ 1  2  3]
 [-1 -2 -3]]
第二个输入数组：
 [[ 4  5  6]
 [-4 -5 -6]]
堆叠后的输出数组：
  [[ 1  2  3]
 [-1 -2 -3]
 [ 4  5  6]
 [-4 -5 -6]]
```

<<<<<<< HEAD
两个二维数组 a 和 b 被垂直堆叠，创建了一个新的二维数组，其中每个原始数组成为结果数组中的一组行。

### np.stack()
np.stack 会沿着一个新的轴（默认是第 0 轴）将这些数组堆叠起来，形成一个新的、统一的二维 NumPy 矩阵。
    # 例如，如果 M 是 [vec1, vec2, vec3]，np.stack(M) 会生成一个矩阵，其第一行是 vec1，第二行是 vec2，第三行是 vec3。
=======
两个二维数组 a 和 b 被垂直堆叠，创建了一个新的二维数组，其中每个原始数组成为结果数组中的一组行。
>>>>>>> 4978d2d (2025-10-26 update)
