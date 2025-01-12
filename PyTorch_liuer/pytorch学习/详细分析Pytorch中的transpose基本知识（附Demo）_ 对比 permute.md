---
created: 2024-11-17T19:09
updated: 2024-11-17T19:09
---
 

#### 目录

*   [前言](#_4)
*   [1\. 基本知识](#1__6)
*   [2\. Demo](#2_Demo_23)

前言
--

原先的[permute](https://so.csdn.net/so/search?q=permute&spm=1001.2101.3001.7020)推荐阅读：[详细分析Pytorch中的permute基本知识（附Demo）](https://blog.csdn.net/weixin_47872288/article/details/143186849)

1\. 基本知识
--------

[transpose](https://so.csdn.net/so/search?q=transpose&spm=1001.2101.3001.7020) 是 PyTorch 中用于交换张量维度的函数，特别是用于二维张量（矩阵）的转置操作，常用于线性代数运算、深度学习模型的输入和输出处理等

**基本知识如下**

*   **功能**：交换张量的两个维度
*   **输入**：一个张量和两个要交换的维度的索引
*   **输出**：具有新维度顺序的张量

原理分析如下：  
transpose 的核心原理是通过交换指定维度的方式改变张量的形状  
例如，对于一个二维张量 (m, n)，调用 transpose(0, 1) 会返回一个形状为 (n, m) 的新张量，其元素顺序经过了调整

*   **高维张量**： 对于高维张量，transpose 只会影响指定的两个维度，而其他维度保持不变
*   **内存视图**：与 permute 类似，transpose 返回的是原始张量的一个视图，不会进行数据复制

2\. Demo
--------

**示例 1: 基本用法**

```python
import torch

# 创建一个 3x4 的矩阵
matrix = torch.randn(3, 4)
print("原始矩阵形状:", matrix.shape)

# 使用 transpose 交换维度
# 将矩阵的维度从 (3, 4) 变为 (4, 3)
transposed_matrix = matrix.transpose(0, 1)
print("转置后矩阵形状:", transposed_matrix.shape)
```

截图如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fdf70535579c48fca6906cb1287543ab.png)

**示例 2: 高维张量的转置**

```python
import torch

# 创建一个 2x3x4 的张量
tensor = torch.randn(2, 3, 4)
print("原始张量形状:", tensor.shape)

# 使用 transpose 交换第二和第三维
# 将张量的维度从 (2, 3, 4) 变为 (2, 4, 3)
transposed_tensor = tensor.transpose(1, 2)
print("转置后张量形状:", transposed_tensor.shape)
```

截图如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f58c1ae9e4034c73ac925f878b8e3b5f.png)

**示例 3: 在深度学习中的应用**

```python
import torch

# 创建一个假设的批量数据 (批量, 高度, 宽度, 通道)
batch_tensor = torch.randn(5, 256, 256, 3)
print("原始批量形状:", batch_tensor.shape)

# 将通道和宽度维度交换
# 适用于某些模型的输入
batch_transposed = batch_tensor.transpose(2, 3)
print("转置后批量形状:", batch_transposed.shape)
```

截图如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d916efc6bd947b6918f3f4d540a2693.png)

基本的注意事项如下：

*   **只支持交换两个维度**： transpose 只能同时交换两个维度，而无法一次性处理多个维度
*   **数据不复制**：返回的是原始张量的视图，因此内存开销较小
*   **维度索引**：确保指定的维度索引在张量的维度范围内，否则会引发错误

 

文章知识点与官方知识档案匹配，可进一步学习相关知识

[Python入门技能树](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)[人工智能](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)[深度学习](https://edu.csdn.net/skill/python/python-3-246?utm_source=csdn_ai_skill_tree_blog)464317 人正在系统学习中

本文转自 <https://blog.csdn.net/weixin_47872288/article/details/143187574>，如有侵权，请联系删除。