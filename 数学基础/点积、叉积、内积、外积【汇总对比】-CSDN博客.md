 

**目录**

[点积(dot product)](#t0)

[代数定义](#t1)

[几何定义](#t2)

[与内积的关系](#t3)

[叉积(corss product)](#t4)

[定义](#t5)

[几何意义](#t6)

[内积(inner product)](#t7)

[定义](#t8)

[例子](#t9)

[外积(outer product)](#t10)

[定义](#t11)

[与欧几里得内积对比](#t12)

[张量的外积](#t13)

* * *

点积(dot product)
---------------

又叫标量积、数量积([scalar](https://so.csdn.net/so/search?q=scalar&spm=1001.2101.3001.7020) product)。它是两个数字序列的相应条目的乘积之和。在欧几里得几何中，两个向量的笛卡尔坐标的点积被广泛使用。它通常被称为欧几里得空间的**内积**（或很少称为**投影积**），是[内积](https://so.csdn.net/so/search?q=%E5%86%85%E7%A7%AF&spm=1001.2101.3001.7020)的一种特殊情况，尽管它不是可以在欧几里得空间上定义的唯一内积。

在代数上，[点积](https://so.csdn.net/so/search?q=%E7%82%B9%E7%A7%AF&spm=1001.2101.3001.7020)是两个数字序列的相应条目的乘积之和。在几何上，它是两个向量的欧几里得大小和它们之间夹角的余弦的乘积。这两个定义在使用笛卡尔坐标时是等价的。在现代几何中，欧几里得空间通常使用向量空间来定义。在这种情况下，点积用于定义长度（向量的长度是向量本身的点积的平方根）和角度（两个向量夹角的余弦等于它们的点积与它们长度的乘积的商）。

### 代数定义

![](https://i-blog.csdnimg.cn/blog_migrate/25b253eb0abd662426e21669b741ee02.png)

### 几何定义

![](https://i-blog.csdnimg.cn/blog_migrate/02b96618460bc8fc054e537932fc5a42.png)

### 与内积的关系

![](https://i-blog.csdnimg.cn/blog_migrate/7d87bfec0c381a1cfcdb3dcb80035516.png)

* * *

叉积(corss product)
-----------------

或**向量积****(**vector product )（有时是**有向面积积**，以强调其几何意义）是在三维有向欧几里得向量空间，并用符号x表示. 给定两个线性独立的向量 **a**和**b**，叉积**a** × **b**（读作“a cross b”）是一个垂直于**a**和**b**的向量，因此垂直于包含它们的平面。

### 定义

![](https://i-blog.csdnimg.cn/blog_migrate/d4421096a67d195a31d372ce7033c703.png)

下图为使用 Sarrus 规则得到**a**和**b****的叉积**

![](https://i-blog.csdnimg.cn/blog_migrate/116269ec76ca54d7cbda58ef87f75f7d.png)

叉积也可以表示为形式行列式：

![](https://i-blog.csdnimg.cn/blog_migrate/9f7978d98b4e21491bae1f442ad6b2df.png)

这个行列式可以使用Sarrus 规则或辅因子扩展来计算。使用 Sarrus 规则，它扩展为

![](https://i-blog.csdnimg.cn/blog_migrate/a038b266ced9d881c0026aa936b8eb78.png)

 沿用第一行使用辅因子改为，它展开为

![](https://i-blog.csdnimg.cn/blog_migrate/73c2ce783a401ba696236e235221a2f0.png)

它直接给出了结果的分量。

### 几何意义

![](https://i-blog.csdnimg.cn/blog_migrate/6686c48d1c967ab0e2f3a2e9bf3beb7d.png)

* * *

内积(inner product)
-----------------

空间中两个向量的内积是一个标量，通常用尖括号表示，例如。内积空间推广了欧氏向量空间，在欧氏向量空间中内积是笛卡尔坐标的点积或标量积。

无穷维内积空间广泛用于泛函分析。复数域上的内积空间有时称为**酉空间**。具有内积的向量空间概念的第一次使用是由于朱塞佩·皮亚诺，1898 年。

![](https://i-blog.csdnimg.cn/blog_migrate/19bde8bab27d7b7e84ae6db91a8f2355.png)

使用内积定义的两个向量之间角度的几何解释( |_x_| 与 |_y_| 为范数在二维三维空间的表现)

### 定义

![](https://i-blog.csdnimg.cn/blog_migrate/daa05abbd6fc7692620d7a20900ac79b.png)

### 例子

![](https://i-blog.csdnimg.cn/blog_migrate/8a12316fae495b33fda2013bcca2113d.png)

* * *

外积(outer product)
-----------------

在线性代数中，两个坐标向量的**外积**是一个矩阵。如果这两个向量的维度是_n_和_m_，那么它们的外积是一个_n_ × _m_矩阵。更一般地说，给定两个张量（多维数字数组），它们的外积是张量。张量的外积也称为张量积，可用于定义张量代数。

### 定义

![](https://i-blog.csdnimg.cn/blog_migrate/14fb97a50b11429089a30ebc6550f18c.png)

### 与欧几里得内积对比

![](https://i-blog.csdnimg.cn/blog_migrate/82f094f7f32a83ff5f6fe20c8c49b825.png)

### 张量的外积

![](https://i-blog.csdnimg.cn/blog_migrate/148977614045a2b31c0854f3d116316e.png)

本文转自 <https://blog.csdn.net/Dust_Evc/article/details/127502272>，如有侵权，请联系删除。