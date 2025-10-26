 

#### 文章目录

*   [super().\_\_ init\_\_ ()有什么用？](#super___init____2)
*   *   [1、分别理解super()和 \_\_ init \_\_()](#1super____init____11)
    *   *   [1.1、super()](#11super_12)
        *   [1.2、\_\_ init \_\_()](#12___init____19)
        *   [1.3、super(). \_\_ init \_\_()](#13super____init____27)
        *   *   [1.3.1、关于“覆盖”的疑问](#131_40)
    *   [2、super() 在 python2、3中的区别](#2super__python23_86)
    *   [3、关于继承顺序](#3_122)
    *   [4、从多个实例中对比super（python3）](#4superpython3_170)
    *   *   [4.1、实例](#41_172)
        *   [4.2、运行结果与对比](#42_182)
        *   [4.3、完整代码](#43_185)

super().\_\_ init\_\_ ()有什么用？
-----------------------------

```python
super().__init__() 、 super(B,self).__init__()
```

python里的super().\_\_init\_\_()有什么作用？很多同学没有弄清楚。

`super()用来调用父类(基类)的方法，__init__()是类的构造方法，`  
`super().__init__() 就是调用父类的init方法， 同样可以使用super()去调用父类的其他方法。`

### 1、分别理解super()和 \_\_ init \_\_()

#### 1.1、super()

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7ce36ba20e756b6724d5b1ff60b72ed7.png)

```python
需要注意的是python2、3的super写法稍有不同。
```

#### 1.2、\_\_ init \_\_()

\_\_init\_\_() 是python中的构造函数，在创建对象的时"自动调用"。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4b1faf893198af56c5e7f5497a33b197.png)

```python
定义类时可以不写init方法，系统会默认创建，
你也可以写一个，让你的类在创建时完成一些“动作”。
```

#### 1.3、super(). \_\_ init \_\_()

如果子类B和父类A，都写了init方法，  
那么A的init方法就会被B覆盖。想调用A的init方法需要用super去调用。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eb00a9790c3b078d1b91ef0d90a7ad0e.png)  
当然，在B内部，除了用super调用父类的方法，也可以用父类名调用，例：

```python
class B(A):
    def __init__(self):
        A.__init__(self)
        print("B init")
```

##### 1.3.1、关于“覆盖”的疑问

有人可能会误解“覆盖”的意思，认为“覆盖”了就是没有，为什么还能通过super调用？  
覆盖了并不是没有了，A的方法终都还在，但需要在B内部用super调用。

```
例：
A里写了一个方法hi(), B继承自A, B里也写了一个方法hi()。
B的对象在外部调用hi(), 就只能调用B里面写的这个hi()。
想通过B调用A的hi(),只能在B内部用super().hi()调用。
```

```python
class A:
    def hi(self):
        print("A hi")

class B(A):
    def hello(self):
        print("B hello")
        
b = B()
b.hi()       # B里没有写hi(),这里调用的是继承自A的hi()

------------------------------------------------------------------
class A:
    def hi(self):
        print("A hi")

class B(A):
    def hi(self):
        print("B hi")
        
b = B()
b.hi()    # 这里调用的就是B自己的hi()
------------------------------------------------------------------
class A:
    def hi(self):
        print("A hi")

class B(A):
    def hi(self):
        super().hi()         # 通过super调用父类A的hi()
        print("B hi")
        
b = B()
b.hi()    # 这里调用的就是B里面的hi()
```

### 2、super() 在 python2、3中的区别

Python3.x 和 Python2.x 的一个区别: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx :

> 例：  
> python3 直接写成 ： super().\_\_init\_\_()  
> python2 必须写成 ：super(本类名,self).\_\_init\_\_()

Python3.x 实例：

```python
class A:
     def add(self, x):
         y = x+1
         print(y)
class B(A):
    def add(self, x):
        super().add(x)
b = B()
b.add(2)  # 3
```

Python2.x 实例：

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
class A(object):   # Python2.x 记得继承 object
    def add(self, x):
         y = x+1
         print(y)
class B(A):
    def add(self, x):
        super(B, self).add(x)
b = B()
b.add(2)  # 3
```

### 3、关于继承顺序

最底层：先写一个父类A

```python
class A:
    def __init__(self):
        print('A')
```

第二层：让 B、C、D 继承自A

```python
class B(A):
    def __init__(self):
        print('B')
        super().__init__()

class C(A):
    def __init__(self):
        print('C')
        super().__init__()

class D(A):
    def __init__(self):
        print('D')
        super().__init__()
```

第三层： E、F、G 继承

```python
class E(B, C):
    def __init__(self):
        print('E')
        super().__init__()

class F(C, D):
    def __init__(self):
        print('F')
        super().__init__()

class G(E, F):
    def __init__(self):
        print('G')
        super().__init__()
```

看看G的继承顺序  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9ed6a29dab02a36c3447d216e7dedb8e.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ec76ef57c43ac1dd2c2cecbb2bbe6b4c.png)  
我们发现G继承自E, F是并列的，初始化的时候不会先把E初始化完毕才初始化F。

### 4、从多个实例中对比super（python3）

下面是三种不同的继承、调用，对比他们的区别，搞清楚super().\_\_init\_\_()的用途。

#### 4.1、实例

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b4038f859a3feab10b9241bdd78c77d6.png)

| 子类名称 | 继承内容 |
| --- | --- |
| Puple | 继承所有 |
| Puple\_Init | 继承，但覆盖了init方法 |
| Puple\_Super | 继承，但覆盖了init方法，并在init里面添加了super().\_\_init\_\_() |

#### 4.2、运行结果与对比

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3049abbd3193970362dfec61807c1ea8.png)

#### 4.3、完整代码

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/212b747c8c845e3399d1d95bae716e65.png)

本文转自 <https://blog.csdn.net/a__int__/article/details/104600972>，如有侵权，请联系删除。