1\. 引言
------

`pprint`的英文全称`Data pretty printer`，顾名思义就是让显示结果更加直观漂亮。

`print()`和`pprint()`都是python的打印模块，功能基本一样，唯一的区别就是`pprint()`模块打印出来的数据结构更加完整，每行为一个数据结构，更加方便阅读打印输出结果。特别是对于特别长的数据打印，`print()`输出结果都在一行，不方便查看，而`pprint()`采用分行打印输出，所以对于数据结构比较复杂、数据长度较长的数据，适合采用`pprint()`打印方式。

在介绍完上述理论知识后，我们不妨来举个栗子吧！

2\. 使用背景
--------

我们来看一个打印[嵌套字典](https://zhida.zhihu.com/search?content_id=201091646&content_type=Article&match_order=1&q=%E5%B5%8C%E5%A5%97%E5%AD%97%E5%85%B8&zhida_source=entity)的例子，如下所示：

```python
d = {
    "apple": {"juice":4, "pie":5},
    "orange": {"juice":6, "cake":7},
    "pear": {"cake":8, "pie":9}
}
```

如果使用默认的`print`来进行打印，得到输出如下：

```python
{'apple': {'juice': 4, 'pie': 5}, 'orange': {'juice': 6, 'cake': 7}, 'pear': {'cake': 8, 'pie': 9}}
```

上述输出都堆在一行，显得很混乱，缺少可读性。为了让输出显得有条理，我曾经写过一个for循环来打印如下内容：

```python
for k,v in d.items():
    print(k, "->", v)
```

此时的输出如下：

```python
apple -> {'juice': 4, 'pie': 5}
orange -> {'juice': 6, 'cake': 7}
pear -> {'cake': 8, 'pie': 9}
```

上述代码很容易让人理解，但我必须浪费宝贵的时间来输入for循环。上述常见就是Python的`pprint`发挥作用的地方。

3\. pprint 大法好
--------------

有了上述的简短介绍，我们这里直接使用`pprint`来打印上述字典，样例代码如下：

```python
from pprint import pprint
pprint(d)
```

输出如下：

```python
{'apple': {'juice': 4, 'pie': 5},
 'orange': {'cake': 7, 'juice': 6},
 'pear': {'cake': 8, 'pie': 9}}
```

需要注意的是，`pprint`以人类可读的格式很好地格式化了嵌套字典，而不需要像前面的示例中那样来编写for循环实现同样的功能。

4\. 设定输出宽度
----------

在了解了`pprint`的入门示例后，我们来看看该函数的其他高级用法。这里我们不妨以一个三层嵌套字典为例来进行讲解，示例如下：

```python
d = {
    "apple": {
        "juice": {1:2, 3:4, 5:6}, 
        "pie": {1:3, 2:4, 5:7}, 
    },
    "orange": {
        "juice": {1:5, 2:3, 5:6},
        "cake": {5:4, 3:2, 6:5},
    },
    
    "pear": {
        "cake": {1:6, 6:1, 7:8},
        "pie": {3:5, 5:3, 8:7},
    }
}
```

其实，在`pprint`函数中有一个参数`width`可以控制每行输出的宽度，直接使用`pprint`输出如下：

```python
pprint(d)
# output
{'apple': {'juice': {1: 2, 3: 4, 5: 6}, 'pie': {1: 3, 2: 4, 5: 7}},
 'orange': {'cake': {3: 2, 5: 4, 6: 5}, 'juice': {1: 5, 2: 3, 5:6}},
 'pear': {'cake': {1: 6, 6: 1, 7: 8}, 'pie': {3: 5, 5: 3, 8: 7}}}
```

将宽度设置为50，此时输出如下：

```python
pprint(d, width=50)
# output:
{'apple': {'juice': {1: 2, 3: 4, 5: 6},
           'pie': {1: 3, 2: 4, 5: 7}},
 'orange': {'cake': {3: 2, 5: 4, 6: 5},
            'juice': {1: 5, 2: 3, 5: 6}},
 'pear': {'cake': {1: 6, 6: 1, 7: 8},
          'pie': {3: 5, 5: 3, 8: 7}}}
```

将宽度设置为30，此时输出如下：

```python
pprint(d, width=30)
# output

{'apple': {'juice': {1: 2,
                     3: 4,
                     5: 6},
           'pie': {1: 3,
                   2: 4,
                   5: 7}},
 'orange': {'cake': {3: 2,
                     5: 4,
                     6: 5},
            'juice': {1: 5,
                      2: 3,
                      5: 6}},
 'pear': {'cake': {1: 6,
                   6: 1,
                   7: 8},
          'pie': {3: 5,
                  5: 3,
                  8: 7}}}
```

5\. 设定输出缩进
----------

我们以下面这个字典为例来讲解缩进参数`indent` 的作用：

```python
d = {
    "apple": {"juice":4, "pie":5},
    "orange": {"juice":6, "cake":7},
    "pear": {"cake":8, "pie":9}
}
```

默认不设置缩进的输出如下：

```python
pprint(d)
# output
{'apple': {'juice': 4, 'pie': 5},
 'orange': {'cake': 7, 'juice': 6},
 'pear': {'cake': 8, 'pie': 9}}
```

将缩进设置为4时的输出如下：

```python
pprint(d, indent=4)
# output
{   'apple': {'juice': 4, 'pie': 5},
    'orange': {'cake': 7, 'juice': 6},
    'pear': {'cake': 8, 'pie': 9}}
```

将缩进设置为8时的输出如下：

```python
pprint(d, indent=8)
# output
{       'apple': {'juice': 4, 'pie': 5},
        'orange': {'cake': 7, 'juice': 6},
        'pear': {'cake': 8, 'pie': 9}}
```

6\. 总结
------

本文重点介绍了Python中的`pprint`模块，使用该模块可以提升我们减少我们编写代码的行数同时增加我们复杂数据结构输出的可读性。

  

您学废了吗？

本文转自 <https://zhuanlan.zhihu.com/p/508317313>，如有侵权，请联系删除。