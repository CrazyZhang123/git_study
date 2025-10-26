
官网：
https://matplotlib.org/stable/users/explain/colors/colormaps.html

> 定义：cmap参数接受一个值（每个值代表一种配色方案），并将该值对应的颜色图分配给当前图窗。

代码示例：

```python
# 要获取所有已注册颜色图的列表，您可以执行以下操作： 
from matplotlib import colormaps 
list(colormaps)

plt.cm.get_cmap('cmap')

#numpy.arange(start, stop, 步长, dtype)
#np.linspace(start, stop, 数列个数num=50, endpoint=True, retstep=False, dtype=None)
color_list = plt.cm.Set3(np.linspace(0, 1, 12))

#在深色背景上绘制一系列线条时，可以在定性色图中选择一组离散的颜色
plt.cm.magma(np.linspace(0, 1, 15))
```

类别
--

1、[Sequential](https://zhida.zhihu.com/search?content_id=113557519&content_type=Article&match_order=1&q=Sequential&zhida_source=entity)：连续化色图。通常使用**单一色调**，逐渐改变亮度和颜色渐渐增加。应该用于表示**有顺序的信息**。可以直观看到数据从低到高的变化

*   以中间值颜色命名（eg：viridis 松石绿）

![](https://pic4.zhimg.com/v2-bd653812de05a25fdde6caa61340de57_1440w.jpg)

*   以色系名称命名，由低饱和度到高饱和度过渡（eg：YlOrRd = yellow-orange-red，其它同理）

![](https://pic3.zhimg.com/v2-b8a17c2fb46ac78ef30362891a28e0d0_1440w.jpg)

![](https://pica.zhimg.com/v2-caa151bf648ad62d05bb867dd6f49168_1440w.jpg)

2、[Diverging](https://zhida.zhihu.com/search?content_id=113557519&content_type=Article&match_order=1&q=Diverging&zhida_source=entity)：发散。改变**两种不同颜色的亮度和饱和度**，这些颜色在中间以不饱和的颜色相遇；当绘制的信息具有**关键中间值（例如地形）或数据偏离零**时，应使用此值。正值和负值分别表示为颜色图的不同颜色。

![](https://pic1.zhimg.com/v2-38e19d08a09b4024a82c70c7feedbfa8_1440w.jpg)

3、离散化色图：[Cyclic](https://zhida.zhihu.com/search?content_id=113557519&content_type=Article&match_order=1&q=Cyclic&zhida_source=entity)：循环。改变两种不同颜色的亮度，**在中间和开始/结束时以不饱和的颜色相遇**。应该用于**在端点处环绕的值**，例如相角，风向或一天中的时间。

![](https://pic3.zhimg.com/v2-d332810f35ada3d941a4468344f66d7c_1440w.jpg)

4、[Miscellaneous](https://zhida.zhihu.com/search?content_id=113557519&content_type=Article&match_order=1&q=Miscellaneous&zhida_source=entity)：杂色。

![](https://pica.zhimg.com/v2-8f54209384870007b86f62ae242520c6_1440w.jpg)

备注：

*   每种颜色图可通过添加后缀**\_r**来**反转**，形象化理解就是把上面的色条水平翻转（eg：若当前使用set3的前三个颜色，则改成set3\_r后就变成使用set3的后三个颜色）

参考：[blog.csdn](https://link.zhihu.com/?target=https%3A//blog.csdn.net/lly1122334/article/details/88535217)

[https://matplotlib.org/examples/color/colormaps\_reference.html](https://link.zhihu.com/?target=https%3A//matplotlib.org/examples/color/colormaps_reference.html)