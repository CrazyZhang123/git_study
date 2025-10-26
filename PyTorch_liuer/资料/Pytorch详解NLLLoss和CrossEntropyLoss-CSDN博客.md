 

pytorch的官方文档写的也太简陋了吧…害我看了这么久…

### NLLLoss

在图片单标签分类时，输入m张图片，输出一个m\*N的[Tensor](https://so.csdn.net/so/search?q=Tensor&spm=1001.2101.3001.7020)，其中N是分类个数。比如输入3张图片，分三类，最后的输出是一个3\*3的Tensor，举个例子：  
![在这里插入图片描述](https://gitee.com/zhang-junjie123/picture/raw/master/image/8a3755fa30895854d08fb5e2f5e71d76.png)  
第123行分别是第123张图片的结果，假设第123列分别是猫、狗和猪的分类得分。  
可以看出模型认为第123张都更可能是猫。  
然后对每一行使用[Softmax](https://so.csdn.net/so/search?q=Softmax&spm=1001.2101.3001.7020)，这样可以得到每张图片的概率分布。  
![在这里插入图片描述](https://gitee.com/zhang-junjie123/picture/raw/master/image/7ffc02374d73d2d33de417625700fca8.png)  
这里dim的意思是计算Softmax的维度，这里设置dim=1，可以看到每一行的加和为1。比如第一行0.6600+0.0570+0.2830=1。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c9e57051dbcdb78c68e91b419c5d61cf.png#pic_center)

如果设置dim=0，就是一列的和为1。比如第一列0.2212+0.3050+0.4738=1。  
我们这里一张图片是一行，所以dim应该设置为1。  
然后对Softmax的结果取自然对数：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/352fe49e57ee06212b6a39f5f5c29621.png#pic_center)

Softmax后的数值都在0~1之间，所以ln之后值域是负无穷到0。  
**NLLLoss的结果**就是==把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值。  ==
假设我们现在Target是\[0,2,1\]（第一张图片是猫，第二张是猪，第三张是狗）。第一行取第0个元素，第二行取第2个，第三行取第1个，去掉负号，结果是：\[0.4155,1.0945,1.5285\]。再求个均值，结果是：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a5d76465c7b2a98d4c1f4201f31e8476.png#pic_center)

下面使用NLLLoss函数验证一下：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1ce4a7c5149ebb1b560ae8eb7f079975.png#pic_center)  
嘻嘻，果然是1.0128！

### CrossEntropyLoss

CrossEntropyLoss就是把以上Softmax–Log–NLLLoss合并成一步，我们用刚刚随机出来的input直接验证一下结果是不是1.0128：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f3d2eb4e5f86216ff0c58767b3e906fe.png#pic_center)  
真的是1.0128哈哈哈哈！我也太厉害了吧！

如果你也觉得我很厉害，请打赏以鼓励我做的更好，非常感谢！

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/41aa622505a269f70ed1f141a4eda242.jpeg#pic_center)

本文转自 <https://blog.csdn.net/qq_22210253/article/details/85229988>，如有侵权，请联系删除。