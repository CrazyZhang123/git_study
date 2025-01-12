---
created: 2024-10-01T22:43
updated: 2024-10-01T22:45
---
 

#### 文章目录

^a32722

*   [torch.sigmoid( )](#torchsigmoid__1) ^200f74
*   [torch.nn.sigmoid( )](#torchnnsigmoid__10)
*   [torch.nn.funtional.sigmoid( )](#torchnnfuntionalsigmoid__20)
*   [总结](#_27)

torch.[sigmoid](https://so.csdn.net/so/search?q=sigmoid&spm=1001.2101.3001.7020)( )
-----------------------------------------------------------------------------------

这是一个方法，拥有Parameters以及Returns。

* * *

![请添加图片描述](https://i-blog.csdnimg.cn/blog_migrate/050111b4b058785e3f8b18cc99a23259.png)

* * *

参考官网的解释，就可以从输出很明显的看到输出就是属于概率\[0,1\]之间了  
所以sigmoid函数就是将输出值映射到\[0,1\]

torch.nn.sigmoid( )
-------------------

* * *

![请添加图片描述](https://i-blog.csdnimg.cn/blog_migrate/b21635c6f9d5d843910955f90631e8de.png)

有一个很明显的class，所以这说明了  
torch.nn.Sigmoid在我们的神经网络中使用时，我们应该将其看作是网络的一层，而不是简单的函数使用。会构建计算图。

torch.nn.funtional.sigmoid( )
-----------------------------

* * *

![请添加图片描述](https://i-blog.csdnimg.cn/blog_migrate/94aadd2d19ad4e0c20ede9c8b03de821.png)

事实上，torch.nn.functional从这个包名就能看出来，这个包里的都是函数。同样的，按照官网的文档的内容，我们也可以判断出torch.nn.funtional.sigmoid是一个方法，可以直接在我们的神经网络的forward中使用，并不需要在init的时候初始化。也就是说torch.nn.functional.sigmoid和torch.sigmoid没有什么区别，同理，本文对于其他的激活函数一样适用。

总结
--

**具体情况具体分析，一般使用torch.sigmoid( )**。

 


本文转自 <https://blog.csdn.net/weixin_44673253/article/details/125310670>，如有侵权，请联系删除。