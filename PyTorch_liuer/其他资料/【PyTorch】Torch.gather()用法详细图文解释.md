---
created: 2025-01-25T23:50
updated: 2025-01-25T23:50
---
 

官方文档：[TORCH.GATHER](https://pytorch.org/docs/stable/generated/torch.gather.html "TORCH.GATHER")  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/01338ed5c6a78d87ed8c585b5f4fa5a1.png#pic_center)

使用[gather](https://so.csdn.net/so/search?q=gather&spm=1001.2101.3001.7020 "gather")函数的时候，涉及到3个[Tensor](https://so.csdn.net/so/search?q=Tensor&spm=1001.2101.3001.7020)及它们在这个函数中扮演的角色：

input：输入（糖果区里等待被购买的各类糖果）

index：索引（商场导购员）

output：输出（来买糖果的人最终买走的某类糖果）

另外还涉及到1个参数：

dim：需要在input中使用index来“指路”的维度（选择糖果的范围）

#### 使用要求/注意点1

根据官方文档的描述，**input和index都要有相同数量的维度**(the same number of dimensions)。

> 这里是第1个容易混淆的点：
> 
> input和index只需要维度的数量相同，不需要每个维度的大小也相同。
> 
> 举例：
> 
> input是一个形状（也就是shape）为(2, 4, 5)的Tensor，这个input是3维的，第1维的大小是2，第2维的大小是4，第3维的大小是5；
> 
> ①如果此时有一个index也是一个形状为(2, 4, 5)的Tensor，这个index也是3维的，可以作为input的index使用[gather](https://so.csdn.net/so/search?q=gather&spm=1001.2101.3001.7020)函数；
> 
> ②如果此时有一个index是一个形状为(2, 4)的Tensor，这个index是2维的，与input的维度的数量不同，不能作为input的index使用gather函数；
> 
> ③那如果此时有index形状为(1, 3, 4)，或者(3, 4, 5), 或者(3, 5, 6)呢？它们与input的维度的数量相同了，都是3维的，可以作为input的index去使用gather函数吗？**这些情况我们在后面进一步讨论。**

#### 使用要求/注意点2

根据官方文档的描述，**对于dim之外的维度，index在这些维度上的大小（也就是index.size(d)，d指除了dim指定的维度外的其他维度）都不比input在这些维度上的大小（也就是input.size(d)）大**。也就是index.size(d)<=input.size(d)。

为什么有这条规定？index维度的大小为什么一定不能超过input维度的大小？**这个问题留到后面解答。**

（事实上，不仅index维度的大小不能超过input维度的大小，index中每一个元素的取值，都要小于input被dim指定的那个维度的大小。为什么这么说？**这个问题也留到后面解答**）

> 这里是第2个容易混淆的点：
> 
> 官方文档只说明了除了dim指定的那个维度上，index和input在这些维度上的大小关系，但是：对于dim指定的那个维度，index和input在dim这个维度上的大小关系能怎样？
> 
> 答案是：可以相等，可以index的更大，也可以input的更大。**这个答案留到后面解释。**

知道了这个注意点之后，就可以解答在“注意点1”中提到的问题了：此时若有input是一个形状为(2, 4, 5)的Tensor，如果此时有index形状为(1, 3, 4)，或者(3, 4, 5), 或者(3, 5, 6)呢？它们与input的维度的数量相同了，都是3维的，可以作为input的index去使用gather函数吗？

对于形状为(1, 3, 4)的index，可以。因为不管令dim=0，还是dim=1，还是dim=2，dim以外的维度都能满足index.size(d)<=input.size(d)；

对于形状为(3, 4, 5)的index，要看情况：如果令dim=0，可以，因为其他维度的大小分别为4和5，可以满足index.size(d)<=input.size(d)；如果令dim=1，不可以，因为其他维度的大小分别为3和5，维度大小为3的这个维度不满足index.size(d)<=input.size(d)；如果令dim=2，不可以，因为其他维度的大小分别为3和4，维度大小为3的这个维度不满足index.size(d)<=input.size(d)。

对于形状为(3, 5, 6)的index，不可以。因为不管令dim=0，还是dim=1，还是dim=2，dim以外的维度都不满足index.size(d)<=input.size(d)。

#### 使用要求/注意点3

根据官方文档的描述，output这个Tensor的形状与index的形状相同。也就是说，不仅二者的维度数量一致，每个维度上的大小也会一致。

> 这里是第3个容易混淆的点：
> 
> output的形状取决于index，而不是input。

#### 函数作用理解

这个函数里涉及到的input、index、output这3个Tensor和dim这1个参数之间是什么关系呢？简单来说就是：通过自定义index，从input中选择出想要的那些元素作为output。

在这一小节里解释通过input、index和dim，如何得到output。本文开头处提到它们的形象比喻，此处就使用它们的比喻来解释。

| 输入/参数/输出 | 本义 | 比喻义 |
| --- | --- | --- |
| input（类型：Tensor） | 输入 | 糖果区里等待被购买的各类糖果 |
| index（类型：Tensor） | 索引 | 为来买糖果的人提供一对一服务的商场导购员 |
| dim（类型：int） | 要用index中的值来做新下标的维度 | 选择糖果的范围 |
| output（类型：Tensor） | 输出 | 来买糖果的人最终买走的某类糖果 |

具体运算过程如下图1-图4所示：

图1  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9beffea13439cc8f3fbb608e0e6eb66a.png#pic_center)  
图2  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5b387bec5345381b50d8219b43b6f28b.png#pic_center)  
图3  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b3ff9be3826f8cfc676b29b787caaef0.png#pic_center)  
图4  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9e0724ce77c142952cf6d23da4847cc9.png#pic_center)

验证上述示例的代码：

```cobol
import　torchrandom_seed = 200torch.manual_seed(random_seed)input = torch.randint(0, 100, (2, 3, 4))print("input:")print(input) index = torch.randint(0, 2, (2, 1, 2))print("index:")print(index) output = input.gather(0, index)print("output:")print(output) # 控制台输出input:tensor([[[62, 29, 76, 60],         [82, 27, 88, 11],         [57, 50, 71,  9]],         [[33, 71, 66, 34],         [20, 81,  3, 39],         [15, 33, 19, 89]]])index:tensor([[[0, 1]],         [[1, 0]]])output:tensor([[[62, 71]],         [[33, 29]]])
```

接下来回答前文中的问题。

问题1：为什么index维度的大小一定不能超过input维度的大小？

回答：不妨设想一下如果超过了的话，根据运算流程，会出现什么问题。index维度的下标里面，除了dim指定的那个维度外，index的其他维度的下标都是要对应到input的。如果index在这些维度上的大小超过了input，而input和index都不会广播（官方文档里说了），那就越界了，会造成错误，所以index维度的大小一定不能超过input维度的大小（可以相等）。

问题2：不仅index维度的大小不能超过input维度的大小，index中每一个元素的取值，都要小于input被dim指定的那个维度的大小。为什么这么说？

回答：index中这个元素的值，就是用来作为被dim指定的那个维度的下标的，其余维度的下标就由index的下标里面非dim指定的那些维度的下标来充当，由此形成完整的一个下标，这个下标再拿去input里面找对应的元素，这个元素的值作为output与index这个元素位置对应的那个元素的值。如果index中这个元素的值≥input被dim指定的那个维度的大小，也是会产生越界问题。

问题3：对于dim指定的那个维度，为什么说index和input在dim这个维度上的大小关系可以是任意的（可以相等，可以index的更大，也可以input的更大）？

回答：用前文中图片里的那个比喻来讲就比较清晰了：糖果和顾客不是一一对应的关系。来买糖果的顾客的数量可以比糖果区里拥有的糖果的数量多（这样的话可能出现不同顾客买到相同种类的糖果），可以更少（这样的话有些种类的糖果没被买过），也可以一样。

其他参考：

[Torch.gather()及Tensor.gather()的详细说明【配图，代码】及应用示例](https://blog.csdn.net/zwwcqu/article/details/126296268?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-126296268-blog-109243383.235%5Ev36%5Epc_relevant_default_base3&spm=1001.2101.3001.4242.1&utm_relevant_index=3 "Torch.gather()及Tensor.gather()的详细说明【配图，代码】及应用示例")

[理解 PyTorch 中的 gather 函数](https://cloud.tencent.com/developer/article/1913768 "理解 PyTorch 中的 gather 函数")（这个链接里讲得很详细）

[图解PyTorch中的torch.gather函数](https://zhuanlan.zhihu.com/p/352877584 "图解PyTorch中的torch.gather函数")（这个链接第一次看会有点乱，但是里面提到了强化学习[DQN算法](https://so.csdn.net/so/search?q=DQN%E7%AE%97%E6%B3%95&spm=1001.2101.3001.7020 "DQN算法")中对gather函数的应用，方便理解）

* * *

* * *

* * *

[【PyTorch】Torch.gather()用法详细图文解释-CSDN博客](https://blog.csdn.net/Mocode/article/details/131039356 "【PyTorch】Torch.gather()用法详细图文解释-CSDN博客")

本文转自 <https://blog.csdn.net/u013250861/article/details/139223852?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-139223852-blog-129675818.235^v43^control&spm=1001.2101.3001.4242.2&utm_relevant_index=4>，如有侵权，请联系删除。