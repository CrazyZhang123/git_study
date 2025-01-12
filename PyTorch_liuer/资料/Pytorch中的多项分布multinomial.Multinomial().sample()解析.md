 

> 在读《动手学深度学习 Pytorch》，疑惑于：
> 
> ```python
> fair_probs = torch.ones([6]) / 6
> multinomial.Multinomial(1, fair_probs).sample()
> ```
> 
> 故有此篇文章。
> 
> P.S. 心态崩了，Typora出bug，没保存 😦

```python
# 导入必要的包
import torch
from torch.distributions import multinomial
```

多项分布**Multinomial()**是**torch.distributions.multinomial**中的一个类，接受四个参数(**total\_count**\=**1**, **probs**\=**None**, **logits**\=**None**, **validate\_args**\=**None**)：

**total\_count**接受的是**int**型参数，指的是单次抽样中样本的个数。

**probs**接受的是**Tensor**型参数，指的是各事件发生的概率，也可以传入频数。如果传入的是频数，可以通过**probs**属性查看对应的概率分布。

```python
>> fair_probs = torch.ones([6]) / 6	# 例子来源于《动手学深度学习 Pytorch》
>> multinomial.Multinomial(1, fair_probs).probs

tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])

>> fair_probs = torch.ones([6])	# 可以看到传入频数和概率的结果是一样的
>> torch.ones([6])/6 == multinomial.Multinomial(1, fair_probs).probs

tensor([True, True, True, True, True, True])
```

**logits**接受的是**Tensor**型参数，和**probs**的作用一样，不过其指的是各事件概率的自然对数，同样的也可以传入频数，在传入频数后可以通过**logits**属性查看对应的对数概率分布。

```python
>> fair_probs = torch.ones([6]) / 6	# 例子来源于《动手学深度学习 Pytorch》
>> multinomial.Multinomial(1, fair_probs).logits

tensor([-1.7918, -1.7918, -1.7918, -1.7918, -1.7918, -1.7918])

>> fair_probs = torch.ones([6])	# 可以看到传入频数和概率的结果是一样的
>> torch.log(torch.ones([6]) / 6) = multinomial.Multinomial(1, fair_probs).logits

tensor([True, True, True, True, True, True])
```

**validate\_args**用于指定是否检查参数的合法性，但奇怪的一点是，即便设置它为**False**，依然会对不合理的参数抛出异常。这里我引用ChatGPT回答埋个坑：

> 在 PyTorch 中，`validate_args=True` 表示分布的参数将始终进行完整的检查，以确保它们具有良好的数值特性和合法性。这通常会导致较慢的运行速度和更高的计算成本，但可以提供更加健壮和可靠的代码。
> 
> 相比之下，`validate_args=False` 表示分布的参数只进行基本的检查，以提高代码的[性能](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782?utm_source=glcblog&spm=1001.2101.3001.7020)和效率。这通常会导致更快的运行速度和更低的计算成本，但可能会带来一些风险和不确定性，特别是当参数存在不合法或异常值时。
> 
> 具体而言，当 `validate_args=True` 时，PyTorch 会进行以下检查：
> 
> *   参数是否满足分布的特定要求，例如，参数是否需要是正整数、实数、正定矩阵等；
> *   参数是否存在非法值或异常值，例如，是否存在 NaN、inf、负数等；
> *   参数是否满足特定的约束条件，例如，参数是否需要大于等于某个值、小于等于某个值、总和为1等；
> *   参数是否与其他参数相容，例如，某些参数是否需要具有相同的大小、形状、维度等。
> 
> 当 `validate_args=False` 时，PyTorch 通常只进行基本的检查，例如，参数是否为 None 或空值、是否具有正确的类型、是否具有正确的形状等。这些检查通常不需要耗费太多计算资源，因此可以提高代码的性能和效率。但是，这种方式也可能会忽略一些不合法或异常值，从而引发一些潜在的风险和不确定性。因此，在使用 `validate_args=False` 时需要谨慎，并且最好尽可能避免使用不合法或异常的参数。

需要注意的是，这个回答并不一定会是正确的，在我突发奇想去问ChatGPT Multinomial()参数相关问题时，它答错了三次以上，比如：

> 以下是错误回答的摘选：
> 
> 1.  `probs` 和 `logits` 二选一即可，如果都指定了，则优先使用 `probs`。如果两个参数都没有指定，则默认使用 `probs=torch.tensor([1/K, ..., 1/K])`，其中 `K` 表示离散随机变量的数量。
> 
> **而实际中，如果两个参数都指定或都没有指定，会抛出ValueError: Either `probs` or `logits` must be specified, but not both.**
> 
> 2.  如果传递的 `probs` 张量包含了负数或大于1的数，则会抛出异常。
> 
> **我们从之前的例子中可以看到，传递大于1的频数是完全没有问题的，源码中对于probs参数的约束是：must be non-negative, finite and have a non-zero sum（非负，有限，非零和）**
> 
> 3.  如果将 `validate_args` 设置为 `False`，则不会进行参数合法性检查，这可以提高代码的执行效率。
> 
> **这是错误的，即便你将validate\_args设置为False，也会检测参数的合法性，比如：probs中的某个量为负数**
> 
> 可以看到第三个回答和前面我说“埋个坑的回答”是矛盾的，所以请不要尽信ChatGPT。

**sample**()是类**Multinomial**()中用来抽样的[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782?utm_source=glcblog&spm=1001.2101.3001.7020)，仅接收一个参数 (**sample\_shape=torch.Size()**)，用来指定要抽样的次数，默认情况下仅抽样一次，输出一个形状为(**len(probs),** )的张量，否则，输出为(**sample\_shape, len(probs)**)的张量。

```python
>> fair_probs = torch.ones([6])/6
>> multinomial.Multinomial(1, fair_probs).sample().size()

torch.Size([6])

>> multinomial.Multinomial(1, fair_probs).sample((5, 3)).size()

torch.Size([5, 3, 6])
```

本文转自 <https://blog.csdn.net/weixin_42426841/article/details/129317453?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522495bec3942132beeb1ba2242e67cb000%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=495bec3942132beeb1ba2242e67cb000&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-129317453-null-null.142^v100^pc_search_result_base5&utm_term=multinomial.Multinomial%281%2C%20fair_probs%29.sample%28%29&spm=1018.2226.3001.4187>，如有侵权，请联系删除。