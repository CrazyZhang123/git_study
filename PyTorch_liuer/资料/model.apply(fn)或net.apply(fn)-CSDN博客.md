---
created: 2024-12-17T21:56
updated: 2024-12-17T21:55
---
 

首先，我们知道pytorch的任何网络`net`，都是`torch.nn.Module`的子类,都算是`module`，也就是模块。  
pytorch中的`model.apply(fn)`会[递归](https://edu.csdn.net/course/detail/40020?utm_source=glcblog&spm=1001.2101.3001.7020)地将函数`fn`应用到父模块的每个子模块`submodule`，也包括`model`这个父模块自身。  
比如下面的网络例子中。`net`这个模块有两个子模块，分别为`Linear(2,4)`和`Linear(4,8)`。[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)首先对`Linear(2,4)`和`Linear(4,8)`两个子模块调用`init_weights`函数，即`print(m)`打印`Linear(2,4)`和`Linear(4,8)`两个子模块。然后再对`net`模块进行同样的操作。如此完成递归地调用。从而完成`model.apply(fn)`或者`net.apply(fn)`。  
个人水平有限，不足处望指正。  
详情可参考  
[pytorch官网文档](https://pytorch.org/docs/stable/nn.html?highlight=module#torch.nn.Module).

```
import torch.nn as nn
@torch.no_grad()
def init_weights(m):
    print(m)
    
net = nn.Sequential(nn.Linear(2,4), nn.Linear(4, 8))
print(net)
print('isinstance torch.nn.Module',isinstance(net,torch.nn.Module))
print(' ')
net.apply(init_weights)
```

输出

```
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): Linear(in_features=4, out_features=8, bias=True)
)
isinstance torch.nn.Module True
Linear(in_features=2, out_features=4, bias=True)
Linear(in_features=4, out_features=8, bias=True)
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): Linear(in_features=4, out_features=8, bias=True)
)
```

如果我们想对某些特定的子模块`submodule`做一些针对性的处理，该怎么做呢。我们可以加入`type(m) == nn.Linear:`这类判断语句，从而对子模块m进行处理。如下，读者可以细细体会一下。

```
import torch.nn as nn
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2,4), nn.Linear(4, 8))
print(net)
print('isinstance torch.nn.Module',isinstance(net,torch.nn.Module))
print(' ')
net.apply(init_weights)
```

可以先打印网络整体看看。调用`apply`函数后，先逐一打印子模块m，然后对子模块进行判断，打印`Linear`这类子模块`m`的权重。

```
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): Linear(in_features=4, out_features=8, bias=True)
)
isinstance torch.nn.Module True
 
Linear(in_features=2, out_features=4, bias=True)
Parameter containing:
tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], requires_grad=True)
Linear(in_features=4, out_features=8, bias=True)
Parameter containing:
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], requires_grad=True)
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): Linear(in_features=4, out_features=8, bias=True)
)

```

本文转自 <https://blog.csdn.net/qq_37025073/article/details/106739513>，如有侵权，请联系删除。