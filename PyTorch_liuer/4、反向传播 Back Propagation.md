---
created: 2024-09-29T19:43
updated: 2024-09-29T22:40
---

##### 计算图  Computational Graph
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929195009.png)

##### 两层神经网络

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929200022.png)

- 可以发现多个线性层之间直接相连，经过简单的变形和合并，就会发现，线性的层，增加之后和不增加没有本质区别，都可以通过线性变换得到，这就是为了引入非线性函数——激活函数的意义。


##### 链式求导法则
和之前学习的普通函数链式求导法则一样。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929200459.png)

反向传播
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929200841.png)


##### 反向传播示例 y = w * x

forward 就是从前到后
back propagation 从后到前
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929201157.png)


![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929201709.png)

##### 两个练习

![Capture_20240929_222212.jpg](https://gitee.com/zhang-junjie123/picture/raw/master/image/Capture_20240929_222212.jpg)






##### 张量 Tensor in PyTorch

- 张量用于保存数据
- 包括数据 data 和 梯度 grad

![image.png|270](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929202156.png)

##### 使用pyTorch实现线性模型


```python
import torch

x_data = [1.0, 2.0,3.0]
y_data = [2.0, 4.0,6.0]

w = torch.Tensor([1.0])
# 设置为True不会计算关于他的梯度，只有需要的时候才会计算
w.requires_grad = True


# 构建计算图
def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)",4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        #损失
        l = loss(x,y)
        # 反向传播——计算所有的梯度，保存到tensor中的grad里面
        l.backward()
        # item()把数据变成标量,如果是l的话会把计算图不断扩充，因为他是张量
        print('\tgrad:',x,y,w.grad.item())

        #纯数值的修改，不用再加入计算图了
        w.data -= 0.01 * w.grad.data 
        
        #权重w的梯度全部清零，不清0就会累加了，这里不需要梯度累加。
        w.grad.data.zero_()
    # 训练轮数
    print('progress:',epoch,l.item())

print('predict (after training)',4,forward(4).item())


```

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929204239.png)




##### Exercise 3

![Capture_20240929_222225.jpg](https://gitee.com/zhang-junjie123/picture/raw/master/image/Capture_20240929_222225.jpg)


##### Exercise 4 使用pytorch实现

```python
import torch

x_data = [1.0, 2.0,3.0]
y_data = [2.0, 4.0,6.0]

w = torch.Tensor([1.0])
# 设置为True不会计算关于他的梯度，只有需要的时候才会计算
w.requires_grad = True

w2 = torch.Tensor([1.0])
# 设置为True不会计算关于他的梯度，只有需要的时候才会计算
w2.requires_grad = True

b = torch.Tensor([1.0])
# 设置为True不会计算关于他的梯度，只有需要的时候才会计算
b.requires_grad = True

# 构建计算图
def forward(x):
    return x * w * x + w2 * x + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)",4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        #损失
        l = loss(x,y)
        # 反向传播——计算所有的梯度，保存到tensor中的grad里面
        l.backward()
        # item()把数据变成标量,如果是l的话会把计算图不断扩充，因为他是张量
        print('\tgrad:',x,y,'w:',w.grad.item(),' w2:',w2.grad.item(),' b:',b.grad.item(),)

        #纯数值的修改，不用再加入计算图了
        w.data -= 0.01 * w.grad.data 
        w2.data -= 0.01 * w2.grad.data 
        b.data -= 0.01 * b.grad.data 

        #权重w的梯度全部清零，不清0就会累加了，这里不需要梯度累加。
        w.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()



    # 训练轮数
    print('progress:',epoch,l.item())

print('predict (after training)',4,forward(4).item())


```
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929224018.png)
