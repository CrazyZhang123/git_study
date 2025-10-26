---
created: 2024-09-29T22:43
updated: 2024-09-30T17:26
---

### Pytorch 流程 Pytorch Fashion

#### 1、准备数据

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929225957.png)
x_data,y_data是列向量

#### 2、使用类设计模型 Design model using Class
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929230443.png)
- 必须实现构造函数和forward函数
- 可以重写反向传播的求导函数 backward
- torch.nn.Linear(1,1) w权重和 b 偏置  nn是 neural network
- 必须重写父类函数 forward就是相当于\_\_callable\_\_ 直接按照类名进行调用 



![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929231223.png)


#### 3、构建损失和优化器 Construct loss and optimizer
	using PyTorch API
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929232401.png)
- size_average是否求均值，reduce是否降维
- model.parameters() 是可以递归的找到模型的所有权重
- lr 是学习率，pytorch可以针对模型的不同部分使用不同的学习率
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929232928.png)

#### 4、训练轮 Training cycle
	-前向，后向，更新forward, backward,update
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929233140.png)
- criterion 是计算损失 loss
- 打印loss对象的时候，会自动调用__str__()方法，打印的是标量。
- 所有权重参数清零
- 反向传播
- 更新

#### 5、测试模型 Test model
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929233554.png)

完整代码
![[pytorch学习问题总结#^befafd]]
```python
import torch


x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        #                             w  b
        self.linear = torch.nn.Linear(1,1)
    # 必须重写，这个是callable方法，可以直接调用类来实现forward方法
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

# mse loss
criterion = torch.nn.MSELoss(size_average=False) # 不用求均值
#优化器                 随机梯度下降
# model.parameters() 取出模型的所有参数
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    # 打印loss对象会自动调用__str__方法转成标量
    # print(epoch,loss)

    # 算完损失后权重清0
    optimizer.zero_grad()
    # 开始反向传播
    loss.backward()
    # 更新权重
    optimizer.step()

# test model

# output weight and bias
print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

# Test model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_pred = ',y_test.data)

```

100次迭代之后，和预期的结果还有距离。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930120352.png)


1000次迭代之后，和预期的结果很接近。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240930120301.png)


#### Exercise 1
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929233831.png)

```python
import torch


x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        # 对输入数据应用仿射线性变换 y=xA^T +b。  w  b
        self.linear = torch.nn.Linear(1,1)
    # 必须重写，这个是callable方法，可以直接调用类来实现forward方法
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

# mse loss
criterion = torch.nn.MSELoss(size_average=False) # 不用求均值
#优化器                 随机梯度下降
# SGD随机梯度下降  model.parameters() 取出模型的所有参数
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
# Adagrad算法优化  比较差
# optimizer = torch.optim.Adagrad(model.parameters(),lr=0.01)
# Adam
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01, maximize=False)
# Adamax
# optimizer = torch.optim.Adamax(model.parameters(),lr=0.01)
# ASGD 性能比较好
# optimizer = torch.optim.ASGD(model.parameters(),lr=0.01)
# LBFGS 性能比较好 需要闭包 closure
# optimizer = torch.optim.LBFGS(model.parameters(),lr=0.01)
# RMSprop 性能好
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.01)
# Rprop 性能好
optimizer = torch.optim.Rprop(model.parameters(),lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    # 打印loss对象会自动调用__str__方法转成标量
    # print(epoch,loss)

    # 算完损失后权重清0
    optimizer.zero_grad()
    # 开始反向传播，将权重保留到对应的grad里面，step就是更新学习器的权重。
    loss.backward()
    # 所有的优化器都支持step()方法，更新权重
    optimizer.step()
```



#### Exercise 2

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240929233906.png)

链接   [pytorch_with_examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
