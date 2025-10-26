---
created: 2024-09-26T17:16
updated: 2024-09-27T11:40
---

### 1、深度学习的一般流程：
- dataSet
- Model  模型
- Train   训练
- infer    推断

### 2、提出问题
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172042.png)

#### 2.1机器学习流程：
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172131.png)
- 输入dataset,训练模型，后续只需要输入数据就可以推断出结果。
- 有==明确的标签的训练和测试集==  属于 supervised Learning 监督学习
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926172439.png)
- kaggle是机器学习数据分析领域的一个竞赛，会给参赛者一个数据集，等参赛者提交一份模型代码，使用未给的额外数据去测试模型。


<mark style="background: #ADCCFFA6;">模型的泛化能力</mark>：对于训练好的模型，在没有参与训练的数据集中，也能展示出很好的性能。

#### 2.2 模型设计 Model design

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926173926.png)
#### 线性回归
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926174334.png)
先找一个随机值w,然后来评估目前的模型。


#### 计算损失 Compute Loss

单个样本的损失->求所有样本的平均损失->调整w来降低样本的平均损失。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926174924.png)

样本的损失  
$$
loss = (\hat y - y)^2 = (x * \omega - y)^2
$$
数据集的损失  cost fuction  
均方误差mean squared error
$$
cost = \frac{1}{N}\sum_{n=1}^{N} (\hat y_n - y_n)^2
$$

可以通过枚举的方法来绘制关于w的损失曲线
![image.png|700](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926181406.png)
- 但是后续画的图 横坐标是训练轮数(epoch)  纵坐标是损失
- 可视化工具 visdom


练习
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20240926183113.png)
### 方法一
==涉及meshgrid和z值的reshape和转置！！！==

```核心
# 外层是 w，内层是 b
# 所以 mse_list 的顺序是：w 变得慢，b 变得快
# 因为我们上面迭代是外层w,内层b，所以Reshape只能先(len(w), len(b))
# 而 reshape((len(w), len(b))) 会生成一个 (w_steps, b_steps) 的矩阵
# 但 np.meshgrid(w_range, b_range) 默认是 indexing='xy'，返回 (len(b_range), len(w_range)) 形状
# 也就是说，meshgrid的第一个维度是b，第二个维度是w 点，空间中的点应该是(b,w,mse)这种维度
# 而实际我们需要的是(w,b,mse)这种维度，所以我们需要转置一下
```

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0,2.0,3.0])
y_data = np.array([2.0,4.0,6.0])

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2


# x,y为输入的样本，w,b为模型的参数
def compute_gradients(x_data, y_data, w, b):
    # 计算损失函数对w的梯度
    # 计算损失函数对b的梯度
    # 计算损失的总和

    y_pred = w * x_data + b
    grad_w = np.sum(2 * (y_pred - y_data) * x_data)
    grad_b = np.sum(2 * (y_pred - y_data))
    sum_loss = np.sum((y_pred - y_data)**2)
    return np.array([grad_w, grad_b]),sum_loss

# 梯度下降过程计算两个参数w,b的列表
def gradient_descent(grad,x_data,y_data, learning_rate=0.01, precision=0.001, max_iters=1000):

    # 重新初始化w,b 尽量使用np数组
    init_wb = np.array([4.0, 2.0])
    gd_w_list = []
    gd_b_list = []
    gd_sum_loss_list = []

    # 优化的目标是sum_loss最小
    for i in range(max_iters):
        # 将初始化的w,b赋值给变量w,b
        w, b = init_wb
        # 输入就是x_data和y_data
        grad_val, sum_loss_val = grad(x_data, y_data, w, b)
        # 比较好的操作：
        # 1.返回如果是 np.array([grad_w, grad_b]),sum_loss
        # grad_val, sum_loss_val = grad()解包后，grad_val类型不变

        # 2.返回如果是 np.array([grad_w, grad_b，sum_loss])
        # 解包前两个在一块，最后一个单独，就会改变grad_val的类型为list
        # *grad_val, sum_loss_val = grad()解包后，grad_val类型改变。
        
        # 当梯度接近0，视为收敛
        if np.linalg.norm(grad_val,ord=2)<precision:
            break

        # 迭代w,b
        init_wb -= learning_rate*grad_val

        # 记录w,b,sum_loss
        gd_b_list.append(b)
        gd_w_list.append(w)
        gd_sum_loss_list.append(sum_loss_val/len(x_data))
    # 因为要画图，所以最好返回np.array类型
    return np.array(gd_w_list), np.array(gd_b_list), np.array(gd_sum_loss_list)
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2,2.1,0.1):
        # print('w=',w,',b=',b)
        l_sum = 0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val,y_val)
            l_sum += loss_val
            # print(f'\t x_val:{x_val},y_val:{y_val},y_pred_val:{y_pred_val},loss_val:{loss_val}')
            
        # print(f'MSE=',l_sum/len(x_data))
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/len(x_data))


    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# 显示图片
# %matplotlib inline
fig = plt.figure()
ax = plt.axes(projection='3d')

# 梯度下降线图
gd_w_list, gd_b_list, gd_sum_loss_list = gradient_descent(compute_gradients,x_data,y_data)
ax.scatter(gd_w_list, gd_b_list, gd_sum_loss_list, 'red',marker='o',s=10)
# print('gd_w_list=',gd_w_list,',gd_b_list=',gd_b_list,',gd_sum_loss_list=',gd_sum_loss_list)

# 3D面积图

w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)
# print('w_range=',w_range,',b_range=',b_range)

W, B = np.meshgrid(w_range, b_range) 
print(W.shape, B.shape)
# [length(b_range), length(w_range)]
mse_array = np.array(mse_list)
# 外层是 w，内层是 b
# 所以 mse_list 的顺序是：w 变得慢，b 变得快
# 因为我们上面迭代是外层w,内层b，所以Reshape只能先(len(w), len(b))
# 而 reshape((len(w), len(b))) 会生成一个 (w_steps, b_steps) 的矩阵
# 但 np.meshgrid(w_range, b_range) 默认是 indexing='xy'，返回 (len(b_range), len(w_range)) 形状
# 也就是说，meshgrid的第一个维度是b，第二个维度是w 点，空间中的点应该是(b,w,mse)这种维度
# 而实际我们需要的是(w,b,mse)这种维度，所以我们需要转置一下

mse_array = mse_array.reshape(W.shape).T
# print('mse_array',mse_array)

# 直接拿结果来meshgrid是不对的，因为w_list和b_lis是一维的，都是41*41个，
# 所以meshgrid后他们的shape是(41*41,41*41)，而mse_list是(1681,)，所以这就是为什么会报错的原因
# 错误做法：
# # W,B = np.meshgrid(np.array(w_list), np.array(b_list))
# mse_array = np.array(mse_list).reshape(B.shape)  # 自动匹配形状
# 3D 折线
# ax.plot(w_list,b_list,mse_list)
# 3D 散点
# ax.scatter3D(w_list,b_list,mse_list)
ax.plot_surface(W,B,mse_array, edgecolor='none',
                 rstride=1,##retride(row)指定行的跨度
                    cstride=1,##retride(column)指定列的跨度
                    cmap='rainbow', # 颜色
                    alpha=0.3) # 透明度要低一点。
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE Loss')
plt.show()

```

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250812222241990.png)

### 方法二

使用==向量化的方法==计算sum_loss。

``` python
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0,2.0,3.0])
y_data = np.array([2.0,4.0,6.0])

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

def sum_loss_fn(x_data,y_data,w,b):
    y_pred = w * x_data + b
    return np.sum((y_pred - y_data)**2) / len(x_data)

# x,y为输入的样本，w,b为模型的参数
def compute_gradients(x_data, y_data, w, b):
    # 计算损失函数对w的梯度
    # 计算损失函数对b的梯度
    # 计算损失的总和

    y_pred = w * x_data + b
    grad_w = np.sum(2 * (y_pred - y_data) * x_data)
    grad_b = np.sum(2 * (y_pred - y_data))
    sum_loss = sum_loss_fn(x_data,y_data,w,b)
    return np.array([grad_w, grad_b]),sum_loss

# 梯度下降过程计算两个参数w,b的列表
def gradient_descent(grad,x_data,y_data, learning_rate=0.001, precision=0.0001, max_iters=1000):


    # 重新初始化w,b 尽量使用np数组
    init_wb = np.array([4.0, 2.0])
    gd_w_list = []
    gd_b_list = []
    gd_sum_loss_list = []

    # 优化的目标是sum_loss最小
    for i in range(max_iters):
        # 将初始化的w,b赋值给变量w,b
        w, b = init_wb
        # 输入就是x_data和y_data
        grad_val, sum_loss_val = grad(x_data, y_data, w, b)

        # 比较好的操作：
        # 1.返回如果是 np.array([grad_w, grad_b]),sum_loss
        # grad_val, sum_loss_val = grad()解包后，grad_val类型不变

        # 2.返回如果是 np.array([grad_w, grad_b，sum_loss])
        # 解包前两个在一块，最后一个单独，就会改变grad_val的类型为list
        # *grad_val, sum_loss_val = grad()解包后，grad_val类型改变。
        
        # 当梯度接近0，视为收敛
        if np.linalg.norm(grad_val,ord=2)<precision:
            break
                 # 记录w,b,sum_loss
        gd_b_list.append(b)
        gd_w_list.append(w)
        gd_sum_loss_list.append(sum_loss_val)

        # 迭代w,b
        init_wb -= learning_rate*grad_val


    # 因为要画图，所以最好返回np.array类型
    return np.array(gd_w_list), np.array(gd_b_list), np.array(gd_sum_loss_list)



w_start_num = 0
b_start_num = -2
w_end_num = 4.1
b_end_num = 2.1
buchang = 0.05

    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# 显示图片
# %matplotlib inline
fig = plt.figure()
ax = plt.axes(projection='3d')

# 梯度下降线图
gd_w_list, gd_b_list, gd_sum_loss_list = gradient_descent(compute_gradients,x_data,y_data)
ax.scatter(gd_w_list, gd_b_list, gd_sum_loss_list, 'red',marker='o',s=10)
# print('gd_w_list=',gd_w_list[1],',gd_b_list=',gd_b_list[1],',gd_sum_loss_list=',gd_sum_loss_list[1])
# 标记起点和终点
ax.scatter(gd_w_list[0], gd_b_list[0], gd_sum_loss_list[0], color='green', s=100, label='Start')
ax.scatter(gd_w_list[-1], gd_b_list[-1], gd_sum_loss_list[-1], color='blue', s=100, label='End')

# 3D面积图

w_range = np.arange(w_start_num, w_end_num, buchang)
b_range = np.arange(b_start_num, b_end_num, buchang)
# print('w_range=',w_range,',b_range=',b_range)


def sum_loss_fn(x_data,y_data,w,b):
    y_pred = w * x_data + b
    return np.sum((y_pred - y_data)**2) / len(x_data)
W, B = np.meshgrid(w_range, b_range) 

# 向量化：允许 w, b 是数组
vectorized_loss = np.vectorize(lambda w, b: sum_loss_fn(x_data, y_data, w, b))
# for w, b in zip(W.flatten(), B.flatten()):
#     mse = sum_loss_fn(x_data, y_data, w, b)
#     # print('w=',w,',b=',b,',mse=',mse)
#     mse_array = np.append(mse_array,mse)
mse_array = vectorized_loss(W, B)  # 自动对每个 (w, b) 调用 sum_loss_fn
# mse_array = mse_array.reshape(B.shape)
# print('mse_array',mse_array)

# 直接拿结果来meshgrid是不对的，因为w_list和b_lis是一维的，都是41*41个，
# 所以meshgrid后他们的shape是(41*41,41*41)，而mse_list是(1681,)，所以这就是为什么会报错的原因
# 错误做法：
# # W,B = np.meshgrid(np.array(w_list), np.array(b_list))
# mse_array = np.array(mse_list).reshape(B.shape)  # 自动匹配形状
# 3D 折线
# ax.plot(w_list,b_list,mse_list)
# 3D 散点
# ax.scatter3D(w_list,b_list,mse_list)
ax.plot_surface(W,B,mse_array, edgecolor='none',
                 rstride=1,##retride(row)指定行的跨度
                    cstride=1,##retride(column)指定列的跨度
                    cmap='rainbow', # 颜色
                    alpha=0.3) # 透明度要低一点。
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE Loss')
ax.legend()
plt.show()


```

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250812222600659.png)
