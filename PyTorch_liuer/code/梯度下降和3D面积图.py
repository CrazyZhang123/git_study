# import numpy as np
# import matplotlib.pyplot as plt

# x_data = np.array([1.0,2.0,3.0])
# y_data = np.array([2.0,4.0,6.0])

# def forward(x):
#     return x * w + b

# def loss(x,y):
#     y_pred = forward(x)
#     return (y_pred - y)**2

# def sum_loss_fn(x_data,y_data,w,b):
#     y_pred = w * x_data + b
#     return np.sum((y_pred - y_data)**2) / len(x_data)

# # x,y为输入的样本，w,b为模型的参数
# def compute_gradients(x_data, y_data, w, b):
#     # 计算损失函数对w的梯度
#     # 计算损失函数对b的梯度
#     # 计算损失的总和

#     y_pred = w * x_data + b
#     grad_w = np.sum(2 * (y_pred - y_data) * x_data)
#     grad_b = np.sum(2 * (y_pred - y_data))
#     sum_loss = sum_loss_fn(x_data,y_data,w,b)
#     return np.array([grad_w, grad_b]),sum_loss

# # 梯度下降过程计算两个参数w,b的列表
# def gradient_descent(grad,x_data,y_data, learning_rate=0.001, precision=0.0001, max_iters=1000):


#     # 重新初始化w,b 尽量使用np数组
#     init_wb = np.array([4.0, 2.0])
#     gd_w_list = []
#     gd_b_list = []
#     gd_sum_loss_list = []

#     # 优化的目标是sum_loss最小
#     for i in range(max_iters):
#         # 将初始化的w,b赋值给变量w,b
#         w, b = init_wb
#         # 输入就是x_data和y_data
#         grad_val, sum_loss_val = grad(x_data, y_data, w, b)

#         # 比较好的操作：
#         # 1.返回如果是 np.array([grad_w, grad_b]),sum_loss
#         # grad_val, sum_loss_val = grad()解包后，grad_val类型不变

#         # 2.返回如果是 np.array([grad_w, grad_b，sum_loss])
#         # 解包前两个在一块，最后一个单独，就会改变grad_val的类型为list
#         # *grad_val, sum_loss_val = grad()解包后，grad_val类型改变。
        
#         # 当梯度接近0，视为收敛
#         if np.linalg.norm(grad_val,ord=2)<precision:
#             break
#                  # 记录w,b,sum_loss
#         gd_b_list.append(b)
#         gd_w_list.append(w)
#         gd_sum_loss_list.append(sum_loss_val)

#         # 迭代w,b
#         init_wb -= learning_rate*grad_val


#     # 因为要画图，所以最好返回np.array类型
#     return np.array(gd_w_list), np.array(gd_b_list), np.array(gd_sum_loss_list)



# w_start_num = 0
# b_start_num = -2
# w_end_num = 4.1
# b_end_num = 2.1
# buchang = 0.05



    
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# # 显示图片
# # %matplotlib inline
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # 梯度下降线图
# gd_w_list, gd_b_list, gd_sum_loss_list = gradient_descent(compute_gradients,x_data,y_data)
# ax.scatter(gd_w_list, gd_b_list, gd_sum_loss_list, 'red',marker='o',s=10)
# # print('gd_w_list=',gd_w_list[1],',gd_b_list=',gd_b_list[1],',gd_sum_loss_list=',gd_sum_loss_list[1])
# # 标记起点和终点
# ax.scatter(gd_w_list[0], gd_b_list[0], gd_sum_loss_list[0], color='green', s=100, label='Start')
# ax.scatter(gd_w_list[-1], gd_b_list[-1], gd_sum_loss_list[-1], color='blue', s=100, label='End')

# # 3D面积图

# w_range = np.arange(w_start_num, w_end_num, buchang)
# b_range = np.arange(b_start_num, b_end_num, buchang)
# # print('w_range=',w_range,',b_range=',b_range)


# def sum_loss_fn(x_data,y_data,w,b):
#     y_pred = w * x_data + b
#     return np.sum((y_pred - y_data)**2) / len(x_data)
# W, B = np.meshgrid(w_range, b_range) 

# # 向量化：允许 w, b 是数组
# vectorized_loss = np.vectorize(lambda w, b: sum_loss_fn(x_data, y_data, w, b))
# # for w, b in zip(W.flatten(), B.flatten()):
# #     mse = sum_loss_fn(x_data, y_data, w, b)
# #     # print('w=',w,',b=',b,',mse=',mse)
# #     mse_array = np.append(mse_array,mse)
# mse_array = vectorized_loss(W, B)  # 自动对每个 (w, b) 调用 sum_loss_fn
# # mse_array = mse_array.reshape(B.shape)
# # print('mse_array',mse_array)

# # 直接拿结果来meshgrid是不对的，因为w_list和b_lis是一维的，都是41*41个，
# # 所以meshgrid后他们的shape是(41*41,41*41)，而mse_list是(1681,)，所以这就是为什么会报错的原因
# # 错误做法：
# # # W,B = np.meshgrid(np.array(w_list), np.array(b_list))
# # mse_array = np.array(mse_list).reshape(B.shape)  # 自动匹配形状
# # 3D 折线
# # ax.plot(w_list,b_list,mse_list)
# # 3D 散点
# # ax.scatter3D(w_list,b_list,mse_list)
# ax.plot_surface(W,B,mse_array, edgecolor='none',
#                  rstride=1,##retride(row)指定行的跨度
#                     cstride=1,##retride(column)指定列的跨度
#                     cmap='rainbow', # 颜色
#                     alpha=0.3) # 透明度要低一点。
# ax.set_xlabel('w')
# ax.set_ylabel('b')
# ax.set_zlabel('MSE Loss')
# ax.legend()
# plt.show()

import requests
import json

# 1️⃣ 向接口请求数据
response = requests.get("https://api.github.com")

# 2️⃣ 方式一：直接解析（推荐）
data = response.json()      # 内部其实相当于 json.loads(response.text)
print(type(data))           # <class 'dict'>
print(data["current_user_url"])

# 3️⃣ 方式二：自己解析
text = response.text        # 原始字符串
data2 = json.loads(text)    # 手动反序列化
print(data2["current_user_url"])

# 4️⃣ 把 Python 数据再转回 JSON 字符串
json_str = json.dumps(data2, indent=2)
print(type(json_str))       # <class 'str'>
print(json_str[:100])       # 打印前 100 字符