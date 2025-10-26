import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm


'''二维梯度下降法'''
def func_2d_single(x,y):
    '''
    目标函数传入x,y
    :param x,y: 自变量，一维向量
    :return: 因变量，标量
    '''
    z1 = np.exp(-x**2-y**2)
    z2 = np.exp(-(x-1)**2-(y-1)**2)
    z = -(z1-2*z2)*0.5
    return z

def func_2d(xy):
    '''
    目标函数传入xy组成的数组，如[x1,y1]
    :param xy: 自变量，二维向量  （x，y）
    :return: 因变量，标量
    '''
    z1 = np.exp(-xy[0]**2-xy[1]**2)
    z2 = np.exp(-(xy[0]-1)**2-(xy[1]-1)**2)
    z = -(z1-2*z2)*0.5
    return z
def grad_2d(xy):
    '''
    目标函数的梯度
    :param xy: 自变量，二维向量
    :return: 因变量，二维向量  (分别求偏导数，组成数组返回)
    '''
    grad_x = 2*xy[0]*(np.exp(-(xy[0]**2+xy[1]**2)))
    grad_y = 2*xy[1]*(np.exp(-(xy[0]**2+xy[1]**2)))
    return np.array([grad_x,grad_y])
def gradient_descent_2d(grad, cur_xy=np.array([1, 1]), learning_rate=0.001, precision=0.001, max_iters=100000000):
    '''
    二维目标函数的梯度下降法
    :param grad: 目标函数的梯度
    :param cur_xy: 当前的x和y值
    :param learning_rate: 学习率
    :param precision: 收敛精度
    :param max_iters: 最大迭代次数
    :return: 返回极小值
    '''
    print(f"{cur_xy} 作为初始值开始的迭代......")
    x_cur_list = []
    y_cur_list = []
    for i in tqdm(range(max_iters)):
        grad_cur = grad(cur_xy)
        ##创建两个列表，用于接收变化的x，y
        x_cur_list.append(cur_xy[0])
        y_cur_list.append(cur_xy[1])
        if np.linalg.norm(grad_cur,ord=2)<precision:  ##求范数，ord=2 平方和开根
            break    ###当梯度接近于0时，视为收敛
        cur_xy = cur_xy-grad_cur*learning_rate
        x_cur_list.append(cur_xy[0])
        y_cur_list.append(cur_xy[1])
        print('第%s次迭代：x，y = %s'%(i,cur_xy))
    print('极小值 x，y = %s '%cur_xy)
    return (x_cur_list,y_cur_list)

if __name__=="__main__":
    current_xy_list = gradient_descent_2d(grad_2d)
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection='3d')
    a = np.array(current_xy_list[0])
    b = np.array(current_xy_list[1])
    c = func_2d_single(a,b)
    ax.scatter(a,b,c,c='Black',s=10,alpha=1,marker='o')
    x = np.arange(-2,2,0.05)
    y = np.arange(-2,2,0.05)
    ##对x,y数据执行网格化
    x,y = np.meshgrid(x,y)
    z = func_2d_single(x,y)
    ax.plot_surface(x,y,z,
                    rstride=1,##retride(row)指定行的跨度
                    cstride=1,##retride(column)指定列的跨度
                    cmap='rainbow',
                    alpha=0.3
                    )  ##设置颜色映射
    # ax.plot_wireframe(x,y,z,)
    ##设置z轴范围
    ax.set_zlim(-2,2)
    ##设置标题
    plt.title('汽车优化设计之梯度下降--二元函数',fontproperties = 'SimHei',fontsize = 20)
    plt.xlabel('x',fontproperties = 'SimHei',fontsize = 20)
    plt.ylabel('y', fontproperties='SimHei', fontsize=20)
    plt.show()