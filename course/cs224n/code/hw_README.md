https://github.com/PKUFlyingPig/CS224n/tree/master/hw



**列向量 $ u_o $（$ n \times 1 $）和单位矩阵 $ \boldsymbol{I}_n $（$ n \times n $）直接做矩阵乘法在维度上是不兼容的**（矩阵乘法要求前一个矩阵的列数等于后一个矩阵的行数，这里 $ 1 \neq n $）。

之前的表述存在不严谨，正确的理解应该是：**“对 $ u_o v_c^T $ 关于 $ v_c $ 求偏导的结果，其每一列的结构与 $ u_o $ 一致，整体可通过‘将 $ u_o $ 复制到每一列’来构造矩阵”**，而非直接的矩阵乘法。


### 修正后的准确推导
我们从**元素级偏导**出发，重新梳理：

设 $ u_o = \begin{bmatrix} u_{o1} \\ u_{o2} \\ \vdots \\ u_{on} \end{bmatrix} $（$ n \times 1 $），$ v_c = \begin{bmatrix} v_{c1} \\ v_{c2} \\ \vdots \\ v_{cn} \end{bmatrix} $（$ n \times 1 $），则 $ u_o v_c^T $ 是 $ n \times n $ 矩阵，元素为 $ (u_o v_c^T)_{j,k} = u_{oj} v_{ck} $。

对 $ v_c $ 的第 $ i $ 个元素 $ v_{ci} $ 求偏导，得到的偏导矩阵 $ \frac{\partial (u_o v_c^T)}{\partial v_{ci}} $ 的元素为：
- 若 $ k = i $，则 $ \frac{\partial (u_{oj} v_{ck})}{\partial v_{ci}} = u_{oj} $；
- 若 $ k \neq i $，则 $ \frac{\partial (u_{oj} v_{ck})}{\partial v_{ci}} = 0 $。

因此，偏导矩阵 $ \frac{\partial (u_o v_c^T)}{\partial v_c} $ 可表示为：
$$
\frac{\partial (u_o v_c^T)}{\partial v_c} = \begin{bmatrix} 
u_{o1} & 0 & \dots & 0 \\ 
u_{o2} & 0 & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
u_{on} & 0 & \dots & 0 
\end{bmatrix} \quad \text{（第1列）} \\
\oplus \begin{bmatrix} 
0 & u_{o1} & \dots & 0 \\ 
0 & u_{o2} & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & u_{on} & \dots & 0 
\end{bmatrix} \quad \text{（第2列）} \\
\oplus \dots \\
\oplus \begin{bmatrix} 
0 & 0 & \dots & u_{o1} \\ 
0 & 0 & \dots & u_{o2} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \dots & u_{on} 
\end{bmatrix} \quad \text{（第$ n $列）}
$$

即**每一列都是 $ u_o $，其余位置为0**，其结构等价于：
$$
\frac{\partial (u_o v_c^T)}{\partial v_c} = \begin{bmatrix} 
u_{o1} & u_{o1} & \dots & u_{o1} \\ 
u_{o2} & u_{o2} & \dots & u_{o2} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
u_{on} & u_{on} & \dots & u_{on} 
\end{bmatrix}
$$

这种结构可以理解为“将列向量 $ u_o $ 横向复制 $ n $ 次”，而非严格的矩阵乘法，之前的表述是为了方便理解而做的简化，特此修正。