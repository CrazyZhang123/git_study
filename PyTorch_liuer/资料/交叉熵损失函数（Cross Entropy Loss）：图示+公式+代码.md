## （一）交叉熵损失函数（Cross Entropy Loss）：图示+公式+代码

> 交叉熵损失函数（Cross Entropy Loss）：图示+公式+代码

本文通过融合图表示例、公式和代码的多元化方式，阐释交叉熵损失函数，希望对大家从理论到实践，全面理解交叉熵损失函数有一定帮助。

本文内容

1.  [概率视角解释（二分类的交叉熵）](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E6%A6%82%E7%8E%87%E8%A7%86%E8%A7%92%E8%A7%A3%E9%87%8A%E4%BA%8C%E5%88%86%E7%B1%BB%E7%9A%84%E4%BA%A4%E5%8F%89%E7%86%B5)
    1.  [例 1](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E4%BE%8B-1)
    2.  [公式](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E5%85%AC%E5%BC%8F)
    3.  [代码示例](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E4%BB%A3%E7%A0%81%E7%A4%BA%E4%BE%8B)
2.  [概率视角解释（多分类的交叉熵）](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E6%A6%82%E7%8E%87%E8%A7%86%E8%A7%92%E8%A7%A3%E9%87%8A%E5%A4%9A%E5%88%86%E7%B1%BB%E7%9A%84%E4%BA%A4%E5%8F%89%E7%86%B5)
    1.  [公式](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E5%85%AC%E5%BC%8F-1)
    2.  [例 2](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E4%BE%8B-2)
    3.  [代码示例](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E4%BB%A3%E7%A0%81%E7%A4%BA%E4%BE%8B-1)
3.  [训练过程的反向传播（Backpropagation）](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%ADbackpropagation)
4.  [信息学视角](https://www.vectorexplore.com/tech/loss-functions/cross-entropy/#%E4%BF%A1%E6%81%AF%E5%AD%A6%E8%A7%86%E8%A7%92)

交叉熵（Cross Entropy）用于衡量一个概率分布与另一个概率分布之间的距离。

交叉熵是机器学习和深度学习中常常用到的一个概念。在分类问题中，我们通常有一个真实的概率分布（通常是专家或训练数据的分布），以及一个模型生成的概率分布，交叉熵可以衡量这两个分布之间的距离。

模型训练时，通过最小化交叉熵损失函数，我们可以使模型预测值的概率分布逐步接近真实的概率分布。

#### 概率视角解释（二分类的交叉熵）

我们先看一个示例，有个直观的印象；然后从更公式化的角度理解；最后看看代码中的写法。

##### 例 1

以简单的二分类问题为例，模型的预测值 $h_\theta(x_i)$ 由 $\sigma(Wx_i + b)$ 产生一个介于 0 和 1 之间的值，该值可以被解释为样本点 $x_i$ 属于正类的概率。如果这个概率小于0.5，我们将它分类为负例，否则我们将其分类为正例。

假设有一个银行，通过机器学习预测用户的信用卡是否会逾期（目标值 y 为 1 表示逾期）：

| 姓名  | 目标值 y | 预测为 1 的概率 $h_\theta(x_i)$ | 预测为 0 的概率（$1 - h_\theta(x_i)$） | 交叉熵损失                                       |
| --- | ----- | ------------------------- | ------------------------------ | ------------------------------------------- |
| 张三  | 1     | 0.7                       | 0.3                            | `-(0*log(0.3) + (1-0)*log(0.7)) = 0.356675` |
| 李四  | 0     | 0.2                       | 0.8                            | `-(1*log(0.8) + (1-1)*log(0.2)) = 0.223144` |

`(0.356675+0.223144)/2 = 0.289909`（为了和后面对照，舍第 6 位后的数字）

这里，交叉熵的计算可以理解为：目标值 y 对应的预测概率的对数（交叉的名字由来）。

##### 公式

概率表达式如下：

$\begin{align}    p(y_i = 1 \vert x_i) &= h_\theta(x_i) \\    p(y_i = 0 \vert x_i) &= 1 - h_\theta(x_i) \end{align}$

为了便于计算，可以把上面两种情况合并成一个表达式（可以简单验算即得）：

$p(y_i | x_i) = [h_\theta(x_i)]^{(y_i)} [1 - h_\theta(x_i)]^{(1 - y_i)}$

假设有 $N$ 条数据（样本点），数据是独立且同分布，我们可以简单地将数据相乘来写出[似然函数](https://baike.baidu.com/item/%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0/6011241)：

$L(x, y) = \prod_{i = 1}^{N}[h_\theta(x_i)]^{(y_i)} [1 - h_\theta(x_i)]^{(1 - y_i)}$

对上述表达式取对数，并利用对数的性质进行简化，最后求出整个表达式的相反数（==概率越小时，对数结果越小，取相反数转成正的损失值==），即得到交叉熵损失（注：公式计算的是总和）。

$J = -\sum_{i=1}^{N} [y_i\log (h_\theta(x_i)) + (1 - y_i)\log(1 - h_\theta(x_i))]$

##### 代码示例

scikit-learn 的代码示例如下，文档参考 [sklearn.metrics.log\_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。

```python
from sklearn.metrics import log_loss
import math

y_true=[[0,1],[1,0]] # 样本中的真实值
y_pred=[[0.3,0.7],[0.8,0.2]] # 模型预测值
print(log_loss(y_true, y_pred)) # 真实概率分布和预测概率分布的差异
# 输出：0.2899092476264711

# 按照例子中的数据，预期的输出值
print((math.log(0.7) + math.log(0.8))/2) 
# 输出 -0.2899092476264711
```

pytorch 和交叉熵相关的函数有：

1.  [torch.nn.functional.cross\_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) 交叉熵函数
2.  [torch.nn.functional.binary\_cross\_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html) 二分类交叉熵函数
3.  [torch.nn.functional.binary\_cross\_entropy\_with\_logits](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

对应的神经网络组件：

1.  [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 交叉熵损失计算
2.  [torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) 二分类交叉熵损失计算
3.  [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/BCEWithLogitsLoss.html) 这个损失函数将 Sigmoid 层和 BCELoss 结合在一个类中。

pytorch 实现的代码示例如下。
**pytorch深度学习框架内的函数会将标签值做one_hot处理,之后再进行交叉熵的公式运算.**  
**因此在使用pytorch框架的时候,标签无须自己做转换.**

```python
import torch

########### 使用 torch.nn.BCELoss ###############
bceLoss = torch.nn.BCELoss(reduction='mean')
# 和torch.nn.CrossEntropyLoss不一样，这里直接指定目标值的概率
input = torch.tensor([0.7, 0.2], requires_grad=True)
# 和多分类指定类别编号不一样，这里直接指定预期概率
target = torch.tensor([1.0, 0.0], requires_grad=False)
output = bceLoss(input, target)
output.backward()
print(output)
# 输出：tensor(0.2899, grad_fn=<BinaryCrossEntropyBackward0>)
```

#### 概率视角解释（多分类的交叉熵）

##### 公式

和二分类类似，总的损失函数（未取平均值）如下。

$J = -\sum_{i=1}^{N} \sum_{c=1}^{K} y_{ic} \log(h_{\theta}(x_{i}){_c})$

其中

-   $N$ ：样本点的数量
-   $K$ ：类别的数量
-   $y_{ic}$ ：样本目标值的 one-hot 编码。==如果样本 $x_{i}$ 的真实类别（再强调说明一下，不是预测的类别，“交叉”）等于 $c$ 取 1 ，否则取 0==
-   $h_{\theta}(x_{i}){_c}$ ：观测样本 $x_{i}$ 属于类别 $c$ 的预测概率

可以看出，当 $K=2$ 时，这个公式等价于上面二分类的公式。

##### 例 2

下面以一个图片分类问题为例，理解上面的公式。这个例子，根据图片中动物的轮廓、颜色等特征，来预测动物的类别（猫、狗、马）。

| 类别名称 | 类别编号 | one-hot |
| --- | --- | --- |
| 猫 | 0 | \[1, 0, 0\] |
| 狗 | 1 | \[0, 1, 0\] |
| 马 | 2 | \[0, 0, 1\] |

现在有一张藏獒的图片，参考上表，它的真实类别编号为 1， one-hot 编码为 \[0, 1, 0\] 。假定模型的预测概率为\[0.4, 0.4, 0.2\]，如下为计算交叉熵的过程。

| 类别名称 | 预测概率 | 类别的one-hot | 符号函数值 $y_{ic}$ | 单项交叉熵计算过程 | 单项交叉熵结果 |
| --- | --- | --- | --- | --- | --- |
| 猫 | 0.4 | \[1, 0, 0\] | 0 | \-0 \* log(0.4) | 0 |
| 狗 | 0.4 | \[0, 1, 0\] | 1 | \-1 \* log(0.4) | 0.9163 |
| 马 | 0.2 | \[0, 0, 1\] | 0 | \-0 \* log(0.2) | 0 |

这里样本为单张图片，故数量 $N=1$。

单项交叉熵的计算公式：$y_{ic} \log(h_{\theta}(x_{i}){_c}$

总的交叉熵（$\sum_{c=1}^{K}$）为 0.9163（0 + 0.9163 + 0）。

另有一张马的图片，它的真实类别编号为 2， one-hot 编码为 \[0, 0, 1\] 。假定模型的预测概率为\[0.1, 0.12, 0.78\]，如下为计算交叉熵的过程。

| 类别名称 | 预测概率 | 类别的one-hot | 符号函数值 $y_{ic}$ | 单项交叉熵计算过程 | 单项交叉熵结果 |
| --- | --- | --- | --- | --- | --- |
| 猫 | 0.1 | \[1, 0, 0\] | 0 | \-0 \* log(0.1) | 0 |
| 狗 | 0.12 | \[0, 1, 0\] | 0 | \-0 \* log(0.12) | 0 |
| 马 | 0.78 | \[0, 0, 1\] | 1 | \-1 \* log(0.78) | 0.2485 |

总的交叉熵为0.2485。显然，这次预测比藏獒的靠谱很多。

##### 代码示例

以藏獒的数据为例，scikit-learn的计算示例：

```python
from sklearn.metrics import log_loss
y_true=[[0, 1, 0]] # 样本点的 one-hot 真实值
y_pred=[[0.4, 0.4, 0.2]] # 模型预测值


# 真实概率分布和预测概率分布的差异
# normalize：True 表示取平均值，否则为总和，默认为True
print(log_loss(y_true, y_pred, normalize=True)) 
# 输出：0.916290731874155
```

pytorch的计算示例：

```python
import torch
import math

# 可以指定用平均值（mean）还是总和（sum），上面的公式仅列出总和
loss = torch.nn.CrossEntropyLoss(reduction='mean')

# nn.CrossEntropyLoss会对输入值做softmax（做exp），故这里为了方便说明，指定exp后的值
input = torch.tensor([[math.log(0.4), math.log(0.4), math.log(0.2)]], requires_grad=True)

# 狗的分类编号是1
target = torch.tensor([1])
output = loss(input, target)
output.backward()
print(output) 
# 输出：tensor(0.9163, grad_fn=<NllLossBackward0>)

print((math.log(0.4))/1) 
# 输出 -0.916290731874155
```

#### 训练过程的反向传播（Backpropagation）

训练过程中，交叉熵损失函数的优化也是高效的。

#### 信息学视角

交叉熵损失可以用信息论解释。在信息论中，==Kullback-Leibler（KL）散度是用于衡量两个概率分布之间的差异性的==。在分类问题中，我们有两个不同的概率分布：第一个分布对应着我们**真实的类别标签**，在这个分布中，==所有的概率质量都集中在正确的类别上==（其他类别上为 0）；第二个分布对应着**我们的模型的预测**，这个分布==由模型输出的原始分数经过softmax函数转换后得到==。

==在一个理想的情况下，我们模型预测得到的分布应该和真实的分布完全吻合，也就是说模型将100%的概率预测到正确的标签上==。

然而，实际情况不太可能，因为如果我们这么做，原始分数（未经过 softmax 或者其他转换函数）对于正确的类别就是无限大，而对于错误的类别就是负无限小。**解释在下面**

将交叉熵损失理解为两个概率分布之间的 KL 散度，也可以帮助我们理解数据噪声的问题。很多数据集都是部分标记的或者带有噪声（即标签有时候是错误的），如果我们以某种方式将这些未标记的部分也分配上标签，或者将错误的标签看作是从某个概率分布中取样得到的，那么我们依然可以使用最小化 KL 散度的思想（在这种情况下，真实分布已经不再将所有的概率质量都集中在单个标签上了）。

参考：

1.  [Picking Loss Functions - A comparison between MSE, Cross Entropy, and Hinge Loss](https://rohanvarma.me/Loss-Functions/)

---

**解释：**

#### 1. 理想情况：模型预测分布 ≈ 真实分布

真实分布是 **“one-hot 分布”**（所有概率集中在正确类别，其他为 0 ）。 如果模型完美预测，Softmax 输出的分布也应无限接近 one-hot（正确类别概率→1，错误类别→0 ）。

#### 2. 为什么 “原始分数要无限大 / 负无限小”？

模型的 **原始分数（logits）** 是 Softmax 的输入，Softmax 公式为：

$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

其中 $z_i$ 是第 i 类的原始分数（logits）。

要让 Softmax 输出的 $p_{\text{正确类别}} \to 1$，需满足：
$\frac{e^{z_{\text{正确}}}}{\sum_j e^{z_j}} \to 1$

这要求 **$z_{\text{正确}} \gg z_{\text{错误}}$**（正确类的 logits 远大于错误类 ）。

- 极端情况（理想完美预测 ）：

  - 正确类的 $z_{\text{正确}} \to +\infty$
  - 错误类的 $z_{\text{错误}} \to -\infty$

  此时：

  $\text{Softmax}(z)_{\text{正确}} = \frac{e^{+\infty}}{\sum_j e^{z_j}} \approx \frac{e^{+\infty}}{e^{+\infty} + 0 + ... + 0} = 1$

  错误类的概率则趋近于 0，完美匹配真实分布。

#### 3. 实际中为什么做不到？

因为：

- **模型能力有限**：无法让 logits 无限大 / 小（受限于训练数据、模型复杂度 ）。
- **过拟合风险**：即使能做到，也会让模型 “过度关注当前样本的正确类别”，在新样本上泛化能力极差。
- **Softmax 的特性**：Softmax 是 “相对概率”，依赖所有类的 logits 关系。如果强行让正确类 logits 极大，模型训练会极不稳定（梯度爆炸、数值计算溢出 ）。


