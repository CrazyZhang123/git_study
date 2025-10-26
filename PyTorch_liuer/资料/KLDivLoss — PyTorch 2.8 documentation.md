## **_class_ torch.nn.KLDivLoss(_size\_average\=None_, _reduce\=None_, _reduction\='mean'_, _log\_target\=False_)**

[source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L453)

The Kullback-Leibler divergence loss.

**Kullback-Leibler 散度损失（KL 散度损失）。**

For tensors of the same shape $y_{\text{pred}},\ y_{\text{true}}$, where $y_{\text{pred}}$ is the `input` and $y_{\text{true}}$ is the `target`, we define the **pointwise KL-divergence** as

对于形状相同的张量 \(y_{\text{pred}}\) 和 \(y_{\text{true}}\)（其中 \(y_{\text{pred}}\) 为`input`（输入），\(y_{\text{true}}\) 为`target`（目标）），我们定义**逐点 KL 散度**为：

$L(y_{\text{pred}},\ y_{\text{true}})     = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}     = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})$

To avoid underflow issues when computing this quantity, this loss expects the argument `input` in the log-space. The argument `target` may also be provided in the log-space if `log_target`\= True.

==为避免计算过程中出现下溢问题，该损失要求参数`input`（输入）处于对数空间。若设置`log_target=True`，则参数`target`（目标）也可处于对数空间。==

To summarise, this function is roughly equivalent to computing

综上，该函数的计算过程大致等价于：

```python
if not log_target:  # default
    loss_pointwise = target * (target.log() - input)
else:
    loss_pointwise = target.exp() * (target - input)
```

and then reducing this result depending on the argument `reduction` as

然后根据参数`reduction`（归约方式）对结果进行归约：

```python
if reduction == "mean":  # default
    loss = loss_pointwise.mean()
elif reduction == "batchmean":  # mathematically correct  # 数学上的正确方式
    loss = loss_pointwise.sum() / input.size(0)
elif reduction == "sum":
    loss = loss_pointwise.sum()
else:  # reduction == "none"
    loss = loss_pointwise
```

**Note**

As all the other losses in PyTorch, this function expects the first argument, `input`, to be the output of the model (e.g. the neural network) and the second, `target`, to be the observations in the dataset. This differs from the standard mathematical notation $KL(P\ ||\ Q)$ where $P$ denotes the distribution of the observations and $Q$ denotes the model.

与 PyTorch 中的其他所有损失函数一样，该函数要求第一个参数`input`是模型（如神经网络）的输出，第二个参数`target`是数据集中的观测值。**这与标准的数学符号\(KL(P\ ||\ Q)\)不同，在数学符号中，P表示观测值的分布，Q表示模型的分布。**

**Warning**

`reduction`\= “mean” doesn’t return the true KL divergence value, please use `reduction`\= “batchmean” which aligns with the mathematical definition.

==当`reduction`="mean" 时，返回的不是真实的 KL 散度值，请使用`reduction`="batchmean"，这与数学定义一致。==

**Shape:**

-   Input: $(*)$, where $*$ means any number of dimensions. 任意数量的维度。
-   Target: $(*)$, same shape as the input. 与输入形状相同。
-   Output: scalar by default. If `reduction` is ‘none’, then $(*)$, same shape as the input.
-   输出：默认情况下为标量。若`reduction`为 'none'，则输出形状与输入相同，即\((*)\)。

**Examples**

```python
kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
# 输入应是对数空间中的分布
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# Sample a batch of distributions. Usually this would come from the dataset
# 采样一批分布。通常这来自数据集
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, target)
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
log_target = F.log_softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, log_target)
```