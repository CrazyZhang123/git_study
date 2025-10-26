
# 一、LLM分类

主要分为Encoder-Decoder架构、Casual LM(根据前面的token 预测后面的token),prefix LM(预测token有一段可以做self-attention，区别于Casual LM!!,比如chatGLM)

### 主流Casual LM，后面主要计算这个

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251022001131780.png)

# 二、常用符号和结构

### 1、常用符号

- **L**: layer_number 层数量
- **h**: hidden_dim 隐藏层维度
- **V**: vocab size 词表大小
- **b**: batch_size batch大小
- **s**: seq_len 序列长度

### 2、Casual LM结构：

- input token ->emb
- layer -> MHA + FFN
	- MHA: layernorm + MHA
	- FFN: layernorm + linear
- output(softmax)

#### (1)MHA参数计算
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251022002256845.png)
QKVO四个矩阵，需要过linear层，有W和b两个参数；维度分别是h^2+h;因此MHA处的参数量是4(h^2+h)

##### 那 `head_num` 为什么没出现？

因为大多数框架（包括 PyTorch、Transformers）里：

> **多头只是对 `Q, K, V` 的结果在维度上重新 reshape，不会增加参数。**

即：
- Q、K、V 先生成维度 [batch, seq, h] 的张量；
- 然后 reshape → [batch, head_num, seq, head_dim]；
- 计算注意力；
- 再拼接回来 [batch, seq, h]；
- 再过 O 投影。

所以 **head_num 只是划分维度的方式**，不影响参数量。
#### (2) FFN参数计算

(b,s,h)->(b,s,4h)->(b,s,h)
- **升维**: \[h,4h]   W: 4\*h^2    b: 4h
- **降维**: \[4h,h]   W: 4\*h^2   b: h
**FFN的总参数量**: 8\*h^2 + 5h


#### (3) Layer Norm计算

> 论文 **Layer Normalization** 中的操作：
> $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x]} + \epsilon} * \gamma + \beta$$
> 均值和标准差是在输入张量的最后 **D 个维度** 上计算的，其中 D 是参数 `normalized_shape` 的维度数。  
> 例如，当 `normalized_shape` 为 `(3, 5)`（一个二维形状）时，均值和标准差会在输入的最后两个维度上计算：
> $$\text{input.mean((-2, -1))}$$

> 参数 γ（gamma）和 β（beta）是可学习的仿射变换参数，其形状与 `normalized_shape` 相同（前提是 `elementwise_affine=True`）。
> 方差的计算采用有偏估计器，相当于：
> $$\text{torch.var(input, unbiased=False)}$$

layernorm参数来自γ（gamma）和 β（beta）大小是2hidden_size

**layer norm的参数**是 2\*h


### 3、所有的参数量

input + output
- input层: 把token从V(词表大小)映射为hidden_size(embedding操作)，权重有可能和output的权重共享。
#### ALL_size = MHA + layernorm + FFN + layernorm + input + output

=  \[4h^2 + 4h + 8h^2 + 5h + 2\*2h(layernorm)\] \* L(层) + Vh

= 12Lh^2+13Lh + Vh

> **为什么 vh 可以当做小量。** 12Lh^2 和 vh 比，12Lh >> v，因为 h 已经是 4k+，所以如果相对于 12Lh^2 来说，vh 还是比较小的（除非你的模型size 非常的小，6B以上的模型是可以当小项的）
#### = 12Lh^2 + small parm



### 4、测试估算准确性

|           | V     | H    | L   | I     | 实际参数量       |
| --------- | ----- | ---- | --- | ----- | ----------- |
| llama-7B  | 32000 | 4096 | 32  | 11008 | 6738415616  |
| llama-13B | 32000 | 5120 | 40  | 13824 | 13015864320 |
| llama-30B | 32000 | 6656 | 60  | 17920 | 32528943616 |
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251022010449706.png)

可以看到下面的计算以及非常接近实际的参数了。
- **6.7B:**
	- 12\*L\*h^2 = 12 \*32 \*4096^2 = 6,442,450,944
- **13B**
	- 12\*L\*h^2 = 12 \*40 \*5120^2 = 12,582,912,000
- **30B**
	- 12\*L\*h^2 = 12 \*60 \*6656^2 = 31,897,681,920
		- 全量: + 13\*60\*6566 + 32000\*6656 = 32,115,865,600 