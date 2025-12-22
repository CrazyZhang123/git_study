## 一、用 PyTorch 写一个最简探针（逻辑回归）

先写一个抽象掉大模型的版本，只看“有一堆向量 x + 二元标签 y”，在它们上面训练探针。

### 1. 定义一个 probe 模型

就是一个线性层 + Sigmoid：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prob(nn.Module):
    def __init__(self,dim):
        super().__init__()
        # 线性层 y=wx 不使用偏置
        self.linear = nn.Linear(dim,1,bias=False)

    def forward(self,x):
        """
        forward 的 Docstring
        
        :param x: [batch_size,dim] 对应很多个x_l^h
        返回[batch_size] 的概率
        
        """
        # 实际计算的是内积 即 x_l^h * w
        # 线性层输出 [B,1]  squeeze是压缩指定dim上为1的维度，
        # 即去掉dim=-1的维度，即[B,1]->[B]
        logits = self.linear(x).squeeze(-1) # [B]
        # 对logits 应用sigmoid 函数，得到[0,1]之间的概率
        prob = torch.sigmoid(logits) # [B]

        # 返回概率和logits
        return prob,logits




```

对应关系：

- 这里的 `self.linear.weight` 就是论文里的 θ⊤θ⊤ 
- `logits = θ·x`
- `probs = sigmoid(logits) = p_θ(x)`
### 2. 在一堆“中间向量 + 标签”上训练探针

假设你已经有一些样本：

- `X`：若干个 $x_l^h$​，形状 `[num_samples, D]`
- `y`：对应标签（0 = 幻觉 / 虚假，1 = 真实）

```python
# 假设dim是1024维
dim = 1024
probe = Prob(dim)

optimizer = torch.optim.Adam(probe.parameters(),lr=1e-4)

def train_probe(X,y,epochs=100):
    """
    train_probe 的 Docstring
    
    :param X: [batch_size,dim] 对应很多个x_l^h
    :param y: [batch_size] 对应标签[0/1]
    """

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()

        # 前向传播
        prob,logits = probe(X) # prob: [B], logits: [B]

        # 计算二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(logits,y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```
这段代码就是：

- 把论文里的 pθ(x)=σ(θ⊤x)
- 变成了 PyTorch 里的：
    - `logits = linear(x)`
    - `probs = sigmoid(logits)`
    - 用交叉熵训练参数 θθ。

### 3. 测试运行

```python
# 测试代码
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 64
    dim = 1024
    
    # 随机生成输入特征 X
    X = torch.randn(batch_size, dim)
    
    # 生成标签 y (二分类：0 或 1)
    # 这里我们创建一些有规律的标签来让模型更容易学习
    # 比如，如果前512维的和大于某个阈值，则标签为1
    y = (X[:, :512].sum(dim=1) > 0).float()
    
    print("测试数据准备完成:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y分布: {y.sum().item()}/{batch_size} 个正样本")
    print("\n开始训练...")
    
    # 开始训练
    train_probe(X, y, epochs=100)
    
    print("\n训练完成！")
    
    # 测试模型预测
    probe.eval()
    with torch.no_grad():
        test_prob, test_logits = probe(X[:5])  # 取前5个样本测试
        print("\n测试预测结果:")
        print(f"预测概率: {test_prob}")
        print(f"真实标签: {y[:5]}")
        print(f"预测准确: {((test_prob > 0.5).float() == y[:5]).sum().item()}/5")
```

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251218111917994.png)


# 二、MLP+steer

# L2S 完整 PyTorch 实现（论文核心逻辑落地）
以下是论文中 L2S 方法的可运行 PyTorch 代码实现，覆盖**辅助网络定义、训练、推理全流程**，所有公式均与论文一一对应，代码可直接复制到 `.py` 或 notebook 运行（随机数据示范，替换真实数据即可落地）。

---

## 1. 数据集定义：存储 $h_{X,L'}$ 和 $z_{X,L^*}$
核心是构建「输入语境向量 $h_{X,L'}$」与「目标引导向量 $z_{X,L^*}$」的配对数据集，对应论文中训练辅助网络的输入-标签对。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class SteeringDataset(Dataset):
    """
    数据集：每个样本是 (h_{X,L'}, z_{X,L^*})
    - h_context: [N, D]  → 输入语境向量 h_{X,L'}（L'层最后一个输入token的隐表示）
    - z_target:  [N, D]  → P2S计算的目标引导向量 z_{X,L^*} = h^+ - h^-
    """
    def __init__(self, h_context: torch.Tensor, z_target: torch.Tensor):
        assert h_context.shape == z_target.shape, "输入和标签维度必须一致"
        self.h_context = h_context
        self.z_target = z_target

    def __len__(self):
        return self.h_context.size(0)

    def __getitem__(self, idx):
        return self.h_context[idx], self.z_target[idx]
```

> 真实场景替换说明：
> - `h_context`：从 LLM 的 $L'$ 层抽取「输入（图像+文本）最后一个 token」的隐表示 $h_{N_V+N_T}^{L'}(X)$；
> - `z_target`：通过 P2S 方法计算 $z_{X,L^*} = h_{q^+}^{L^*}(X^+) - h_{q^-}^{L^*}(X^-)$。

---

## 2. 辅助网络 + 训练循环（实现公式 (8)）
### 2.1 两层 MLP 定义（对应 $g_{\Theta^*}$）
论文明确使用「轻量级两层 MLP」作为辅助网络，实现从 $h_{X,L'}$ 到 $z_{X,L^*}$ 的映射 $g_{\Theta^*}: \mathbb{R}^D \to \mathbb{R}^D$。

```python
class SteeringMLP(nn.Module):
    """
    两层感知机（MLP）：实现论文中的 g_Θ^*
    - 输入：h_{X,L'} (维度 D)
    - 输出：预测的引导向量 \hat{z}_{X,L^*} (维度 D)
    """
    def __init__(self, dim_hidden: int, dim_mlp: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_hidden, dim_mlp),  # 第一层线性变换
            nn.ReLU(),                       # 激活函数
            nn.Linear(dim_mlp, dim_hidden)   # 第二层映射回原维度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### 2.2 训练循环（优化公式 (8)）
论文的损失函数是：
$$\Theta^* = \arg\min_{\Theta} \mathbb{E}_X\left[\left\|z_{X,L^*}-g_{\Theta}(h_{X,L'})\right\|_2^2\right]$$
本质是最小化「预测引导向量」与「真实引导向量」的 MSE 损失，以下是完整训练逻辑：

```python
def train_steering_mlp(
    dim_hidden: int = 2048,    # 隐层维度（与LLM的hidden_dim一致）
    num_samples: int = 10000,  # 样本数（示范用）
    dim_mlp: int = 1024,       # MLP中间层维度
    batch_size: int = 64,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # ===== 1. 构造示范数据（真实场景替换为自己的 h_context/z_target）=====
    # 模拟 h_{X,L'}：[N, D]
    h_context = torch.randn(num_samples, dim_hidden)
    # 模拟真实z_target（加噪声更贴近真实场景）
    W_true = torch.randn(dim_hidden, dim_hidden)  # 模拟真实映射
    z_target = h_context @ W_true + 0.01 * torch.randn(num_samples, dim_hidden)

    # 构建数据集和加载器
    dataset = SteeringDataset(h_context, z_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ===== 2. 初始化模型、优化器、损失函数 =====
    model = SteeringMLP(dim_hidden=dim_hidden, dim_mlp=dim_mlp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 对应公式中的 L2 损失

    # ===== 3. 训练循环 =====
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for h_batch, z_batch in dataloader:
            # 数据移到设备上
            h_batch = h_batch.to(device)
            z_batch = z_batch.to(device)

            # 前向传播：g_Θ(h_{X,L'}) → 预测引导向量
            z_pred = model(h_batch)

            # 计算损失：||z_target - z_pred||_2^2
            loss = criterion(z_pred, z_batch)

            # 反向传播 + 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            total_loss += loss.item() * h_batch.size(0)

        # 打印epoch平均损失
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - MSE Loss: {avg_loss:.6f}")

    return model


# 运行训练（示范）
if __name__ == "__main__":
    steering_net = train_steering_mlp()
```

> 运行效果：Loss 会持续下降，说明 MLP 成功学到了 $h_{X,L'} \to z_{X,L^*}$ 的映射。
> 真实场景替换：删除「构造示范数据」部分，替换为自己从 LLM 抽取的 $h_context$ 和 $z_target$ 即可。

---

## 3. 推理阶段：实现公式 (9)
论文的推理阶段核心公式：
$$h_p^{L^*} \leftarrow h_p^{L^*} + \alpha \cdot g_{\Theta^*}(h_{X,L'}) \quad (p > N_V + N_T)$$
以下是该逻辑的代码实现，可嵌入 LLM 生成流程：

### 3.1 引导向量施加函数
```python
def apply_l2s_steering(
    steering_net: nn.Module,
    h_X_Lprime: torch.Tensor,
    h_p_Lstar: torch.Tensor,
    alpha: float = 0.5
):
    """
    对 L* 层的隐表示施加 L2S 引导（公式 (9)）
    :param steering_net: 训练好的 g_{Θ^*}
    :param h_X_Lprime: [B, D] → 输入语境向量 h_{X,L'}
    :param h_p_Lstar:  [B, D] → 第p个token在L*层的隐表示 h_p^{L^*}
    :param alpha: 引导强度超参数
    :return: 施加引导后的隐表示 h_p^{L^*} (steered)
    """
    # 确保所有张量在同一设备
    device = next(steering_net.parameters()).device
    h_X_Lprime = h_X_Lprime.to(device)
    h_p_Lstar = h_p_Lstar.to(device)

    # 推理阶段禁用梯度（节省显存）
    with torch.no_grad():
        # 预测输入专属引导向量：g_{Θ^*}(h_{X,L'})
        steer_vec = steering_net(h_X_Lprime)  # [B, D]

    # 公式 (9)：施加线性偏移
    h_p_Lstar_steered = h_p_Lstar + alpha * steer_vec
    return h_p_Lstar_steered
```

### 3.2 嵌入 LLM 生成流程（示范）
需结合 LLM 的「隐层钩子（hook）」实现，以下是伪代码框架（可适配 LLaMA/LLaVA/Qwen 等模型）：

```python
# 假设已准备好：
# - steering_net: 训练好的辅助网络
# - h_X_Lprime: [B, D] → 输入语境向量（L'层最后一个输入token的隐表示）
# - get_hidden_Lstar(): 自定义函数，获取L*层当前token的隐表示 h_p^{L^*}
# - set_hidden_Lstar(): 自定义函数，将修改后的隐表示写回模型
# - generate_next_token(): 自定义函数，从隐表示生成下一个token

alpha = 0.5  # 引导强度（可调参，0.1~1.0 为宜）
max_new_tokens = 100  # 最大生成token数

# 生成循环（对应 p > N_V + N_T 的输出阶段）
for step in range(max_new_tokens):
    # 1. 获取当前step L*层的原始隐表示 h_p^{L^*}
    h_p = get_hidden_Lstar()  # 形状 [B, D]

    # 2. 应用 L2S 引导，修改隐表示
    h_p_steered = apply_l2s_steering(
        steering_net=steering_net,
        h_X_Lprime=h_X_Lprime,
        h_p_Lstar=h_p,
        alpha=alpha
    )

    # 3. 将修改后的隐表示写回模型
    set_hidden_Lstar(h_p_steered)

    # 4. 生成下一个token（正常LLM采样逻辑）
    next_token = generate_next_token()

    # 5. 终止条件（如生成<EOS>）
    if next_token == EOS_TOKEN:
        break
```

> 关键实现点：
> - `get_hidden_Lstar()`/`set_hidden_Lstar()`：需通过 PyTorch 的 `register_forward_hook` 实现，或直接修改 LLM 源码，抽取/修改指定层（L*）的隐表示；
> - 仅对 $p > N_V + N_T$ 的输出 token 施加引导（输入阶段不修改）。

---

## 核心优势（代码视角解读论文）
论文强调 L2S 「轻量化、低显存」，对应代码层面的体现：
1. **辅助网络极小**：`SteeringMLP` 仅两层线性层，参数规模远小于 LLM（如 2048 维隐层的 MLP 仅约 2048×1024×2 ≈ 400 万参数，而 LLaMA-7B 有 70 亿参数）；
2. **训练无需完整LLM**：训练仅用预抽取的隐表示张量，无需加载/运行完整 LLM，显存消耗仅为 MLP 训练开销（单卡即可）；
3. **推理轻量化**：仅在生成阶段对指定层做「向量加法」，几乎不增加推理耗时。

---

## 进阶拓展（可选）
若需适配具体开源模型（如 LLaVA/LLaMA/Qwen），可补充实现：
1. 基于 `transformers` 库的「隐层钩子」，抽取 $h_{X,L'}$ 和 $h_p^{L^*}$；
2. P2S 方法的完整实现（生成对比提示 $T_X^+/T_X^-$，计算 $z_{X,L^*}$）；
3. 多场景调参策略（如 $\alpha$ 网格搜索、$L^*/L'$ 层选择）。

只需告知你熟悉的模型框架，即可补充对应代码。