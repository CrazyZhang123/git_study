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
