"""
L2S (Learn to Steer) 完整实现
轻量级输入依赖引导方法
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class SteeringDataset(Dataset):
    """数据集：每个样本是 (h_{X,L'}, z_{X,L^*})"""
    def __init__(self, h_context, z_target):
        assert h_context.shape == z_target.shape
        self.h_context = h_context
        self.z_target = z_target
    
    def __len__(self):
        return self.h_context.size(0)
    
    def __getitem__(self, idx):
        return self.h_context[idx], self.z_target[idx]

class SteeringMLP(nn.Module):
    """两层MLP作为辅助网络 g_Θ^*"""
    def __init__(self, dim_hidden, dim_mlp=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_hidden, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_hidden)
        )
    
    def forward(self, x):
        return self.net(x)

def train_steering_mlp(dim_hidden=2048, num_samples=5000, dim_mlp=1024,
                      batch_size=64, num_epochs=20, lr=1e-3):
    """训练L2S辅助网络"""
    
    # 1. 生成数据
    print("生成训练数据...")
    h_context = torch.randn(num_samples, dim_hidden)
    W_true = torch.randn(dim_hidden, dim_hidden)
    z_target = h_context @ W_true + 0.01 * torch.randn(num_samples, dim_hidden)
    
    dataset = SteeringDataset(h_context, z_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 初始化模型
    print("初始化模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = SteeringMLP(dim_hidden, dim_mlp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    
    # 3. 训练循环
    print("\n开始训练...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for h_batch, z_batch in dataloader:
            h_batch = h_batch.to(device)
            z_batch = z_batch.to(device)
            
            # 前向传播
            z_pred = model(h_batch)
            
            # 计算损失
            loss = criterion(z_pred, z_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * h_batch.size(0)
        
        avg_loss = total_loss / len(dataset)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}")
    
    print(f"\n训练完成！最终损失: {avg_loss:.8f}")
    return model

def demo_input_dependent():
    """演示输入依赖的引导行为"""
    print("\n" + "="*50)
    print("输入依赖引导演示")
    print("="*50)
    
    dim_hidden = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_steering_mlp(dim_hidden=dim_hidden, num_samples=1000, num_epochs=10)
    model.eval()
    
    # 测试不同输入
    test_inputs = torch.randn(3, dim_hidden)
    
    print("\n测试3个不同输入:")
    with torch.no_grad():
        for i, h_input in enumerate(test_inputs):
            h_input = h_input.to(device)  # 确保数据在正确设备上
            z_pred = model(h_input)
            print(f"输入{i+1}: ||h||={h_input.norm():.3f}, ||z_pred||={z_pred.norm():.3f}")

if __name__ == "__main__":
    print("L2S (Learn to Steer) - 轻量级引导网络")
    print("实现公式: Θ^* = argmin_Θ E_X[||z - g_Θ(h)||_2^2]")
    
    # 运行完整训练
    model = train_steering_mlp(
        dim_hidden=2048,
        num_samples=5000,
        dim_mlp=1024,
        batch_size=128,
        num_epochs=20,
        lr=1e-3
    )
    
    # 演示输入依赖
    demo_input_dependent()
    
    print("\n" + "="*50)
    print("优势总结:")
    print("- 轻量级MLP，参数量仅约4M")
    print("- 训练时间极短，无需调用完整LLM")
    print("- 内存消耗极低")
    print("- 支持输入依赖的灵活引导")
    print("="*50)
