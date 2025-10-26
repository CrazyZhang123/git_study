
```python
python steering_928.py \
	 --model_name llama3.1_8B \
	 --layer 14 \
	 --save_path checkpoints/llama3.1_8B_L14_mat_steer_928.pt \
	 --batch_size 96 \
	 --epochs 100 \
	 --lr 0.001 \
	 --sigma 2.0 \
	 --lambda_mmd 1.0 \
	 --lambda_sparse 0.9 \
	 --lambda_ortho 0.1 \
	 --lambda_pos 0.9
```

对于TruthfulQA数据集（基于Apache-2.0协议授权），我们将样本按40/10/50的比例划分为训练集（train）、开发集（dev，用于超参数调优）和测试集（test）。

对于Toxigen数据集（基于MIT协议授权）和BBQ数据集（基于cc-by-4.0协议授权），这两个数据集已预先划分为训练集和验证集（validation set）。我们将原验证集用作测试集，同时将原训练集按80/20的比例进一步划分为新的训练集和开发集。

对于HelpSteer数据集（基于cc-by-4.0协议授权），我们先为每个属性（attribute）抽取500个正向样本和500个负向样本，再将这些样本按40/10/50的比例划分为训练集、开发集和测试集。

所有方法均在“来自不同数据集的训练集合并后的集合”上进行训练，并在每个任务对应的独立测试集上进行评估。所有数据集均为英文数据集。

```python
def train_multi_task_steering(tasks, num_attributes, batch_size, epochs, lr, sigma, lambda_mmd, lambda_sparse, lambda_ortho, lambda_pos, save_path):
    """
    该函数是 MAT-Steer（Multi-Attribute Targeted Steering）项目的核心训练函数，主要用于训练一个能够同时控制大型语言模型在多个任务/属性上行为的转向模块。
    它通过学习一组转向向量（steering vectors）和对应的门控机制，使模型能够在不重新训练的情况下，针对不同任务特征进行有针对性的干预。
    tasks: 字典，键为任务名称，值为包含正负样本激活值的元组 (positive_activations, negative_activations)
    
    输入参数：
    num_attributes: 整数，表示要控制的属性/任务数量
    batch_size: 整数，训练批次大小
    epochs: 整数，训练轮数
    lr: 浮点数，学习率
    sigma: 浮点数，用于 MMD 损失函数中高斯核的带宽参数
    lambda_*: 浮点数，各类损失函数的权重系数
    save_path: 字符串，训练后模型的保存路径
    """

    # 1. 初始化转向模块
    # task定义：
    # tasks[dataset_name] = (pos_acts, neg_acts)
    input_dim = list(tasks.values())[0][0].shape[1]  
    # print('input_dim:',input_dim) # 4096

    # 实例化引导模块
    model = SteeringModule(input_dim, num_attributes)
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare balanced sampling from each task
    task_names = list(tasks.keys())
    
    from tqdm import tqdm
    # 修改tqdm进度条
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        
        epoch_loss = 0
        num_batches = 0

        min_samples = min(min(tasks[task][0].shape[0], tasks[task][1].shape[0]) for task in tasks)
        
      
        effective_batch_size = min(batch_size // (2 * num_attributes), min_samples)
       
        if effective_batch_size < 1:
            effective_batch_size = 1
        
        for batch_idx in range(0, min_samples, effective_batch_size):
            batch_activations = []
            batch_labels = []
            batch_task_indices = []
            
            for t, task_name in enumerate(task_names):
                pos_acts, neg_acts = tasks[task_name]
                
                pos_indices = torch.randperm(pos_acts.shape[0])[:effective_batch_size]
                neg_indices = torch.randperm(neg_acts.shape[0])[:effective_batch_size]
                
                pos_batch = pos_acts[pos_indices]
                neg_batch = neg_acts[neg_indices]
                
                # 构建批次数据
                batch_activations.append(pos_batch)
                batch_activations.append(neg_batch)
                # 构建批次标签
                batch_labels.extend([1] * effective_batch_size)  # positive
                batch_labels.extend([0] * effective_batch_size)  # negative
                # 任务索引
                batch_task_indices.extend([t] * effective_batch_size * 2)
            batch_activations = torch.cat(batch_activations, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
            batch_task_indices = torch.tensor(batch_task_indices, dtype=torch.long)
            
            adjusted_acts, gates = model(batch_activations)
            # 通过元素级乘法，将调整后的激活值按照 原始范数/调整后范数 的比例进行缩放
            adjusted_acts = normalize_activations(batch_activations, adjusted_acts)
            
            # Compute losses
            total_loss = 0
            
            # (1)MMD 损失 
            # MMD loss per attribute
            loss_mmd = 0
            for t, task_name in enumerate(task_names):
                # 当前任务的idx
                # task_mask：用于筛选出当前任务的样本
                task_mask = batch_task_indices == t
                if task_mask.sum() > 0:
                    # 筛选出当前任务的调整后的激活值
                    task_acts = adjusted_acts[task_mask]
                    # 筛选出当前任务的标签
                    task_lbls = batch_labels[task_mask]
                    # 正负样本掩码
                    pos_mask = task_lbls == 1
                    neg_mask = task_lbls == 0
                    
                    # 确保正负样本数量都>0
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        # 取出当前任务的正负样本调整后的激活值
                        pos_adjusted = task_acts[pos_mask]
                        neg_adjusted = task_acts[neg_mask]
                       
                        original_pos = tasks[task_name][0]
                        # 使用 torch.randperm() 随机采样，确保样本的随机性
                        sample_indices = torch.randperm(original_pos.shape[0])[:min(pos_adjusted.shape[0], original_pos.shape[0])]
                        # 采样数量与调整后的正样本数量保持一致
                        original_pos_sample = original_pos[sample_indices]
                        
                        # 调用 compute_mmd() 函数计算调整后的负样本与原始正样本之间的分布距离，并累加到总损失中。
                        loss_mmd += compute_mmd(neg_adjusted, original_pos_sample, sigma)
            
            # 对所有任务的MMD损失取平均值，确保不同任务数量下损失值的尺度一致。
            loss_mmd = loss_mmd / num_attributes
            
            # (2)负向样本的稀疏性损失
            # Sparsity loss on negative examples
            neg_mask = batch_labels == 0
            if neg_mask.sum() > 0:
                loss_sparse = sparsity_loss(gates[neg_mask])
            else:
                loss_sparse = torch.tensor(0.0)
            
            # (3)正向样本的保留损失
            # Preservation loss on positive examples
            pos_mask = batch_labels == 1
            if pos_mask.sum() > 0:
                loss_pos = preservation_loss(gates[pos_mask])
            else:
                loss_pos = torch.tensor(0.0)
            
            # (4)引导向量的正交性损失
            # Orthogonality loss
            # 输入所有的引导向量
            loss_ortho = orthogonality_loss([sv for sv in model.steering_vectors])
            
            # Combined loss
            batch_loss = (lambda_mmd * loss_mmd + 
                         lambda_sparse * loss_sparse + 
                         lambda_ortho * loss_ortho + 
                         lambda_pos * loss_pos)
            # 反向传播
            batch_loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
            # 计算当前批次的损失值
            batch_loss = batch_loss.item()
            # 累计当前批次的损失值
            epoch_loss += batch_loss
            # 累计的批次数量
            num_batches += 1
        
        # 每10个epoch打印一次平均损失值
        if epoch % 10 == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # 模型保存
    # Save model with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_attributes': num_attributes,
        'task_names': task_names
    }
    # 训练结束后，将模型参数和元数据（输入维度、属性数量、任务名称）保存到指定路径
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    
    return model

```


```python
import torch
import torch.optim as optim
from tqdm import tqdm

def evaluate(model, tasks, batch_size):
    """
    在 test set 上评估模型性能
    tasks: dict，键是任务名，值是 (pos_acts, neg_acts)
    """
    model.eval()
    all_results = {}
    with torch.no_grad():
        for dataset_name, (pos_acts, neg_acts) in tasks.items():
            min_samples = min(pos_acts.shape[0], neg_acts.shape[0])
            if min_samples == 0:
                all_results[dataset_name] = None
                continue

            # 平衡采样
            pos_indices = torch.randperm(pos_acts.shape[0])[:min_samples]
            neg_indices = torch.randperm(neg_acts.shape[0])[:min_samples]
            
            X = torch.cat([pos_acts[pos_indices], neg_acts[neg_indices]], dim=0)
            y = torch.cat([torch.ones(min_samples), torch.zeros(min_samples)], dim=0)
            
            adjusted, gates = model(X)
            adjusted = normalize_activations(X, adjusted)
            
            # 简单分类指标: 用 gates 的平均值作为预测
            preds = (gates.mean(dim=1) > 0.5).float()
            acc = (preds == y).float().mean().item()
            all_results[dataset_name] = acc
    model.train()
    return all_results


def train_multi_task_steering(train_tasks, test_tasks, num_attributes, batch_size, epochs, lr, sigma,
                              lambda_mmd, lambda_sparse, lambda_ortho, lambda_pos, save_path):
    """
    train_tasks: dict[dataset_name] = (pos_acts, neg_acts)  用于训练
    test_tasks: dict[dataset_name] = (pos_acts, neg_acts)   用于评估
    """

    # 1. 初始化转向模块
    input_dim = list(train_tasks.values())[0][0].shape[1]  
    model = SteeringModule(input_dim, num_attributes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    task_names = list(train_tasks.keys())

    # 2. 开始训练
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        epoch_loss = 0
        num_batches = 0

        # 平衡采样大小
        min_samples = min(min(train_tasks[task][0].shape[0], train_tasks[task][1].shape[0]) for task in train_tasks)
        effective_batch_size = min(batch_size // (2 * num_attributes), min_samples)
        if effective_batch_size < 1:
            effective_batch_size = 1

        # === 遍历 batch ===
        for batch_idx in range(0, min_samples, effective_batch_size):
            batch_activations, batch_labels, batch_task_indices = [], [], []
            
            for t, task_name in enumerate(task_names):
                pos_acts, neg_acts = train_tasks[task_name]
                pos_indices = torch.randperm(pos_acts.shape[0])[:effective_batch_size]
                neg_indices = torch.randperm(neg_acts.shape[0])[:effective_batch_size]
                
                pos_batch = pos_acts[pos_indices]
                neg_batch = neg_acts[neg_indices]
                
                batch_activations.append(pos_batch)
                batch_activations.append(neg_batch)
                batch_labels.extend([1] * effective_batch_size)
                batch_labels.extend([0] * effective_batch_size)
                batch_task_indices.extend([t] * effective_batch_size * 2)

            batch_activations = torch.cat(batch_activations, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
            batch_task_indices = torch.tensor(batch_task_indices, dtype=torch.long)

            adjusted_acts, gates = model(batch_activations)
            adjusted_acts = normalize_activations(batch_activations, adjusted_acts)

            # === 损失函数 ===
            total_loss = 0
            # (1) MMD 损失
            loss_mmd = 0
            for t, task_name in enumerate(task_names):
                task_mask = batch_task_indices == t
                if task_mask.sum() > 0:
                    task_acts = adjusted_acts[task_mask]
                    task_lbls = batch_labels[task_mask]
                    pos_mask = task_lbls == 1
                    neg_mask = task_lbls == 0
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_adjusted = task_acts[pos_mask]
                        neg_adjusted = task_acts[neg_mask]
                        original_pos = train_tasks[task_name][0]
                        sample_indices = torch.randperm(original_pos.shape[0])[:min(pos_adjusted.shape[0], original_pos.shape[0])]
                        original_pos_sample = original_pos[sample_indices]
                        loss_mmd += compute_mmd(neg_adjusted, original_pos_sample, sigma)
            loss_mmd = loss_mmd / num_attributes

            # (2) 稀疏性损失
            neg_mask = batch_labels == 0
            loss_sparse = sparsity_loss(gates[neg_mask]) if neg_mask.sum() > 0 else torch.tensor(0.0)

            # (3) 保留损失
            pos_mask = batch_labels == 1
            loss_pos = preservation_loss(gates[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0)

            # (4) 正交损失
            loss_ortho = orthogonality_loss([sv for sv in model.steering_vectors])

            # 合并损失
            batch_loss = (lambda_mmd * loss_mmd +
                          lambda_sparse * loss_sparse +
                          lambda_ortho * loss_ortho +
                          lambda_pos * loss_pos)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += batch_loss.item()
            num_batches += 1

        # === 每10个epoch输出loss和测试集结果 ===
        if epoch % 10 == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch}, Average Loss: {avg_loss:.4f}")
            results = evaluate(model, test_tasks, batch_size)
            for ds, acc in results.items():
                print(f"  [Test] {ds}: {acc:.4f}" if acc is not None else f"  [Test] {ds}: No samples")

    # 3. 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_attributes': num_attributes,
        'task_names': task_names
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    return model

```



```python
import numpy as np
import torch
from sklearn.model_selection import train_test_split

tasks = {}
for dataset_name in datasets:
    # 加载标签和特征
    labels = np.load(f'../features/{args.model_name}_{dataset_name}_labels.npy')
    all_layer_wise_activations = np.load(f'../features/{args.model_name}_{dataset_name}_layer_wise.npy')

    # 转 tensor
    acts = torch.tensor(all_layer_wise_activations, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 正负例划分
    pos_acts = acts[labels == 1]
    neg_acts = acts[labels == 0]

    # train/test 划分（8:2 或 7:3 都行，这里用 8:2）
    pos_train, pos_test = train_test_split(pos_acts, test_size=0.2, random_state=42)
    neg_train, neg_test = train_test_split(neg_acts, test_size=0.2, random_state=42)

    # 存储
    tasks[dataset_name] = {
        "train": (pos_train, neg_train),
        "test": (pos_test, neg_test)
    }

    # 打印统计信息
    print(f"Dataset {dataset_name}:")
    print(f"  Train -> {pos_train.shape[0]} positive, {neg_train.shape[0]} negative")
    print(f"  Test  -> {pos_test.shape[0]} positive, {neg_test.shape[0]} negative")

```