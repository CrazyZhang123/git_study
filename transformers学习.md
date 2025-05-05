### [第六章：必要的 Pytorch 知识 · Transformers快速入门](https://transformers.run/c2/2021-12-14-transformers-note-3/#dataloaders)

#### 1、worker_init_fn函数解析

```python
from torch.utils.data import get_worker_info

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # 获取当前worker进程中的数据集副本
    overall_start = dataset.start  # 数据集的起始索引
    overall_end = dataset.end  # 数据集的结束索引
    # 计算每个worker需要处理的数据量
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id  # 当前worker的ID
    # 配置每个worker的数据集范围
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
```

1. **获取Worker信息**：

   ```python
   worker_info = get_worker_info()
   ```

   使用`get_worker_info()`函数获取当前worker的相关信息，包括worker的ID、数据集副本、总worker数量等。

2. **获取数据集副本**：

   ```python
   dataset = worker_info.dataset
   ```

   获取当前worker进程中的数据集副本。每个worker都会有一个数据集的副本，这样可以避免多个worker之间对数据集的并发访问问题。

3. **获取数据集的起始和结束索引**：

   ```python
   overall_start = dataset.start
   overall_end = dataset.end
   ```

   获取数据集的起始索引`overall_start`和结束索引`overall_end`。这些索引定义了数据集在整个数据集中的范围。

4. **计算每个Worker的数据量**：

   ```python
   per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
   ```

   计算每个worker需要处理的数据量。这里使用`math.ceil`函数确保每个worker至少处理一部分数据，即使数据不能平均分配。

5. **配置每个Worker的数据集范围**：

   ```python
   dataset.start = overall_start + worker_id * per_worker
   dataset.end = min(dataset.start + per_worker, overall_end)
   ```

   根据worker的ID和每个worker的数据量，配置每个worker的数据集范围。`dataset.start`和`dataset.end`分别表示当前worker处理的数据的起始和结束索引。

#### 示例

假设数据集的总长度为10，有两个worker（`num_workers=2`），则：

- **Worker 0**:
  - `worker_id = 0`
  - `dataset.start = 0 + 0 * 5 = 0`
  - `dataset.end = min(0 + 5, 10) = 5`
  - 处理的数据索引范围为 `[0, 4]`
- **Worker 1**:
  - `worker_id = 1`
  - `dataset.start = 0 + 1 * 5 = 5`
  - `dataset.end = min(5 + 5, 10) = 10`
  - 处理的数据索引范围为 `[5, 9]`

#### 2、构建模型部分

```python
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# 继承自 nn.Module，这是 PyTorch 中所有神经网络模型的基类。
class NeuralNetwork(nn.Module):
# 在构造函数 __init__ 中定义了模型的结构。
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # 使用 nn.Flatten() 将输入数据从多维张量展平为一维向量。
# 例如，如果输入是一个形状为 (batch_size, 28, 28) 的图像张量，经过 flatten 后会变成 (batch_size, 784)。
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            # Dropout 是一种正则化技术，随机丢弃一部分神经元以防止过拟合。
参数 p=0.2 表示每次训练时有 20% 的神经元会被随机丢弃。
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

X = torch.rand(4, 28, 28, device=device)
# 模型前向传播
logits = model(X)
# Softmax 是一种归一化函数，将 logits 转换为概率分布。
# 参数 dim=1 表示对每一行（即每个样本的 10 个类别分数）进行归一化。
# 从输出的维度看，dim=1，是第二个维度的10个元素进行归一化处理。
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab.size())
# argmax 返回张量中最大值的索引。
# 对于每个样本，返回概率最大的类别的索引。
y_pred = pred_probab.argmax(-1)
print(f"Predicted class: {y_pred}")
```

```
Using cpu device
torch.Size([4, 10])
Predicted class: tensor([3, 8, 3, 3])
```

#### 3、优化模型参数

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

learning_rate = 1e-3
batch_size = 64
epochs = 3

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
		# 打印100，200，...起步的这样batch对应的loss和已训练的数据量/总数据量
        if batch % 100 == 0:
            # loss.item()获取标量值，
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(torch.float).sum().item()
            

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
Using cpu device
Epoch 1
-------------------------------
loss: 0.935758  [ 6400/60000]
loss: 0.991128  [12800/60000]
loss: 0.655021  [19200/60000]
loss: 0.938772  [25600/60000]
loss: 0.480326  [32000/60000]
loss: 0.526776  [38400/60000]
loss: 1.046211  [44800/60000]
loss: 0.749002  [51200/60000]
loss: 0.550378  [57600/60000]
Test Error: 
 Accuracy: 83.7%, Avg loss: 0.441249 

Epoch 2
-------------------------------
loss: 0.596351  [ 6400/60000]
loss: 0.614368  [12800/60000]
loss: 0.588207  [19200/60000]
loss: 0.698899  [25600/60000]
loss: 0.433412  [32000/60000]
loss: 0.533789  [38400/60000]
loss: 0.772370  [44800/60000]
loss: 0.486120  [51200/60000]
loss: 0.534202  [57600/60000]
Test Error: 
 Accuracy: 85.4%, Avg loss: 0.396990 

Epoch 3
-------------------------------
loss: 0.547906  [ 6400/60000]
loss: 0.591556  [12800/60000]
loss: 0.537591  [19200/60000]
loss: 0.722009  [25600/60000]
loss: 0.319590  [32000/60000]
loss: 0.504153  [38400/60000]
loss: 0.797246  [44800/60000]
loss: 0.553834  [51200/60000]
loss: 0.400079  [57600/60000]
Test Error: 
 Accuracy: 87.2%, Avg loss: 0.355058 

Done!
```

**注意：**一定要在预测之前调用 `model.eval()` 方法将 dropout 层和 batch normalization 层设置为**评估模式**，否则会产生不一致的预测结果。

##### （1）关于f字符串：{loss:>7f}  [{current:>5d}/{size:>5d}]

```
- 对齐部分 >
<：左对齐。
>：右对齐。
^：居中对齐。

- 字段长度 7 
表示字段宽度为 7 个字符。

- 浮点数/整数 f/d
默认6位小数
```

##### （2）模型评估模型 model.eval() 和 禁用梯度计算

```
model.eval() ：
将模型设置为评估模式。
在评估模式下，某些层（如 Dropout 和 BatchNorm）会切换到推理行为：
Dropout 层不会随机丢弃神经元。
BatchNorm 层会使用训练时计算的统计量（均值和方差）。

torch.no_grad() ：
进入一个上下文环境，在该环境中禁用梯度计算。
这样可以减少内存消耗并加速推理过程，因为测试阶段不需要更新模型参数。
```

##### (3) 计算正确的预测数量

correct += (pred.argmax(dim=-1) == y).type(torch.float).sum().item()

**pred.argmax(dim=-1)** ：

- 对每个样本的预测值 pred 取最大值的索引，表示预测的类别。
  结果是一个形状为 (batch_size,) 的张量，包含每个样本的预测类别。

**(pred.argmax(dim=-1) == y)** ：

- 比较预测类别与真实标签 y，生成一个布尔张量，表示每个样本是否预测正确。

**.type(torch.float)** ：

- 将布尔值转换为浮点数（True -> 1.0，False -> 0.0）。

**.sum().item()** ：

- 对布尔张量求和，得到当前批次中正确预测的样本数量。

**.item()** 提取标量值并累加到 correct 中。

#### 4. 保存及加载模型

在之前的文章中，我们介绍过模型类 `Model` 的保存以及加载方法，但如果我们只是将预训练模型作为一个模块（例如作为编码器），那么最终的完整模型就是一个自定义 Pytorch 模型，它的保存和加载就必须使用 Pytorch 预设的接口。

##### 保存和加载模型权重

Pytorch 模型会将所有参数存储在一个状态字典 (state dictionary) 中，可以通过 `Model.state_dict()` 加载。Pytorch 通过 `torch.save()` 保存模型权重：

```
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

为了加载保存的权重，我们首先需要创建一个结构完全相同的模型实例，然后通过 `Model.load_state_dict()` 函数进行加载：

```
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

##### 保存和加载完整模型

上面存储模型权重的方式虽然可以节省空间，但是加载前需要构建一个结构完全相同的模型实例来承接权重。如果我们希望在存储权重的同时，也一起保存模型结构，就需要将整个模型传给 `torch.save()` ：

```
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model, 'model.pth')
```

这样就可以**直接从保存的文件中加载整个模型（包括权重和结构）**：

```
model = torch.load('model.pth')
```

```
{"sentence1": "还款还清了，为什么花呗账单显示还要还款", "sentence2": "花呗全额还清怎么显示没有还款", "label": "1"}
```