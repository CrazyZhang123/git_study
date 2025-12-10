# Qwen2.5-7B-Instruct LoRA 微调代码详解

本文档从功能和语法层面详细解释 Qwen2.5-7B-Instruct 模型的 LoRA 微调代码。

---

## 一、环境准备与数据加载

### 1.1 导入必要的库

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
```

**功能说明：**
- `datasets.Dataset`: Hugging Face 的数据集类，用于高效处理训练数据
- `pandas`: 用于读取和处理 JSON 数据
- `transformers`: Hugging Face 的 Transformers 库，提供模型、分词器和训练工具
  - `AutoTokenizer`: 自动加载分词器
  - `AutoModelForCausalLM`: 自动加载因果语言模型
  - `DataCollatorForSeq2Seq`: 序列到序列任务的数据整理器
  - `TrainingArguments`: 训练参数配置
  - `Trainer`: 训练器类

**语法要点：**
- 使用 `from ... import ...` 导入特定模块，避免命名空间污染

---

### 1.2 加载数据集

```python
# 将JSON文件转换为CSV文件
df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)
```

**功能说明：**
- 从 JSON 文件读取数据并转换为 pandas DataFrame
- 将 DataFrame 转换为 Hugging Face Dataset 格式，便于后续处理

**语法要点：**
- `pd.read_json()`: 读取 JSON 文件，返回 DataFrame
- `Dataset.from_pandas()`: 将 pandas DataFrame 转换为 Hugging Face Dataset
- 数据格式通常包含 `instruction`、`input`、`output` 字段

**数据预览：**
```python
ds[:3]  # 查看前3条数据
```

---

## 二、数据预处理

### 2.1 加载分词器

```python
tokenizer = AutoTokenizer.from_pretrained(
    '/mnt/public/models/Qwen2.5-7B-Instruct', 
    use_fast=False, 
    trust_remote_code=True
)
```

**功能说明：**
- 加载 Qwen2.5-7B-Instruct 模型的分词器
- 分词器负责将文本转换为模型可理解的 token ID

**参数详解：**
- `from_pretrained()`: 从预训练模型路径加载分词器
- `use_fast=False`: 
  - `False`: 使用 Python 实现的分词器（更稳定，兼容性好）
  - `True`: 使用 Rust 实现的 fast tokenizer（速度更快，但可能有兼容性问题）
- `trust_remote_code=True`: 允许执行自定义代码（Qwen 模型需要）

**语法要点：**
- 函数调用时参数可以分行书写，提高可读性

---

### 2.2 数据预处理函数

```python
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    
    # 构建指令部分（system + user）
    instruction = tokenizer(
        f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n", 
        add_special_tokens=False
    )
    
    # 构建回答部分
    response = tokenizer(
        f"{example['output']}", 
        add_special_tokens=False
    )
    
    # 拼接输入序列
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    
    # 构建标签（只对回答部分计算损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

**功能说明：**
- 将原始数据转换为模型训练所需的格式
- 构建 Qwen 模型的对话格式（使用 `<|im_start|>` 和 `<|im_end|>` 标记）
- 只对 assistant 的回答部分计算损失（instruction 部分标签为 -100）

**语法要点：**

1. **f-string 格式化字符串：**
   ```python
   f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>"
   ```
   - 使用 `f""` 前缀，`{}` 内可以嵌入变量或表达式

2. **字典访问：**
   ```python
   example['instruction']  # 访问字典的键
   instruction["input_ids"]  # 访问字典的键
   ```

3. **列表拼接：**
   ```python
   instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
   ```
   - 使用 `+` 连接列表
   - `[tokenizer.pad_token_id]` 创建单元素列表

4. **列表生成式：**
   ```python
   [-100] * len(instruction["input_ids"])
   ```
   - `*` 操作符用于列表重复
   - 生成与 instruction 长度相同的 -100 列表

5. **条件判断与切片：**
   ```python
   if len(input_ids) > MAX_LENGTH:
       input_ids = input_ids[:MAX_LENGTH]
   ```
   - `len()` 获取列表长度
   - `[:MAX_LENGTH]` 列表切片，取前 MAX_LENGTH 个元素

6. **返回值：**
   ```python
   return {
       "input_ids": input_ids,
       "attention_mask": attention_mask,
       "labels": labels
    }
   ```
   - 返回字典，包含训练所需的三个字段

**关键概念：**
- `input_ids`: 文本转换后的 token ID 序列
- `attention_mask`: 注意力掩码，1 表示有效 token，0 表示 padding
- `labels`: 训练标签，-100 表示不计算损失，其他值表示要预测的 token ID
- `MAX_LENGTH = 384`: 最大序列长度，防止超出模型限制

---

### 2.3 应用预处理函数

```python
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
```

**功能说明：**
- 对数据集中的每条数据应用 `process_func` 函数
- 移除原始列，只保留处理后的 `input_ids`、`attention_mask`、`labels`

**语法要点：**
- `map()`: Dataset 的方法，对每条数据应用函数
- `remove_columns`: 移除指定的列，避免数据冗余

---

### 2.4 验证处理结果

```python
# 查看编码后的完整序列
tokenizer.decode(tokenized_id[0]['input_ids'])

# 查看标签对应的文本（过滤掉 -100）
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
```

**功能说明：**
- 验证数据预处理是否正确
- 检查编码和解码是否一致

**语法要点：**
- `tokenizer.decode()`: 将 token ID 转换回文本
- `filter(lambda x: x != -100, ...)`: 过滤掉值为 -100 的元素
- `list()`: 将 filter 对象转换为列表

---

## 三、模型加载与配置

### 3.1 加载预训练模型

```python
import torch

model = AutoModelForCausalLM.from_pretrained(
    '/mnt/public/models/Qwen2.5-7B-Instruct', 
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

**功能说明：**
- 加载 Qwen2.5-7B-Instruct 预训练模型
- 自动分配到可用 GPU
- 使用 bfloat16 精度以节省显存

**参数详解：**
- `device_map="auto"`: 自动将模型分配到多个 GPU（如果可用）
- `torch_dtype=torch.bfloat16`: 
  - 使用半精度浮点数，减少显存占用
  - bfloat16 比 float16 更稳定，适合训练

**语法要点：**
- `import torch`: 导入 PyTorch 库
- `torch.bfloat16`: PyTorch 的数据类型常量

---

### 3.2 模型训练配置

```python
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
model.config.use_cache = False      # 防止 checkpoint 与多卡冲突
```

**功能说明：**
- `enable_input_require_grads()`: 启用输入梯度计算（梯度检查点需要）
- `use_cache = False`: 禁用 KV 缓存，节省显存，避免多卡训练时的冲突

**语法要点：**
- 方法调用：`model.enable_input_require_grads()`
- 属性赋值：`model.config.use_cache = False`

---

### 3.3 检查模型数据类型

```python
model.dtype  # 输出: torch.bfloat16
```

**功能说明：**
- 验证模型的数据类型是否正确

---

## 四、LoRA 配置

### 4.1 导入 PEFT 库

```python
from peft import LoraConfig, TaskType, get_peft_model
```

**功能说明：**
- `LoraConfig`: LoRA 配置类
- `TaskType`: 任务类型枚举
- `get_peft_model`: 将基础模型转换为 PEFT 模型

---

### 4.2 配置 LoRA 参数

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,                  # Lora 秩
    lora_alpha=32,        # Lora alpha，具体作用参见 Lora 原理
    lora_dropout=0.1      # Dropout 比例
)
```

**功能说明：**
- 配置 LoRA（Low-Rank Adaptation）参数
- LoRA 是一种参数高效的微调方法，只训练少量参数

**参数详解：**
- `task_type=TaskType.CAUSAL_LM`: 任务类型为因果语言模型
- `target_modules`: 要应用 LoRA 的模块列表
  - `q_proj`, `k_proj`, `v_proj`: 注意力机制的查询、键、值投影层
  - `o_proj`: 注意力输出投影层
  - `gate_proj`, `up_proj`, `down_proj`: MLP 层的投影
- `inference_mode=False`: 训练模式（True 为推理模式）
- `r=8`: LoRA 的秩（rank），控制低秩矩阵的维度，越小参数越少
- `lora_alpha=32`: LoRA 的缩放因子，通常设为 r 的倍数
- `lora_dropout=0.1`: Dropout 比例，防止过拟合

**语法要点：**
- 使用关键字参数调用函数，提高可读性
- 列表作为参数值：`["q_proj", "k_proj", ...]`

---

### 4.3 应用 LoRA

```python
model = get_peft_model(model, config)
```

**功能说明：**
- 将基础模型包装为 PEFT 模型
- 只训练 LoRA 参数，冻结原始模型参数

**语法要点：**
- 函数调用返回新对象，需要重新赋值

---

### 4.4 查看可训练参数

```python
model.print_trainable_parameters()
# 输出: trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.2643
```

**功能说明：**
- 显示可训练参数数量和总参数数量
- 验证 LoRA 是否生效（可训练参数应该远少于总参数）

**结果解读：**
- 可训练参数：20,185,088（约 2000 万）
- 总参数：7,635,801,600（约 76 亿）
- 可训练比例：0.2643%（仅训练 0.26% 的参数）

---

## 五、训练配置与执行

### 5.1 配置训练参数

```python
args = TrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100, 
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False
)
```

**功能说明：**
- 配置训练的超参数和设置

**参数详解：**
- `output_dir`: 模型保存路径
- `per_device_train_batch_size=4`: 每个设备的批次大小
- `gradient_accumulation_steps=4`: 梯度累积步数
  - 有效批次大小 = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`
  - 这里有效批次大小 = 4 × 4 = 16（单卡）
- `logging_steps=10`: 每 10 步记录一次日志
- `num_train_epochs=3`: 训练 3 个 epoch
- `save_steps=100`: 每 100 步保存一次模型
- `learning_rate=1e-4`: 学习率 0.0001
- `save_on_each_node=True`: 多节点训练时，每个节点都保存
- `gradient_checkpointing=False`: 不使用梯度检查点（可节省显存但增加计算时间）

**语法要点：**
- 科学计数法：`1e-4` 表示 0.0001
- 字符串路径：`"./output/Qwen2.5_instruct_lora"`

---

### 5.2 创建训练器

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
```

**功能说明：**
- 创建 Trainer 对象，封装训练逻辑

**参数详解：**
- `model`: 要训练的模型
- `args`: 训练参数
- `train_dataset`: 训练数据集
- `data_collator`: 数据整理器
  - `DataCollatorForSeq2Seq`: 序列到序列任务的数据整理器
  - `padding=True`: 自动填充到相同长度

**语法要点：**
- 对象实例化：使用类名和参数创建对象
- 嵌套函数调用：`DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)`

---

### 5.3 开始训练

```python
trainer.train()
```

**功能说明：**
- 执行训练过程
- 自动处理前向传播、反向传播、优化器更新等

**语法要点：**
- 方法调用：`trainer.train()`
- 无参数方法调用

---

## 六、模型合并与推理

### 6.1 加载模型和 LoRA 权重

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/mnt/public/models/Qwen2.5-7B-Instruct/'
lora_path = './output/Qwen2.5_instruct_lora/checkpoint-100'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    mode_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)
```

**功能说明：**
- 加载基础模型和训练好的 LoRA 权重
- 用于推理测试

**语法要点：**
- 方法链式调用：`.eval()` 设置模型为评估模式
- 路径字符串：使用单引号或双引号定义路径

---

### 6.2 构建输入

```python
prompt = "你是谁？"
inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
).to('cuda')
```

**功能说明：**
- 使用 Qwen 的聊天模板构建输入
- 设置角色和内容

**参数详解：**
- `apply_chat_template()`: 应用聊天模板，自动添加特殊标记
- `add_generation_prompt=True`: 添加生成提示
- `tokenize=True`: 自动分词
- `return_tensors="pt"`: 返回 PyTorch 张量
- `return_dict=True`: 返回字典格式
- `.to('cuda')`: 将张量移动到 GPU

**语法要点：**
- 列表嵌套字典：`[{"role": "user", "content": "..."}, ...]`
- 方法链式调用：`.apply_chat_template(...).to('cuda')`

---

### 6.3 生成回答

```python
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**功能说明：**
- 使用模型生成回答
- 只提取新生成的部分（去掉输入部分）

**参数详解：**
- `gen_kwargs`: 生成参数
  - `max_length=2500`: 最大生成长度
  - `do_sample=True`: 使用采样（而非贪婪解码）
  - `top_k=1`: Top-K 采样，只考虑概率最高的 1 个 token
- `torch.no_grad()`: 禁用梯度计算，节省显存和计算

**语法要点：**

1. **字典解包：**
   ```python
   model.generate(**inputs, **gen_kwargs)
   ```
   - `**` 操作符将字典解包为关键字参数

2. **上下文管理器：**
   ```python
   with torch.no_grad():
       # 代码块
   ```
   - `with` 语句确保在代码块结束后自动恢复梯度计算

3. **张量切片：**
   ```python
   outputs[:, inputs['input_ids'].shape[1]:]
   ```
   - `[:, start:]`: 取所有批次，从 start 位置到末尾
   - `inputs['input_ids'].shape[1]`: 获取输入序列长度

4. **索引访问：**
   ```python
   outputs[0]  # 取第一个批次的结果
   ```

---

## 七、总结

### 7.1 代码流程

1. **数据准备**：加载 JSON 数据，转换为 Dataset 格式
2. **数据预处理**：使用分词器将文本转换为 token ID，构建训练格式
3. **模型加载**：加载预训练模型，配置训练模式
4. **LoRA 配置**：设置 LoRA 参数，只训练少量参数
5. **训练执行**：使用 Trainer 进行训练
6. **模型推理**：加载训练好的模型进行测试

### 7.2 关键技术点

- **LoRA 微调**：参数高效微调，只训练 0.26% 的参数
- **对话格式**：使用 Qwen 的特殊标记构建对话
- **损失计算**：只对 assistant 的回答部分计算损失
- **半精度训练**：使用 bfloat16 节省显存
- **梯度累积**：通过累积梯度模拟更大的批次

### 7.3 语法要点总结

- **f-string**：字符串格式化
- **列表操作**：拼接、切片、生成式
- **字典操作**：访问、解包
- **函数调用**：位置参数、关键字参数
- **方法链式调用**：`.method1().method2()`
- **上下文管理器**：`with` 语句
- **张量操作**：切片、索引、设备移动

---

## 八、常见问题

### Q1: 为什么要设置 `labels` 中 instruction 部分为 -100？
**A:** 在训练时，模型只需要学习生成 assistant 的回答，不需要学习 instruction 部分。将 instruction 部分的标签设为 -100，PyTorch 的交叉熵损失函数会忽略这些位置。

### Q2: `gradient_accumulation_steps` 的作用是什么？
**A:** 当显存不足无法使用大批次时，可以通过梯度累积模拟大批次。例如，`batch_size=4, gradient_accumulation_steps=4` 相当于 `batch_size=16`，但显存占用更小。

### Q3: LoRA 的 `r` 和 `lora_alpha` 如何选择？
**A:** 
- `r` 越大，可训练参数越多，表达能力越强，但可能过拟合
- `lora_alpha` 通常设为 `r` 的 2-4 倍，控制 LoRA 权重的缩放
- 常见组合：`r=8, alpha=32` 或 `r=16, alpha=64`

### Q4: 为什么要使用 `use_fast=False`？
**A:** Qwen 模型使用自定义分词器，fast tokenizer 可能不完全兼容，使用 Python 实现更稳定。

---

**文档结束**

