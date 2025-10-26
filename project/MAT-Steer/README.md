# **基于目标干预的语言模型多属性引导（MAT-Steer）**

作者：[Duy Nguyen](https://duykhuongnguyen.github.io/)、[Archiki Prasad](https://archiki.github.io/)、[Elias Stengel-Eskin](https://esteng.github.io/)、[Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## **概述**
本仓库提供了**“基于目标干预的语言模型多属性引导（MAT-Steer）”** 方法的实现代码。MAT-Steer支持对语言模型进行跨多属性的选择性token级干预。


## **环境安装**
### 1. 配置环境
```bash
conda env create -f environment.yaml  # 从yaml文件创建conda环境
conda activate iti  # 激活名为iti的环境
python -m ipykernel install --user --name iti --display-name "iti"  # 安装ipykernel内核
```

### 2. 创建必要目录
```bash
mkdir -p features  # 创建特征存储目录
mkdir -p validation/checkpoints  # 创建验证用模型 checkpoint 存储目录
mkdir -p validation/results_dump/summary_dump/test  # 创建测试结果汇总存储目录
mkdir -p validation/results_dump/summary_dump/val  # 创建验证结果汇总存储目录
mkdir -p validation/answer_dump/summary_dump/test  # 创建测试答案汇总存储目录
mkdir -p validation/answer_dump/summary_dump/val  # 创建验证答案汇总存储目录
```


## **运行MAT-Steer**
### **1. 提取模型激活值**
进入`get_activations`目录，提取指定层的最后一个token的激活值：
```bash
CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name truthfulqa --layer 14
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python get_activations.py --model_name llama3.1_8B --dataset_name toxigen --layer 14
CUDA_VISIBLE_DEVICES=2 python get_activations.py --model_name llama3.1_8B --dataset_name bbq --layer 14
=======
CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name toxigen --layer 14
CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name bbq --layer 14
>>>>>>> 4978d2d (2025-10-26 update)
```
（注：`CUDA_VISIBLE_DEVICES=0`指定使用第1块GPU；`--dataset_name`后可替换为目标数据集，如truthfulqa、toxigen、bbq；`--layer 14`指定提取第14层的激活值）


### **2. 训练MAT-Steer模型**
进入`validation`目录，训练多属性引导向量：
```bash
python steering.py \
 --model_name llama3.1_8B \  # 基础模型名称
 --layer 14 \  # 干预目标层（需与激活值提取层一致）
 --save_path checkpoints/llama3.1_8B_L14_mat_steer.pt \  # 模型checkpoint保存路径
 --batch_size 96 \  # 批处理大小
 --epochs 100 \  # 训练轮次
 --lr 0.001 \  # 学习率
 --sigma 2.0 \  # 高斯核参数（用于MMD损失）
 --lambda_mmd 1.0 \  # MMD（最大均值差异）损失权重
 --lambda_sparse 0.9 \  # 稀疏性损失权重
 --lambda_ortho 0.1 \  # 正交性损失权重
 --lambda_pos 0.9  # 位置约束损失权重
```

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0
=======

>>>>>>> 4978d2d (2025-10-26 update)
### **3. 评估MAT-Steer模型**
在TruthfulQA数据集上评估训练好的模型：
```bash
python run_mat_eval.py \
 --model_name llama3.1_8B \  # 基础模型名称
 --checkpoint checkpoints/llama3.1_8B_L14_mat_steer.pt \  # 训练好的MAT-Steer checkpoint路径
 --layer 14 \  # 干预目标层
 --instruction_prompt default \  # 使用默认指令提示词
 --baseline  # 同时运行基线模型（无干预）作为对比
```


### **4. 在推理时应用目标干预**
若要在自定义代码中结合pyvene使用MAT-Steer进行推理时干预，可参考以下示例：
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interveners import MATIntervener, create_mat_pyvene_config
import pyvene as pv

# 加载基础模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    torch_dtype=torch.float16,  # 使用FP16精度加载模型
    device_map="auto"  # 自动分配设备（GPU/CPU）
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# 加载MAT-Steer的checkpoint
mat_intervener = MATIntervener.load_from_checkpoint(
    "checkpoints/llama3.1_8B_L14_mat_steer.pt", 
    multiplier=1.0  # 干预强度乘数（1.0为默认强度）
)

# 为第14层创建干预配置
pv_config = create_mat_pyvene_config([14], mat_intervener)

# 用干预配置包装基础模型
intervenable_model = pv.IntervenableModel(pv_config, model)

# 使用带MAT-Steer干预的模型进行生成
inputs = tokenizer("法国的首都是什么？", return_tensors="pt")  # 输入文本编码
outputs = intervenable_model.generate(**inputs, max_new_tokens=50)  # 生成文本（最多新增50个token）
```


## **引用**
如果您觉得本工作对您的研究有帮助，请考虑引用我们的论文：
```bash
@article{nguyen2025multi,
    title={Multi-Attribute Steering of Language Models via Targeted Intervention},
    author={Nguyen, Duy and Prasad, Archiki and Stengel-Eskin, Elias and Bansal, Mohit},
    journal={arXiv preprint arXiv:2502.12446},
    year={2025}
}
```


### 关键术语补充说明
- **激活值（Activations）**：指神经网络中各层在处理输入后产生的中间输出，是模型编码输入信息的核心表征，干预激活值可直接调节模型行为。
- **Token级干预（Token-level Intervention）**：针对输入序列中单个token对应的激活值进行修改，实现更精细的局部控制，而非对整个序列的激活值做全局调整。
- **Checkpoint**：模型训练过程中保存的权重文件，包含训练后的参数状态，可用于后续推理或继续训练。
- **Pyvene**：一个用于对语言模型进行激活值干预的工具库，支持灵活定义和应用各类干预策略。