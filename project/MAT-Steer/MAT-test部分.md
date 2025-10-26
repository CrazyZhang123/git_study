
- 解决seed的问题
- gates_biases的维度问题，转换成(3,)方便广播
- 干预后的解码问题
- 改进之前的代码，to('cuda')加速推理效率
- 改进interveners.py代码

## MAT-test.py
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interveners import MATIntervener, create_mat_pyvene_config
import pyvene as pv
import random
import os
import numpy as np

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# 定义支持的Hugging Face模型名称和路径映射字典
HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    # 'llama_7B': 'huggyllama/llama-7b',
    # 'alpaca_7B': 'circulus/alpaca-7b', 
    # 'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    # 'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    # 'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    # 'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    # 'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    # 'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.1_8B': 'Meta-Llama-3.1-8B',
    # 'llama3.1_8B_chat': 'mathewhe/Llama-3.1-8B-Chat',
    # 'qwen2.5_7B': 'Qwen/Qwen2.5-7B',
    'llama2_chat_7B': 'Llama-2-7b-chat-hf', 
    'llama2_13B': 'Llama-2-13b-hf', 
    'llama3_8B': 'Meta-Llama-3-8B',
    'llama3_8B_instruct': 'Llama-3-8B-Instruct',
    'llama3.1_8B': 'Meta-Llama-3.1-8B',
    'qwen2.5_7B': 'Qwen/Qwen2.5-7B'
}
# # 修改：为所有模型路径添加前缀，指向本地存储位置
for i in HF_NAMES.keys():
    HF_NAMES[i] = '/webdav/Storage(default)/MyData/llms/' + HF_NAMES[i]
# seed everything
seed_everything(520)
model_file = HF_NAMES['llama3.1_8B']
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_file, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_file)

# Load MAT-Steer checkpoint
mat_intervener = MATIntervener.load_from_checkpoint("./validation/checkpoints/llama3.1_8B_L14_mat_steer.pt",
                                                     multiplier=1.0
                                                     ) # 加载到 GPU
# 手动移动 MATIntervener 的内部张量到 CUDA 设备，并转换为 float16
device = 'cuda'  # 或 'cuda:0' 如果指定 GPU
dtype = torch.float16

mat_intervener.gates_weights = mat_intervener.gates_weights.to(device=device, dtype=dtype)
# 删掉(3,1)的1维度变成 (3)
mat_intervener.gates_biases = mat_intervener.gates_biases.squeeze(-1).to(device=device, dtype=dtype)
mat_intervener.steering_vectors = mat_intervener.steering_vectors.to(device=device, dtype=dtype)  # 假设有 steering_vectors 属性
print(f"gates_weights: {mat_intervener.gates_weights.shape}")
print(f"gates_biases: {mat_intervener.gates_biases.shape}")
print(f"steering_vectors: {mat_intervener.steering_vectors.shape}")
# print(dir(mat_intervener))  # 列出所有属性，确认张量属性
# print(mat_intervener.gates_weights.dtype, mat_intervener.gates_weights.device)  # 检查 dtype 和 device

# Create intervention config for layer 14
pv_config = create_mat_pyvene_config([14], mat_intervener)

# Wrap model with interventions
intervenable_model = pv.IntervenableModel(pv_config, model)

# Use the model for generation with MAT-Steer interventions
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to('cuda')
# outputs = intervenable_model.generate(**inputs, base=model,max_new_tokens=50)
outputs = intervenable_model.generate(inputs,max_new_tokens=50,output_original_output=True)
# outputs = intervenable_model.generate(inputs,max_new_tokens=500,output_original_output=True)

# %%
# Decode and print outputs（outputs 是 (base_outputs, counterfactual_outputs) 元组）
print("Original output:", tokenizer.decode(outputs[0][0], skip_special_tokens=True))
print("Intervened output:", tokenizer.decode(outputs[1][0], skip_special_tokens=True))

# 原始模型输出
origin_outputs = model.generate(**inputs,max_length=50)
print("Origin output:", tokenizer.decode(origin_outputs[0], skip_special_tokens=True))

```

## intervener.py

- 增加对模型的dtype处理，以及迁移到GPU上。
- 

```python
class MATIntervener():
# ...
def __call__(self, b, s):
        """Apply MAT-Steer intervention.
        
        Args:
            b: input tensor of shape (batch_size, seq_len, hidden_dim)
            s: additional state (unused)
        """
        # Extract the last token activation
        x = b[0, -1]  # (hidden_dim,)
        self.states.append(x.detach().clone())
        
        # Move tensors to the same device as input
        device = x.device
        # Convert weights and biases to the same device and dtype as x
        # 添加 , dtype=x.dtype
        steering_vectors = self.steering_vectors.to(device, dtype=x.dtype)
        gates_weights = self.gates_weights.to(device, dtype=x.dtype)
        gates_biases = self.gates_biases.to(device, dtype=x.dtype)
        
                # 修正 gates_biases 形状
        if gates_biases.dim() > 1:
            gates_biases = gates_biases.squeeze(-1)  # 从 (num_attributes, 1) 转换为 (num_attributes,)

        # Compute gates: g_t = sigmoid(W_t @ x + b_t) for each attribute t
        gates = torch.sigmoid(torch.matmul(gates_weights, x) + gates_biases)  # (num_attributes,)
        
        # Compute steering delta: sum_t g_t * v_t
        delta = torch.sum(gates.unsqueeze(1) * steering_vectors, dim=0)  # (hidden_dim,)
        
        # Apply intervention with global multiplier
        intervention = self.multiplier * delta
        self.actions.append(intervention.detach().clone())
        
        # Apply intervention
        x_adjusted = x + intervention
        
        # Preserve original norm if requested
        if self.layer_norm_preserve:
            original_norm = torch.norm(x, p=2)
            adjusted_norm = torch.norm(x_adjusted, p=2)
            if adjusted_norm > 1e-8:  # Avoid division by zero
                x_adjusted = x_adjusted * (original_norm / adjusted_norm)
        
        # Update the tensor in-place
        b[0, -1] = x_adjusted
        
        return b
```

