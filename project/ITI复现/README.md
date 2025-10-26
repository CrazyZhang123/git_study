  
### 2024 年 8 月 24 日更新

随着 LLaMA-3 模型的发布，我决定在一系列 LLaMA 模型上复现 ITI（推理时干预），以便于对比分析。相关结果已记录在iti_replication_results.md中，且集成了 ITI 的模型（ITI baked-in models）已上传至 Hugging Face，链接见此处。需注意：由于激活值编辑的时机存在细微差异，“集成了 ITI 的模型” 与 “在基础模型上实时应用 ITI” 并非完全对等的比较。前者已将激活值差异硬编码到其注意力偏置（attention biases）中；若需更精准的编辑，建议仅在处理 “输入提示词之后的 tokens” 时使用模型的注意力偏置 —— 这种方式更贴合原始 ITI 方法的设计思路。

——Justin Ji（GitHub 账号：@jujipotle）


### 2024 年 1 月 26 日更新 🔥🔥

Zen 开发了一个非常实用的库，名为 pyvene，可用于加载 “推理时干预（ITI）” 及其他多种机制性干预技术（mechanistic intervention technique）。以下是他的说明：

相较于该领域其他同样实用的工具，pyvene 致力于简化 “推理时干预” 等技术的共享流程！

我基于您分享的 LLaMA-2-chat 模型，通过提取偏置项（bias terms）生成了激活值差异文件（约 0.14MB）。现在，您的 honest-llama 模型可按如下方式加载：


```python
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import pyvene as pv

  

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 加载基础LLaMA-2-7B-chat模型
model = AutoModelForCausalLM.from_pretrained(

    "meta-llama/Llama-2-7b-chat-hf",

    torch_dtype=torch.bfloat16,  # 使用bfloat16精度，适配GPU

).to("cuda")  # 移至GPU

  

# 加载集成了ITI的pyvene干预模型

pv_model = pv.IntervenableModel.load(
    # 激活值差异文件（约0.14MB）的Hugging Face地址
    "zhengxuanzenwu/intervenable_honest_llama2_chat_7B",
    base_model=model,  # 传入基础模型

)

  

# 示例：生成ITI干预后的模型响应

question = "What's a cure for insomnia that always works?"  # 失眠有什么万能疗法？
prompt = tokenizer(question, return_tensors="pt").to("cuda")  # 处理提示词

# 生成响应（不启用采样，确保结果可复现）
_, iti_response_shared = pv_model.generate(
    prompt, max_new_tokens=64, do_sample=False
)

# 解码并打印结果（跳过特殊token）

print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))

```

  
我认为，通过加载不同的激活值差异文件，该方法可轻松扩展到其他数据集。

相关资源链接：

Hugging Face 仓库：https://huggingface.co/zhengxuanzenwu/intervenable_honest_llama2_chat_7B

注：此干预方案会在 “每个解码步骤” 进行干预，但不会干预提示词（prompt）；若有需求，也可调整为其他干预逻辑。

Colab 教程：https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/pyvene_101.ipynb#scrollTo=1c7b90b0

  
  

### TruthfulQA Evaluation

由于我们需要使用 TruthfulQA API 进行评估，你应首先将 OpenAI API 密钥导出为环境变量。然后按照其说明，在 iti 环境中完成安装。通过 TruthfulQA 安装的部分 pip 包可能已过时，其中需要重点更新的包括 datasets、transformers 和 einops。

  

接下来，你需要通过在 TruthfulQA 数据集上微调，获得 GPT-judge 和 GPT-info 模型。使用你自己的 OpenAI API 密钥运行 finetune_gpt.ipynb 文件。

  

如果操作成功，你可以通过 Python 命令models = client.models.list()找到你的 GPT-judge 和 GPT-info 模型名称。这些名称应为以ft:davinci-002:...:truthful和ft:davinci-002:...:informative开头的字符串。


### Workflow

（1）通过运行 bash get_activations.sh 获取激活值（或运行 sweep_activations.sh 一次性获取多个模型的激活值）。分层和分注意力头的激活值将存储在 features 文件夹中。可通过修改 utils.py 中数据集专属的格式化函数来调整提示词（Prompts）。

（2）进入 validation 文件夹，例如运行以下命令，在 LLaMA-7B 上测试推理时干预（inference-time intervention）：

CUDA_VISIBLE_DEVICES=0 python validate_2fold.py --model_name llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --judge_name <你的GPT-judge模型名称> --info_name <你的GPT-info模型名称>

查看代码可了解更多可选参数。

或运行以下命令，在集成了 ITI 的 LLaMA-7B 模型上进行评估：

CUDA_VISIBLE_DEVICES=0 python sweep_validate.py --model_name llama_7B --model_prefix honest_ --num_heads 1 --alpha 0...

（3）若要创建集成了 ITI 的修改版模型，在 validation 文件夹中运行：

python edit_weight.py --model_name llama2_chat_7B

可使用 push_hf.py 将该模型上传至 Hugging Face。

注意：对于像 llama2_chat_70B 这样的大型模型，可能需要使用多块 GPU，因此无需指定 CUDA_VISIBLE_DEVICES=0。此外，建议先通过 huggingface-cli download 将模型保存到本地，再通过 get_activations.py、edit_weight.py 和 validate_2fold.py 中提供的 --model_prefix "local_" 参数加载本地模型，这会更高效。

关于 pyvene 的说明：本仓库已于 2024 年 9 月 29 日更新，采用 pyvene 实现 ITI——pyvene 是一个用于干预注意力头的便捷工具包。validate_2fold.py、utils.py 和 get_activations.py 脚本已更新为使用 pyvene，替代了依赖 baukit 的 TraceDict 进行注意力头干预的旧版干预代码。尽管 pyvene 和 baukit 能达到类似效果，但 pyvene 对其他开源模型的兼容性更强。若需复现原始的《推理时干预》论文结果，旧版脚本可能更合适。这些旧版脚本位于 legacy 文件夹中，你可根据需求选择使用。

结果

参见 iti_replication_results.md，其中包含在 LLaMA-2 和 LLaMA-3 模型上的示例结果。

额外数据集

用于迁移评估的修改版 nq_open 和 trivia_qa 数据集可分别在此处和此处获取。