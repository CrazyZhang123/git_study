# Guides

## 1.pyvene

pyvene 是一个开源的 Python 库，用于对 PyTorch 模型的内部状态进行干预操作。在人工智能的多个领域中，干预都是一项重要操作，涵盖模型编辑、引导、鲁棒性增强以及可解释性分析等方向。

pyvene 具备多项特性，能让干预操作更便捷：

- 干预是基础单元，以字典形式定义，因此可本地保存，并作为可序列化对象通过 HuggingFace 平台共享。
- 干预支持组合与自定义：你可以在多个位置执行干预，针对任意神经元集合（或其他粒度层级）进行操作，支持并行或串行执行，还能在生成式语言模型的解码步骤中实施干预等。
- 干预可直接在任意 PyTorch 模型上使用！无需从零开始定义新的模型类，且能轻松对各类架构（循环神经网络 RNN、残差网络 ResNets、卷积神经网络 CNN、Mamba 模型等）进行干预。



## 2.*Wrap* and *intervene*

封装与干预

使用 pyvene 的常规工作流程分为三步：**加载模型、定义干预配置并封装模型**，随后运行经过干预的模型。该过程会返回两部分结果 —— 原始输出与干预后输出，同时还会返回所有你指定需收集的内部激活值。示例如下：

```python
import torch
import pyvene as pv
# 自动加载与指定模型匹配的分词器，加载用于因果语言建模（Causal Language Modeling）的预训练模型。
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1. Load the model
# 加载模型
model_name = "meta-llama/Llama-2-7b-hf" # the HF model you want to intervene on
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  # torch_dtype=torch.bfloat16：使用 bfloat16 数据类型以减少内存占用并加速计算。
   model_name, torch_dtype=torch.bfloat16, device_map="cuda")


# 2. Wrap the model with an intervention config
# 使用 pyvene 包装模型以进行干预
pv_model = pv.IntervenableModel({
   "component": "model.layers[15].mlp.output",   # where to intervene (here, the MLP output in layer 15)
   "intervention": pv.ZeroIntervention           # what intervention to apply (here, zeroing out the activation)
}, model=model)


# 3. Run the intervened model
# orig_outputs：原始模型（未干预）的输出。
# intervened_outputs：干预后模型的输出（第15层 MLP 输出被置零）。
orig_outputs, intervened_outputs = pv_model(
   tokenizer("The capital of Spain is", return_tensors="pt").to('cuda'),
   output_original_output=True # 指示 pv_model 返回两组输出：

)


# 4. Compare outputs
# 比较干预前后的输出
print(intervened_outputs.logits - orig_outputs.logits)
```

#### 解释

1.**pv.IntervenableModel**：

- pyvene 提供的一个类，用于在模型的特定部分进行干预。
- 它将原始模型（model）包装为一个可干预的模型对象。

**干预配置：**

> **"component": "model.layers[15].mlp.output"：**
>
> - 指定干预的目标位置为模型第15层的 MLP（多层感知机）模块的输出。
> - LLaMA 模型由多个 Transformer 层组成，每一层包含注意力机制和 MLP 模块。这里的干预点是第15层 MLP 的输出激活值。
>
> **"intervention": pv.ZeroIntervention：**
>
> - 指定干预类型为 ZeroIntervention，即将目标组件的激活值置为零。

**model=model**：

- 将加载的 LLaMA 模型传递给 pyvene，以便在其上进行干预。

#### 返回

```python
tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.4375,  1.0625,  0.3750,  ..., -0.1562,  0.4844,  0.2969],
         [ 0.0938,  0.1250,  0.1875,  ...,  0.2031,  0.0625,  0.2188],
         [ 0.0000, -0.0625, -0.0312,  ...,  0.0000,  0.0000, -0.0156]]],
      device='cuda:0')
```

## 3.*Share* and *load* from HuggingFace

从 HuggingFace 共享与加载

**pyvene 支持通过 HuggingFace 共享与加载干预方案**

以下代码块可复现论文《推理时干预：从语言模型中获取真实答案》（*Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*）中的 “诚实型 Llama-2 对话”（honest_llama-2 chat）功能。所需加载的激活值文件大小仅约 0.14MB，占用存储空间极小！

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv


# 1. Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
).to("cuda")


# 2. Load intervention from HF and wrap model
# 见下面解释
pv_model = pv.IntervenableModel.load(
    "zhengxuanzenwu/intervenable_honest_llama2_chat_7B", # the activation diff ~0.14MB
    model,
)


# 3. Let's run it!
# 运行干预后的模型
print("llama-2-chat loaded with interventions:")
q = "What's a cure for insomnia that always works?"
# 治疗失眠的万能方法是什么？
# “What's a cure for...” 是常用表达，意为 “…… 的治疗方法是什么”；“that always works” 是定语从句，修饰 “cure”，表示 “总能奏效的、万能的”。
prompt = tokenizer(q, return_tensors="pt").to("cuda")
# 使用干预后的模型生成回答，最多生成 64 个新 token，不使用采样（do_sample=False）
_, iti_response_shared = pv_model.generate(prompt, max_new_tokens=64, do_sample=False)
# 将生成的 token 解码为文本并打印，忽略特殊 token（如 <s> 或 </s>）。
print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))
```

#### 解释

**pv_model = pv.IntervenableModel.load(...)：**

- 从 Hugging Face 加载干预方案 "zhengxuanzenwu/intervenable_honest_llama2_chat_7B"，并将其应用于指定模型。

- "zhengxuanzenwu/intervenable_honest_llama2_chat_7B" 是一个存储在 Hugging Face 上的干预方案，来自论文 *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*。

  该方案包含预计算的激活差异（activation difference），用于在推理时修改模型行为以提高答案的真实性。

- pv.IntervenableModel.load函数：它将基础模型（model）包装为一个可干预模型（pv_model），可以在推理时应用干预。
  - 干预方案可能修改模型某些层的激活值（例如 MLP 或注意力层的输出），以引导模型生成更真实的回答。
  - **加载预训练干预方案避免了从头训练干预的复杂性。**



## 4.*IntervenableModel* is just an *nn.Module*

IntervenableModel 本质上是一个 nn.Module

pyvene wraps PyTorch models in the [`IntervenableModel`](https://stanfordnlp.github.io/pyvene/api/pyvene.models.intervenable_base.IntervenableModel.html#pyvene.models.intervenable_base.IntervenableModel) class. This is just a subclass of `nn.Module`, so you can use it just like any other PyTorch model! For example:

pyvene 将 PyTorch 模型封装在 IntervenableModel 类中。这个类本质上只是 nn.Module 的一个子类，因此你可以像使用其他任何 PyTorch 模型一样使用它！例如：

```python
import torch
import torch.nn as nn
# typing 模块：用于类型注解（如 List、Optional、Dict），提高代码可读性和类型安全性。
from typing import List, Optional, Tuple, Union, Dict

class ModelWithIntervenables(nn.Module):
    def __init__(self):
        super(ModelWithIntervenables, self).__init__()
        # 将预定义的 pv_gpt2（假设是一个 pyvene 包装的 GPT-2 模型）作为模型的一部分。
        self.pv_gpt2 = pv_gpt2
        self.relu = nn.ReLU()
        self.fc = nn.Linear(768, 1)
        # Your other downstream components go here

    def forward(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        _, counterfactual_x = self.pv_gpt2(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces
        )
        return self.fc(self.relu(counterfactual_x.last_hidden_state))
```

## 5.Complex Intervention Schema as an Object

作为对象的复杂干预方案

**pyvene 提供的一个核心抽象概念，便是对干预方案的封装**。尽管抽象化设计能带来友好的用户交互界面，但 pyvene 仍可支持相对复杂的干预方案。以下辅助函数会生成用于 “单个注意力头路径修补” 的方案，该方案可用于复现论文《Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small》（《实际场景中的可解释性：GPT-2 small 模型中间接宾语识别的电路机制》）中的实验：



## 6.因果抽象简明指南：从干预到获得可解释性见解

作者：吴正轩

基础干预虽然有趣，但我们无法系统地得出任何因果结论。为了获得真正的可解释性见解，我们希望以数据驱动的方式测量模型的反事实行为。换句话说，如果模型对您的干预做出系统性反应，那么您就可以开始将网络中的特定区域与高层概念关联起来。我们也将此过程称为与模型内部的对齐搜索过程。

### 利用静态干预理解因果机制

下面是一个更具体的例子：

```python
def add_three_numbers(a, b, c):
    var_x = a + b
    return var_x + c
```

这个函数解决了一个三位数求和问题。假设我们训练了一个神经网络来完美解决这个问题。"我们能在神经网络中找到(a + b)的表示吗？"我们可以使用这个库来回答这个问题。具体来说，我们可以按以下步骤操作：

步骤1：形成可解释性（对齐）假设：我们假设一组神经元N与(a + b)对齐。

步骤2：反事实测试：如果我们的假设正确，那么在示例之间交换神经元N应该会产生预期的反事实行为。例如，(1+2)+3中N的值与(2+3)+4中N的值交换后，输出应该是(2+3)+3或(1+2)+4，具体取决于交换的方向。

步骤3：假设的拒绝采样：多次运行测试，并根据反事实行为匹配情况汇总统计数据。根据结果提出新的假设。

要将上述步骤转换为使用该库的API调用，只需一次调用：

```python
intervenable.eval_alignment(
    train_dataloader=test_dataloader,
    compute_metrics=compute_metrics,
    inputs_collator=inputs_collator
)
```

其中，您需要提供测试数据（基本上是干预数据和您要寻找的反事实行为）以及您的指标函数。该库将尝试根据您在配置中指定的干预来评估对齐情况。

### 利用可训练干预理解因果机制

当神经网络很大时，上述对齐搜索过程可能会很繁琐。对于单个假设的对齐，您基本上需要设置针对不同层和位置的不同干预配置来验证您的假设。我们可以将其转化为一个优化问题，而不是进行这种暴力搜索过程，这还有其他好处，如分布式对齐。

本质上，我们希望训练一种干预，使其具有我们期望的反事实行为。如果我们确实能训练出这样的干预，我们就可以说，具有因果信息的内容存在于被干预的表示中！下面，我们展示一种可训练干预类型——RotatedSpaceIntervention：

```python
class RotatedSpaceIntervention(TrainableIntervention):

    """在旋转空间中的干预。"""
    def forward(self, base, source):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # 交换
        rotated_base[:self.interchange_dim] = rotated_source[:self.interchange_dim]
        # 反向基
        output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
        return output
```

我们不是在原始表示空间中交换激活值，而是首先对它们进行旋转，然后进行交换，接着对被干预的表示进行反旋转。此外，我们尝试使用SGD来学习一种旋转，使我们能够产生预期的反事实行为。如果我们能找到这样的旋转，我们就认为存在对齐。If the cost is between X and Y.ipynb教程通过分布式对齐搜索的高级版本——Boundless DAS涵盖了这一点。最近也有研究指出了进行分布式对齐搜索的潜在局限性。

您现在也可以通过一次API调用来训练您的干预：

```python
intervenable.train_alignment(
    train_dataloader=train_dataloader,
    compute_loss=compute_loss,
    compute_metrics=compute_metrics,
    inputs_collator=inputs_collator
)
```

其中，您需要传入一个可训练的数据集，以及您自定义的损失和指标函数。可训练的干预随后可以保存到您的磁盘上。您还可以使用intervenable.evaluate()根据自定义目标评估您的干预。



# Basic Tutorials

## 1、LMs Generation

语言模型生成干预

You can also intervene the generation call of LMs. Here is a simple example where we try to add a vector into the MLP output when the model decodes.

你也可以在语言模型（LMs）的生成过程中进行干预。以下是一个简单示例：我们在模型解码时，向 MLP 输出层添加一个向量。

```python
import torch
import pyvene as pv

# built-in helper to get tinystore
# 使用内置辅助函数获取 TinyStory 模型
#  调用 pv.create_gpt_neo() 加载一个小型 GPT-Neo 模型（名为 TinyStory），并获取其分词器（tokenizer）和模型对象（tinystory）。第一个返回值（_）通常为配置，此处忽略。
_, tokenizer, tinystory = pv.create_gpt_neo()

# 通过模型的词嵌入层（wte = word token embedding）查找 token ID 为 14628 的词向量。
emb_happy = tinystory.transformer.wte(
    torch.tensor(14628)) 

# 创建一个可干预模型对象 pv_tinystory。这里配置了对每一层（l 从 0 到总层数减一）的 MLP 输出（mlp_output）施加“加法干预”（AdditionIntervention）——即在解码过程中，把指定向量加到 MLP 的输出上。
pv_tinystory = pv.IntervenableModel([{
    "layer": l,
    "component": "mlp_output",
    "intervention_type": pv.AdditionIntervention
    } for l in range(tinystory.config.num_layers)],
    model=tinystory
)
# prompt and generate

# 调用干预模型的 generate 方法，同时生成两个版本的输出：
# unintervened_story：未干预的原始生成结果。
# intervened_story：在每层 MLP 输出上加上 emb_happy * 0.3（缩放后的“快乐”向量）后的干预生成结果。
# source_representations 参数指定了要添加的干预向量（此处为缩放后的“快乐”嵌入）。
prompt = tokenizer(
    "Once upon a time there was", return_tensors="pt")
unintervened_story, intervened_story = pv_tinystory.generate(
    prompt, source_representations=emb_happy*0.3, max_length=256
)

# 将干预后生成的第一个样本（intervened_story[0]）解码为自然语言文本，并跳过特殊标记（如 [PAD]、[EOS] 等），然后打印输出。
print(tokenizer.decode(
    intervened_story[0], 
    skip_special_tokens=True
))
```



intervene on generation with source example passed in. The result will be slightly different since we no longer have a static vector to be added in; it is layerwise addition.

**在生成过程中传入一个源示例进行干预。结果会略有不同，因为我们不再添加一个静态向量，而是进行逐层加法干预。**

```python
import torch
import pyvene as pv

# built-in helper to get tinystore
_, tokenizer, tinystory = pv.create_gpt_neo()

# 定义一个自定义干预函数 pv_patcher：
# 将源表征（s） 乘以 0.1 后，加到基础表征（b） 上。
# 这意味着：我们不是加一个固定的“快乐向量”，而是从另一个输入（源示例）中动态提取表征，并逐层加到当前生成过程中
def pv_patcher(b, s): return b + s*0.1

# 创建一个可干预模型 pv_tinystory，配置为：
# 对模型的每一层（从 0 到总层数减一）的 MLP 输出（mlp_output），应用我们刚定义的 pv_patcher 函数进行干预。
# 也就是说，在生成每个 token 时，都会把源示例对应位置的 MLP 输出 * 0.1 加到当前层的 MLP 输出上。
pv_tinystory = pv.IntervenableModel([{
    "layer": l,
    "component": "mlp_output",
    "intervention": pv_patcher
    } for l in range(tinystory.config.num_layers)],
    model=tinystory
)
# prompt and generate
# prompt：基础输入，用于启动故事生成（“从前有…”）
# happy_prompt：源示例，用于提取“干预信号”（这里只包含一个词 “ Happy”，代表“快乐”语义）
prompt = tokenizer(
    "Once upon a time there was", return_tensors="pt")
happy_prompt = tokenizer(
    " Happy", return_tensors="pt")

# prompt：基础输入序列
# happy_prompt：源示例序列（提供干预信号）
# 表示“把源示例中第 0 个位置（即 ‘Happy’）的表征，注入到基础生成过程的每一层”
# 返回值：第一个是未干预结果（忽略），第二个 intervened_story 是干预后的生成结果
_, intervened_story = pv_tinystory.generate(
    prompt, happy_prompt, 
    unit_locations = {"sources->base": 0},
    max_length=256
)

print(tokenizer.decode(
    intervened_story[0], 
    skip_special_tokens=True
))
```
