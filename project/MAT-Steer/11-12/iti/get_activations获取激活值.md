
## 1、加载dataset
加载dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")\['validation'\]

然后tokenized_tqa ->(Q:{} A:{})格式化，经过tokenizer，收集prompts和labels
> def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

## 2、创建收集器collector和IntervenableModel
创建收集最后一个token的所有 注意力头激活值的**收集器collector**,创建干预模型。
- 遍历每一层的编号（`layer`）。
- 为每层创建一个 `Collector` 对象，它是一个**可调用对象**（即实现了 `__call__` 方法的类）。
- 然后构建一个 `pv_config` 配置：
    - `"component"` 指向模型中要“注入”的具体模块（在这里是 self-attn 的 output projection 的输入）。
    - `"intervention"` 是一个函数，表示当该组件执行时，该函数会被调用。
- 一个典型的 **“钩子（hook）注入”模式**，用于**在模型前向传播时拦截中间层激活**。
详见[[hook 钩子]]
💡这其实是为 **`IntervenableModel`** 准备配置，用来告诉它“在哪一层注入什么函数”。

>    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) \#head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

**wrapper函数**
- 它只是返回了一个**闭包（closure）**。
- `intervener` 是传入的 `Collector` 实例。
- 所以 `wrapped` 其实就是调用 `Collector.__call__`。
- 即 `wrapper(collector) <=> lambda *a, **kw: collector(*a, **kw)`

**为什么要有?**
==因为有的框架（比如 `patched-vision`、`TransformerLens`、`HookedTransformer`）要求 hook 必须是**纯函数**（不带状态的函数），而不是直接传类对象。  
通过 `wrapper`，可以确保返回的对象是一个可调用函数。==
```
def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

```

### 3、获取激活值
- 遍历前面收集的tokenizer化的提示词
- 调用get_llama_activations_pyvene收集激活值
- all_layer_wise_activations收集所有层最后一个token的hidden激活值。
- all_head_wise_activations收集batch\[0]的最后一个token的所有注意力头激活值。
```python
    # prompt (1,seq_len)
    for prompt in tqdm(prompts):
        # 调用get_llama_activations_pyvene函数，传入可干预模型、收集器、当前提示和设备信息
        # 层级别激活值、头部级别激活值
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        # [层索引, token索引, 特征维度]
        # 使用切片[:,-1,:]提取所有层的最后一个token位置的激活值
        # 注意这个维度就没了
        # layer_wise_activations shape = (33, 25, 4096) 
        # layer_wise_activations[:,-1,:] shape是 [layer_num, hidden_dim]
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        # 因为使用collector收集的就是last token的注意力头的激活值分数
        # 维度：[layer_num,hidden_dim]  head_num*D_head
        all_head_wise_activations.append(head_wise_activations.copy())
```

#### 3.1 get_llama_activations_pyvene
将prompt放入device，然后放入需要收集的模型中，显式输出hidden_states。
- 将所有的hidden_states按行进行stack堆叠，然后去掉维度为0的冗余部分，然后丢到cpu上。
- 遍历收集器来收集的是b\[0,-1\]，即batch中第一个的最后一个token的所有注意力头。
	-  collector.collect_state默认都是true，collected_model()运行后，每个layer激活值都放在对应的collector.states列表中，然后按行进行堆叠,然后移动到cpu上,通过numpy()将张量转换为numpy数组，便于后续处理，放到head_wise_hidden_states列表中。
	- 重置collector实例——清空collector.states列表
- 将head_wise_hidden_states列表的所有np数组转为张量，按行堆叠。
- 返回hidden_states,head_wise_hidden_states,mlp输出(空)
```python
get_llama_activations_pyvene
    def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state: \#默认都是True 
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states
```
