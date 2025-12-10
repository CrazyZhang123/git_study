# 详细流程总结 — `evaluate_bias_toxicity_dataset(...)`

下面把你给出的函数按执行阶段拆成明确的步骤，解释每一步在做什么（包括不同分支、batch/单样本逻辑与回退策略），并给出一个 Markdown (Mermaid) 流程图用于可视化。

---

## 高层概览（一句话）

函数读取 ToxiGen/BBQ 的 JSON 数据 → 若有缓存则直接加载 → 模型按 batch 生成 `generated_texts`（baseline 可批量，intervention 多为逐样本）→ 将 `generated_texts` 用 judge_prompt 交给 judge_model 进行判别（Yes/No）→ 将判别结果转为 `predicted_labels` → 与 `true_labels` 比对并保存详细/汇总 CSV，返回准确率与详细 DataFrame。

---

## 详细步骤（逐步、含分支与回退）

### 0. 初始与参数

* 接收 `dataset_name, model, tokenizer, device, judge_model, judge_tokenizer, ...`
* `judge_model`/`judge_tokenizer` 若为 `None`，则使用 `model`/`tokenizer`（即生成者也做判别）。

---

### 1. 缓存检测（shortcut）

* 条件：`not force_recompute` 且 `output_path` 存在文件
* 若满足：读取 `output_path` → 计算并返回 `accuracy, results_df`（跳过后续计算）
* 若读取失败（异常），打印错误并继续重新计算

---

### 2. 载入数据集 JSON

* 根据 `dataset_name` 选择路径（优先 `*_sampled_400.json`，否则 `*_processed.json`）
* `task_type` = `'toxicity'` 或 `'bias'`
* 读取 JSON，`data` 为样本列表，每样本包含至少 `text` 与 `label`（0/1）
* 构建 `prompts = [sample['text'] for sample in data]` 与 `true_labels`

---

### 3. 构造 Judge Prompt 模板

* `toxicity` 或 `bias` 各有专用的 `judge_prompt_template`（要求回答 "Yes" 或 "No"）

---

### 4. 模型生成阶段（生成 `generated_texts`）

* `generated_texts = []`
* 以 `batch_size` 步长遍历 `prompts`：

  * 对每个 batch：

    * 尝试批量 `tokenizer(...)` → `.to(device)`（padding/truncation）
    * 记录 `input_lens`（attention_mask 求和）以便后面切分 output
    * **若 `baseline=True`（baseline 模式）**：

      * 可以直接用 `model.generate(**inputs, do_sample=False, num_beams=1, pad/eos ids...)` 得到 `output_ids`（批量）
      * 逐条解码 `output_ids[input_len:]` 得到 `generated_texts`（批量解码）
    * **若 `baseline=False`（干预模式，IntervenableModel）**：

      * 可能不支持批量 → 对 batch 中每个 prompt 逐条：

        * `single_input = tokenizer(prompt)` → `model.generate(single_input, output_original_output=True)`（得到 tuple）
        * 用 `input_len` 从生成结果中截取生成段并 `decode`，追加到 `generated_texts`
      * 完成后 `continue`（跳过批量解码逻辑）
    * **异常处理**：

      * 若批量生成/解码抛异常，则回退到逐个生成（逐 prompt 生成并解码），任何单个生成失败则 append `""` 并打印错误

---

### 5. Judge 阶段（对 `generated_texts` 做判别）

* 构建 `judge_prompts = [judge_prompt_template.format(text=text) for text in generated_texts]`
* `predicted_labels = []`, `judge_responses = []`
* 以 `judge_batch_size` 步长遍历 `judge_prompts`：

  * 批量 `judge_tokenizer(...)` → `.to(device)` → 得 `judge_input_lens`
  * **若 `judge_model == model and not baseline`**（即用同一个干预模型做判别）：

    * 可能需要逐条处理（同上）：

      * 对每条 prompt 逐条 `generate(output_original_output=True)` → decode → 得到 `judge_response`
      * 从 `judge_response` 中提取 `'yes'` / `'no'`（查找并比较位置）来决定 `predicted_label`（1/0）
      * 若无法判断，用长度或阈值作为回退规则
    * `continue`（跳过批量解码逻辑）
  * **否则（baseline 或 使用独立 judge_model）**：

    * 批量 `judge_model.generate(**judge_inputs, do_sample=False, num_beams=1, ...)` 得到 `judge_output_ids`
    * 批量解码 `judge_output_ids[input_len:]` 得 `judge_responses`
    * 从每条 `judge_response` 中提取 'yes'/'no' → 得 `predicted_labels`
  * **异常回退**：

    * 若批量判别失败，回退逐条判别；若仍失败，则用关键词列表（toxic_keywords / bias_keywords）在 `generated_text` 中匹配来产生 `predicted_label`

---

### 6. 构建结果与统计

* 遍历 `idx in range(total)`：

  * `is_correct = (predicted_labels[idx] == true_labels[idx])`
  * 累加 `correct_predictions`
  * 构建字典条目包含：`index, text, generated_text, true_label, predicted_label, judge_response, correct`
* `results_df = pd.DataFrame(results)`
* `accuracy = correct_predictions / total_predictions`

---

### 7. 保存与返回

* 若 `output_path`，写 `results_df.to_csv(output_path)`
* 构建 `summary_df` 并写 `summary_path`（若给定）
* 打印 summary（总样本数、正确数、准确率）
* 返回 `(accuracy, results_df)`

---

## 关键点与注意事项（为什么这样实现）

* **baseline 模式**允许批量高效生成；**intervention 模式**往往不支持批量，需要逐条生成 → 所以函数兼顾两种路径。
* **judge_model 可独立**，这样可以用专门判别器或人类标签器替代模型自评，灵活性高。
* **多重回退**（批量失败 → 逐条；判别失败 → 关键词）保证评估流程鲁棒，不会因单次生成/判别错误中断整个 run。
* **缓存读取**加速复试与多次 seed 评估。
* **判别逻辑基于文本解析**（查找 yes/no 的位置），这是常见的「prompted classification via LM」做法；存在自评偏差（judge 与 generator 相同模型时需谨慎）。

---

## Mermaid 流程图（Markdown 可直接展示）

```mermaid
flowchart TD
  A[Start evaluate_bias_toxicity_dataset] --> B{force_recompute?\nand output_path exists?}
  B -- Yes (use cache) --> C [Load CSV from output_path\nCompute accuracy\nReturn results]
  B -- No --> D[Load dataset JSON\n(set json_path, task_type)]
  D --> E[Build prompts & true_labels]
  E --> F[Prepare judge_prompt_template]
  F --> G[Generation Stage: loop batches]
  G --> G1{baseline?}
  G1 -- Yes --> G2[Batch-tokenize -> model.generate (batch)\nBatch-decode -> append generated_texts]
  G1 -- No --> G3[For each prompt in batch:\nsingle-tokenize -> model.generate(single)\ndecode -> append generated_texts]
  G2 --> H[Generation complete]
  G3 --> H
  H --> I[Construct judge_prompts from generated_texts]
  I --> J[Judging Stage: loop judge batches]
  J --> J1{judge_model == model and not baseline?}
  J1 -- Yes --> J2[For each judge prompt:\nsingle-tokenize -> judge_model.generate(single)\ndecode -> judge_response -> extract Yes/No -> predicted_label]
  J1 -- No --> J3[Batch-tokenize -> judge_model.generate(batch)\nBatch-decode -> judge_responses -> extract Yes/No -> predicted_labels]
  J2 --> K[Judge complete]
  J3 --> K
  K --> L[If any errors -> fallback:\n- per-sample generation/judging\n- final fallback: keyword matching]
  L --> M[Build results list:\nindex,text,generated_text,true_label,predicted_label,judge_response,correct]
  M --> N[Compute accuracy = correct / total]
  N --> O[Save detailed CSV (output_path) and summary (summary_path) if provided]
  O --> P[Print summary\nReturn accuracy, results_df]
```

---

如果你需要我：

* 把这个 Mermaid 图单独导出为图片（SVG/PNG），或
* 把函数改成**强制批量/并行**版本（例如通过队列处理 IntervenableModel），或
* 替换 model-as-judge 为某个现成的 toxic classifier（如 Detoxify）并改写判别逻辑，

我可以直接帮你实现其中任意一个。要哪个我就直接改给你。
