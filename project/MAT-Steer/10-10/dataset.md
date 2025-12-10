
bbq
```python
            # 从磁盘加载BBQ数据集
            file_path = '/webdav/Storage(default)/MyData/datasets/BBQ'
            datasets = load_from_disk(file_path)

            # # 遍历BBQ数据集的所有子类别（如age、gender_identity等）
            # for category in datasets.keys():
            #     # 获取当前类别的数据分片（如datasets['age']）
            # category = 'age'
            # 
            TOTAL_SAMPLES = 5000
            categories = list(datasets.keys())
            samples_per_category = TOTAL_SAMPLES // len(categories)
            print(f"共有 {len(categories)} 个类别，每类采样 {samples_per_category} 条，总计约 {TOTAL_SAMPLES} 条样本")

            processed_data = []

            # ========== 5. 遍历每个分类并均衡采样 ==========
            for category in categories:
                data = datasets[category]
                print(f"\n📂 处理类别: {category}，样本总数: {len(data)}")

                category_samples = []

                for item in data:
                    context = item.get("context", "")
                    question = item.get("question", "")
                    base_text = f"{context} {question}" if context else question
                    base_text = base_text.strip()

                    # 提取3个选项
                    answers = [item.get(f"ans{i}", "").strip() for i in range(3)]
                    correct_label = item.get("answer_label", item.get("label", 0))

                    # 跳过无效样本
                    if not isinstance(correct_label, int) or correct_label not in [0, 1, 2]:
                        continue

                    # 遍历选项
                    for i, answer in enumerate(answers):
                        if not answer:
                            continue
                        full_text = format_bbq(base_text, answer)
                        label = 1 if i == correct_label else 0
                        category_samples.append({
                            "text": full_text,
                            "label": label,
                            "category": category
                        })

                # 从该类别中随机采样固定数量
                if len(category_samples) > samples_per_category:
                    category_samples = random.sample(category_samples, samples_per_category)
                else:
                    print(f"⚠️ 类别 {category} 样本不足，仅 {len(category_samples)} 条")

                processed_data.extend(category_samples)

```

toxgen
```python
 elif dataset_name == "toxigen":
            # Expect format with text/prompt and toxicity_score or label
            version = 'annotations'
            file_path = f'/webdav/Storage(default)/MyData/datasets/toxigen/{version}'
            # 从磁盘加载数据集
            datasets = load_from_disk(file_path)
            import re
            import numpy as np
            from sklearn.metrics import accuracy_score

            def fix_bytes_prefix(s):
                """去除 b'...' 前缀并解码"""
                if not isinstance(s, str):
                    return s
                s = s.strip()
                if s.startswith("b'") or s.startswith('b"'):
                    s = s[2:]
                    if (s.endswith("'") or s.endswith('"')):
                        s = s[:-1]
                    # 解码 \\\\n 等转义字符
                    s = bytes(s, "utf-8").decode("unicode_escape")
                return s.strip()

            def convert_toxigen_to_mcq(dataset, text_field='Input.text', label_field='Input.prompt_label'):
                """
                将 Toxigen 数据集转换为多选题（单选）格式，每个样本一个问题多个选项
                """
                # 1. 收集所有标签
                labels = set()
                for item in dataset:
                    if label_field in item and item[label_field] is not None:
                        labels.add(item[label_field].strip())
                # 如果标签少于2个，则补一个 neutral 以确保是多选题
                if len(labels) < 2:
                    labels.add("neutral")
                label_set = sorted(list(labels))
                label2id = {l: i for i, l in enumerate(label_set)}

                examples = []
                for item in dataset:
                    if text_field not in item or label_field not in item:
                        continue
                    text = fix_bytes_prefix(item[text_field])
                    label = item[label_field].strip()
                    if label not in label2id:
                        continue
                    examples.append({
                        "question": text,
                        "choices": label_set,
                        "label": label2id[label],
                        "raw_label": label
                    })
                return examples, label2id

            def format_truthfulqa(question, choice):
                return f"Q: {question} A: {choice}"

            def build_mcq_records(examples):
                """
                将每道题转换为多条记录：每个选项一条
                """
                processed_data = []
                for ex in examples:
                    q = ex["question"]
                    choices = ex["choices"]
                    correct_label = ex["label"]
                    for i, choice in enumerate(choices):
                        label = 1 if i == correct_label else 0
                        processed_data.append({
                            "text": format_truthfulqa(q, choice),
                            "label": label
                        })
                return processed_data

            # ========== 示例 ==========

            # 假设你已有 HuggingFace DatasetDict：
            # dataset = datasets['train']  # 例如 HuggingFace 的数据集对象
            # 若不是列表，可转换： dataset = dataset.to_list()
            dataset = datasets['train'] # 保证是 list

            examples, label2id = convert_toxigen_to_mcq(dataset)

            # 随机抽样
            sample_size = min(5000, len(examples))
            random_indices = np.random.choice(len(examples), size=sample_size, replace=False)
            sampled_examples = [examples[i] for i in random_indices]

            # 转换为 Q/A 格式
            processed_data = build_mcq_records(sampled_examples)
```