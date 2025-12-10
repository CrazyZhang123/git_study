

## 子数据集生成
```
import pandas as pd
import random
from datasets import load_from_disk

seed=42
# 设置随机种子，保证可复现
random.seed(seed)

# 加载 BBQ 数据集
file_path = '/webdav/Storage(default)/MyData/datasets/BBQ'
datasets = load_from_disk(file_path)

# 保存路径
save_dir = '/root/myproject/zjj/MAT-steer/get_activations/data/'

# 需要提取的字段
fields = ['context', 'question', 'ans0', 'ans1', 'ans2', 'answer_label']

all_data = []

# 遍历每个子类别（例如 age、gender_identity 等）
for category in datasets.keys():
    data = datasets[category]
    print(f"处理类别: {category}，样本数: {len(data)}")

    # 仅保留需要的字段
    for item in data:
        entry = {key: item.get(key, "") for key in fields}
        entry['category'] = category
        all_data.append(entry)

# 将所有类别的数据合并为 DataFrame
df = pd.DataFrame(all_data)
print(f"合并后的总样本数: {len(df)}")

# 打乱数据
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# 如果样本不足，则自动均衡到最小类样本数
categories = df['category'].unique()
min_samples = min(df['category'].value_counts())

# 取每个类别的 min_samples 条记录
balanced_df = df.groupby('category', group_keys=False).apply(lambda x: x.sample(n=min_samples, random_state=seed))
balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

print(f"均衡后每类样本数: {min_samples}，总样本数: {len(balanced_df)}")

# 按 5:5 划分训练集和测试集（与 truthfulqa 一致）
balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
split_idx = len(balanced_df) // 2
train_df = balanced_df.iloc[:split_idx].copy()
test_df = balanced_df.iloc[split_idx:].copy()

# 保存为 CSV 文件
train_path = f'{save_dir}/bbq_train_seed{seed}.csv'
test_path = f'{save_dir}/bbq_test_seed{seed}.csv'

train_df.to_csv(train_path, index=False, encoding='utf-8')
test_df.to_csv(test_path, index=False, encoding='utf-8')

print(f"✅ 训练集已保存到: {train_path} ({len(train_df)} 条)")
print(f"✅ 测试集已保存到: {test_path} ({len(test_df)} 条)")

```


所有类别
item来自 ['age', 'disability_status', 'gender_identity', 'nationality', 'physical_appearance', 'race_ethnicity', 'race_x_gender', 'race_x_ses', 'religion', 'ses', 'sexual_orientation']
bbq_test_seed42_{item}.csv
## 评估方法

```
CUDA_VISIBLE_DEVICES=1 python my_run_mat_eval.py \
    --checkpoint ./checkpoints/llama3.1_8B_L14_mat_steer_1206.pt \
    --dataset bbq \
    --model_name llama3.1_8B \
    --questions_path ./get_activations/data/toxigen_test_seed42_sample394.csv\
    --baseline \
    --seed 42 \
    --force_recompute
```
