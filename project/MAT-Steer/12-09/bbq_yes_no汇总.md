
## 日志
Loaded 2334 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_disability_status.csv
We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
Accuracy: 0.3745                                                                                 Accuracy_norm: 0.3333                                                                            Evaluating age...                                                                                Loaded 5520 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_age.csv                                                                                         Accuracy: 0.3674                                  Accuracy_norm: 0.3333
Evaluating nationality...
Loaded 4620 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_nationality.csv
Accuracy: 0.3931
Accuracy_norm: 0.3333
Evaluating race_x_ses...
Loaded 16740 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_race_x_ses.csv
Accuracy: 0.4114
Accuracy_norm: 0.3333

Evaluating gender_identity...
Loaded 8508 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_gender_identity.csv
We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
Accuracy: 0.3544
Accuracy_norm: 0.3333
Evaluating physical_appearance...
Loaded 2364 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_physical_appearance.csv
Accuracy: 0.4061
Accuracy_norm: 0.3333
Evaluating race_ethnicity...
Loaded 10320 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_race_ethnicity.csv
Accuracy: 0.4016
Accuracy_norm: 0.3333

Evaluating race_x_gender...
Loaded 23940 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_race_x_gender.csv
We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
Accuracy: 0.3599
Accuracy_norm: 0.3333
Evaluating religion...
Loaded 1800 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_religion.csv
Accuracy: 0.4506
Accuracy_norm: 0.3333
Evaluating ses...
Loaded 10296 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_ses.csv
Accuracy: 0.3503
Accuracy_norm: 0.3333
Evaluating sexual_orientation...
Loaded 1296 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_pairs_seed42_sexual_orientation.csv
Accuracy: 0.3858
Accuracy_norm: 0.3333

## 汇总

合并MC1方法和两个表格，包含类别、Accuracy和 MC1：

| 类别 (Category)       | Accuracy | MC1    |
| ------------------- | -------- | ------ |
| disability_status   | 0.3745   | 0.4614 |
| age                 | 0.3674   | 0.4495 |
| nationality         | 0.3931   | 0.4896 |
| race_x_ses          | 0.4114   | 0.4692 |
| gender_identity     | 0.3544   | 0.2948 |
| physical_appearance | 0.4061   | 0.4543 |
| race_ethnicity      | 0.4016   | 0.3881 |
| race_x_gender       | 0.3599   | 0.4073 |
| religion            | 0.4506   | 0.4567 |
| ses                 | 0.3503   | 0.4231 |
| sexual_orientation  | 0.3858   | 0.4444 |
合并后的表格（包含 MC2）：

| 类别 (Category)       | 数量 (Samples) | Accuracy | .Accuracy | MC1    | MC2    |
| ------------------- | ------------ | -------- | --------- | ------ | ------ |
| disability_status   | 2334         | 0.3745   | 0.4353    | 0.4614 | 0.4223 |
| age                 | 5520         | 0.3674   | 0.3725    | 0.4495 | 0.3937 |
| nationality         | 4620         | 0.3931   | 0.4422    | 0.4896 | 0.4399 |
| race_x_ses          | 16740        | 0.4114   | 0.3892    | 0.4692 | 0.4569 |
| gender_identity     | 8508         | 0.3544   | 0.4081    | 0.2948 | 0.3215 |
| physical_appearance | 2364         | 0.4061   | 0.420     | 0.4543 | 0.4388 |
| race_ethnicity      | 10320        | 0.4016   | 0.3757    | 0.3881 | 0.3637 |
| race_x_gender       | 23940        | 0.3599   | 0.3691    | 0.4073 | 0.3698 |
| religion            | 1800         | 0.4506   | 0.4372    | 0.4567 | 0.4356 |
| ses                 | 10296        | 0.3503   | 0.4003    | 0.4231 | 0.3761 |
| sexual_orientation  | 1296         | 0.3858   | 0.4066    | 0.4444 | 0.3984 |

**统计摘要：**
- MC1 范围：0.2948 - 0.4896（最高：nationality，最低：gender_identity）
- MC2 范围：0.3215 - 0.4569（最高：race_x_ses，最低：gender_identity）
- 所有类别的 Accuracy_norm 均为 0.3333（随机基线）

**注意：**
- 数量列存在差异：日志输出显示的是测试样本数（如 disability_status: 2334），而 CSV 中的 `total_samples` 不同（如 disability_status: 778）。这可能是因为：
  - 日志输出是测试集样本数
  - CSV 是实际评估的样本数（可能经过过滤或采样）
- MC1 分数来自 CSV 文件的 baseline 结果
- Accuracy 和 Accuracy_norm 来自日志输出

需要我进一步分析这些差异吗？