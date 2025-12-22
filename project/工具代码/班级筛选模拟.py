# import numpy as np

# # 模拟 1000 个特征，分布在 40 层楼
# # 对每个数字重复 25 次，得到 1000 个样本。
# layers = np.repeat(np.arange(40), 25)
# # 后期层的输出得分普遍比前期层高
# base_scores = layers / 40.0 
# output_scores = np.clip(base_scores + np.random.normal(0, 0.2, 1000), 0, 1)

# # 模拟引导成功率
# # 真实成功率 = 基础分 + 输出得分贡献 + 噪声
# success_rates = 0.2 + (output_scores * 1.2) + np.random.normal(0, 0.05, 1000)

# # --- 图 4 逻辑：全校筛选 ---
# all_mean = np.mean(success_rates) # 约 0.6
# high_score_global = success_rates[output_scores > 0.9]
# print(f"图4 - 全模型起步分: {all_mean:.2f}")
# print(f"图4 - 全模型高分筛选后: {np.mean(high_score_global):.2f}")

# # --- 图 5 逻辑：只看尖子班 (30-40层) ---
# elite_indices = np.where(layers >= 30)[0]
# elite_mean = np.mean(success_rates[elite_indices]) # 约 0.8
# high_score_elite = success_rates[np.intersect1d(elite_indices, np.where(output_scores > 0.9)[0])]
# print(f"图5 - 深层起步分: {elite_mean:.2f}")
# print(f"图5 - 深层再次高分筛选后: {np.mean(high_score_elite):.2f}")

import math

# 假设的一个简单词频库 (来自语料库统计)
word_probs = {"apple": 0.001, "lenses": 0.0005, "contact": 0.0008}
pair_probs = {
    ("apple", "apple"): 0.0000001,  # 极低，谁会说 apple apple?
    ("contact", "lenses"): 0.0002   # 很高，常用搭配
}

def calculate_pmi(word_a, word_b):
    p_a = word_probs[word_a]
    p_b = word_probs[word_b]
    p_ab = pair_probs.get((word_a, word_b), 1e-10)
    
    # PMI 公式: log( P(ab) / (P(a)*P(b)) )
    pmi = math.log2(p_ab / (p_a * p_b))
    return pmi

# 场景 1: 精准引导 (好方向盘)
# 特征是 apple，输出也是 apple
pmi_good = calculate_pmi("apple", "apple")
print(f"精准引导 PMI (apple-apple): {pmi_good:.2f}") # 应该是负数

# 场景 2: 被带偏的引导 (混合特征)
# 特征是 contact，输出是 lenses
pmi_mixed = calculate_pmi("contact", "lenses")
print(f"混合特征 PMI (contact-lenses): {pmi_mixed:.2f}") # 应该是高正数
