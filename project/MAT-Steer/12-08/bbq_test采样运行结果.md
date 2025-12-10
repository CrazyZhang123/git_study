
### 采样394条
> 和truthfulqa数据量一样。

Summary results saved to: /root/myproject/zjj/MAT-steer/validation/results_dump/summary_dump/llama3.1_8B_seed_42_alpha_1_bbq.csv

=== Evaluation Results for bbq (MC scores, intervention) ===
**Total samples: 394**
MC1 Score: 0.4239
MC2 Score: 0.3872
MC3 Score: 0.4239
Running baseline evaluation…
DEBUG [evaluate_bias_toxicity_dataset_v2]: force_recompute=True, output_path=/root/myproject/zjj/MAT-steer/validation/results_dump/answer_dump/llama3.1_8B_seed_42_alpha_1_bbq_baseline.csv, exists=True
DEBUG: force_recompute=True, skipping cache and recomputing...
Loaded 394 samples from /root/myproject/zjj/MAT-steer/get_activations/data/bbq_test_seed42_sample394.csv
Converting BBQ to TruthfulQA format...
Converting BBQ to TruthfulQA format: 100%|███████████████████████████████████████████████████████████████████| 394/394 [00:00<00:00, 12082.82it/s]
Converted 394 samples to TruthfulQA format
Running tqa_run_probs on 394 questions...
tqa_run_probs: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 394/394 [01:37<00:00,  4.05it/s]
Detailed results saved to: /root/myproject/zjj/MAT-steer/validation/results_dump/answer_dump/llama3.1_8B_seed_42_alpha_1_bbq_baseline.csv
Summary results saved to: /root/myproject/zjj/MAT-steer/validation/results_dump/summary_dump/llama3.1_8B_seed_42_alpha_1_bbq_baseline.csv

=== Evaluation Results for bbq (MC scores, baseline) ===
Total samples: 394
MC1 Score: 0.4239
MC2 Score: 0.3872
MC3 Score: 0.4239

## 采样940
> 与toxigen数据量相同

Detailed results saved to: /root/myproject/zjj/MAT-steer/validation/results_dump/answer_dump/llama3.1_8B_seed_42_alpha_1_bbq.csv
Summary results saved to: /root/myproject/zjj/MAT-steer/validation/results_dump/summary_dump/llama3.1_8B_seed_42_alpha_1_bbq.csv

=== Evaluation Results for bbq (MC scores, intervention) ===
Total samples: 940
MC1 Score: 0.4181
**MC2 Score: 0.3911**
MC3 Score: 0.4181
Running baseline evaluation…


=== Evaluation Results for bbq (MC scores, baseline) ===
Total samples: 940
MC1 Score: 0.4181
MC2 Score: 0.3911
MC3 Score: 0.4181

## 全量结果

dataset,task_type,total_samples,MC1,MC2,MC3,baseline

bbq,bias,4752,0.42108585858585856,0.3963618725388261,0.42108585858585856,False