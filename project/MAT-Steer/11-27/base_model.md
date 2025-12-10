You now have three TruthfulQA-style evaluators plus a sampler. Typical workflow:

## 1. Pre-sample a subset for reuse

> 可以不需要这个文件，直接测试

### (1) safetybench
```
  python validation/sample_eval_subset.py \
    --dataset safetybench \
    --input /mnt/private/zjj/data/test_en.json \
    --output validation/samples/safetybench_seed42_1k.json \
    --sample-size 1000 \
    --seed 42
```

  (Use bbq or truthfulqa plus the appropriate --input path for other datasets.)
 

## 2. Evaluate 

### 2.1 Evaluate SafetyBench

```
  python validation/test_safetybench_truthfulqa.py \
    --answers /mnt/private/zjj/SafetyBench/opensource_data/test_answers_en.json \
    --model-name llama3.1_8B \
    --device cuda:0 \
    --max-samples 5000 \
    --seed 42 \
    --save-sample validation/samples/safetybench_seed42_5k.json
```

 第二次跑
```
 python validation/test_safetybench_truthfulqa.py \
    --answers /mnt/private/zjj/SafetyBench/opensource_data/test_answers_en.json \
    --model-name llama3.1_8B \
    --device cuda:0 \
    --sample-file validation/samples/safetybench_seed42_5k.json
```
  (To reuse a saved subset, swap --data … --max-samples … --seed … for --sample-file validation/samples/
  safetybench_seed42_1k.json.)

### 2.2. Evaluate BBQ

```
  python validation/test_bbq_truthfulqa.py \
    --data /mnt/private/zjj/MAT-Steer/get_activations/bbq_sample_11165000.csv \
    --model-name llama3.1_8B \
    --device cuda:0 
```

```
  python validation/test_bbq_truthfulqa.py \
    --data /mnt/private/zjj/MAT-Steer/get_activations/bbq_sample_11165000.csv \
    --model-name llama3.1_8B \
    --device cuda:0 \
    --max-samples 1000 \
    --seed 42 \
    --save-sample validation/samples/bbq_seed42_1k.csv
```

### 2.3 Evaluate TruthfulQA

```
  python validation/test_truthfulqa_truthfulqa.py \
    --data /mnt/private/zjj/MAT-Steer/get_activations/truthfulqa_train.csv \
    --model-name llama3.1_8B \
    --device cuda:1 
```

  Flags:

  - --model-name accepts llama2_13B, llama3_8B, llama3.1_8B, qwen2.5_7B; use --model-path /custom/hf/dir to override.
  - --sample-file loads a pre-sampled dataset.
  - --save-sample writes the current random subset for future runs.
  - --force-recompute re-runs probability extraction instead of reusing cache.