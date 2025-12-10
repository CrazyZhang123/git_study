正常运行
> sample394

```bash
CUDA_VISIBLE_DEVICES=1 python my_run_mat_eval.py \
    --checkpoint ./checkpoints/llama3.1_8B_L14_mat_steer_1206.pt \
    --dataset bbq \
    --model_name llama3.1_8B \
    --questions_path ./get_activations/data/toxigen_test_seed42_sample394.csv\
    --baseline \
    --seed 42 \
    --force_recompute
```





