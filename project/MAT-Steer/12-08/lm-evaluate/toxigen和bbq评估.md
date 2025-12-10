
```
lm_eval --model hf \
    --model_args pretrained=/webdav/Storage\(default\)/MyData/llms/Meta-Llama-3.1-8B \
    --tasks toxigen_local,bbq \
    --device cuda:0 \
    --batch_size auto\
    --output_path ./results/toxigen_local_results.json
```
  