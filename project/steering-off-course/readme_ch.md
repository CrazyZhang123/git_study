# 《偏离轨道》：语言模型引导中的可靠性挑战
### 《偏离轨道：语言模型引导中的可靠性挑战》官方代码

## [ACL 出版物](https://aclanthology.org/2025.acl-long.974/)

# 入门指南
1. 根据每个仓库的指南创建全新环境，以确保版本稳定性
### 关于 Hugging Face 版本的说明：
1. 某些模型需要比其他模型更新的 Hugging Face 版本。
2. 请参考这些模型的官方模型卡片（例如 [OLMo2](https://huggingface.co/allenai/OLMo-2-1124-7B)）来确定具体情况。

## 仓库结构
- [./DoLA](https://github.com/patqdasilva/steering-off-course/tree/main/DoLa)、[./function_vectors](https://github.com/patqdasilva/steering-off-course/tree/main/function_vectors) 和 [./icl_task_vectors](https://github.com/patqdasilva/steering-off-course/tree/main/icl_task_vectors)
    - 包含我们从相应论文中使用的各个代码库。
- [./notebooks](https://github.com/patqdasilva/steering-off-course/tree/main/notebooks)
    - 包含用于复现我们结果的脚本，详情如下。
- [./config.py](https://github.com/patqdasilva/steering-off-course/blob/main/config.py)
    - 包含有关模型路径和命名约定的有用信息
- [./figures](https://github.com/patqdasilva/steering-off-course/tree/main/figures)
    - 包含我们测试过的所有方法、模型和任务的所有结果图表。
    - 要复现我们的图表，请使用每个 ./notebooks/plot*.ipynb 文件

# 运行实验
## 使用 Bash 或 Slurm 运行
- 我们设置了一个便捷的 Jupyter 笔记本 ./notebooks/run_slurm.ipynb，可通过易于设置的超参数运行所有实验。
- 这是我们运行实验的主要方法，有助于管理众多输入参数。
    - ./*.sh 文件的输入与下面 [Python 指南](#to-run-using-python) 中详细说明的相同
- 请在相应的 "./*.sh" 文件中修改 Slurm 指令以适应您自己的配置。
- 要在不使用 Slurm 的情况下运行，请从 subprocess.run 调用中删除 'sbatch' 和 '*slurm_cmd'

## 使用 Python 运行
### 函数向量（Function Vectors）
1. 要创建函数向量，必须首先计算对任务的间接影响
```bash
cd function_vectors
python ./src/compute_indirect_effect.py \
    --model_name [model_name] \ # 模型名称
    --model_fp [model_fp] \ # 模型的绝对文件路径
    --dataset_name [dataset_name] \ # 数据集（参见 ./function_vectors/dataset_files 中的任务）
    --root_data_dir ./dataset_files \ # 数据存储位置
    --save_path_root ./results/[model_name] # 结果保存位置
```
我的

```
python ./src/compute_indirect_effect.py \
    --model_name llama3-8B \
    --model_fp "/webdav/Storage(default)/MyData/llms/Meta-Llama-3-8B" \
    --dataset_name generate \
    --root_data_dir ./dataset_files \
    --save_path_root ./results/llama3-8B
```



2. 然后，可以复现我们的参数搜索

```bash
cd function_vectors
python ./experiments/run_param_sweep.py \
  --model_name [model_name] \ # 模型名称
  --model_fp [model_fp] \ # 模型的绝对文件路径
  --data_name [task] \ # 数据集（参见 ./function_vectors/dataset_files 中的任务）
  --nshot_baseline True # 是否使用预定义的 N-shot 示例
```

### 任务向量（Task Vectors）
```bash
cd icl_task_vectors
./run_script.sh experiments.main # 运行前必须在 ./configs 中设置您需要的模型
```

### DoLA 方法
对于 TruthfulQA 多项选择
```bash
cd DoLa
python tfqa_mc_eval.py \
    --model-name [model_name] \ # 用于命名的模型名称
    --model-fp [model_fp] \ # 模型的绝对文件路径
    --data-path "./tfqa" \ # tfqa 数据集的下载位置
    --output-path [out_fp] \ # 输出文件路径
    --num-gpus $n_gpu \ # GPU 数量
    --early-exit-layers [early_exit_layers] \ # 例如，对于 32 层的模型，0-25% 的区间为 '0,2,4,6,8,32'
    --relative_top $alpha \ # 论文中讨论的 alpha 超参数
    --post_softmax "y" \ # 是否包含 softmax，如 3.1 节“评估”中所述
    --ln_type $ln_type \ # 嵌入前应用的层归一化类型。我们设置为 'none'
    --len_bias_const 0 # 3.1 节“评估”和表 7 中描述的长度偏差常数
```

对于 FACTOR 数据集
```bash
cd ../DoLa
python factor_eval.py \
    --model-name [model_name] \ # 用于命名的模型名称
    --model-fp [model_fp] \ # 模型的绝对文件路径
    --data-path "./factor/news_factor.csv" \ # news_factor 数据集的下载位置
    --output-path [out_fp] \ # 输出文件路径
    --num-gpus $n_gpu \ # GPU 数量
    --early-exit-layers [early_exit_layers] \ # 例如，对于 32 层的模型，0-25% 的区间为 '0,2,4,6,8,32'
    --relative_top $alpha \ # 论文中讨论的 alpha 超参数
    --post_softmax "y" \ # 是否包含 softmax，如 3.1 节“评估”中所述
    --ln_type $ln_type \ # 嵌入前应用的层归一化类型。我们设置为 'none'
    --len_bias_const 0 # 3.1 节“评估”和表 7 中描述的长度偏差常数
```

### Logit Lens 方法
```bash
cd function_vectors
python ./logit_lens/logit_lens.py \
  --model_name [model_name] \ # 模型名称
  --model_fp [model_fp] \ # 模型的绝对文件路径
  --data_name [task] \ # 数据集（参见 ./function_vectors/dataset_files 中的任务，或 'tqa'）
  --n_shots [n_shots] \ # 提示中包含的示例数量
  --max_samples [max_samples] # 最大样本数量
```

# 引用我们的工作
如果引用我们的预印本，请使用以下格式：
```bibtex
@inproceedings{silva-etal-2025-steering,
    title = "Steering off Course: Reliability Challenges in Steering Language Models",
    author = "Da Silva, Patrick Queiroz  and
      Sethuraman, Hari  and
      Rajagopal, Dheeraj  and
      Hajishirzi, Hannaneh  and
      Kumar, Sachin",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.974/",
    doi = "10.18653/v1/2025.acl-long.974",
    pages = "19856--19882",
    ISBN = "979-8-89176-251-0",
    abstract = "语言模型（LMs）的引导方法作为微调的轻量级替代方案受到关注，能够有针对性地修改模型激活。然而，先前的研究主要报告少数模型的结果，在理解这些方法的稳健性方面存在关键差距。在这项工作中，我们系统地检查了三种主要的引导方法——DoLa、函数向量和任务向量。与原始研究（仅评估少数模型）不同，我们测试了多达 36 个模型，分属 14 个家族，参数规模从 1.5B 到 70B 不等。我们的实验揭示了这些引导方法的有效性存在显著差异，大量模型的引导性能没有改善，有时甚至下降。我们的分析揭示了这些方法背后假设的根本缺陷，对其作为可扩展引导解决方案的可靠性提出了挑战。"
}
```