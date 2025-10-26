
# from datasets import load_dataset,load_from_disk

# # 加载 toxigen 数据集

# BBQ_data = load_dataset("Elfsong/BBQ")


# file_path = f'./BBQ'
# # 保存数据集
# BBQ_data.save_to_disk(file_path)
# # 从磁盘加载数据集
# datasets = load_from_disk(file_path)
# # 查看数据集
# print(datasets)

from datasets import load_dataset,load_from_disk

# 加载 toxigen 数据集

# import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # To load the raw human annotations
toxigen_data = load_dataset("toxigen/toxigen-data", "prompts")
print("成功加载默认配置的数据集")
print(f"数据集包含的split: {list(toxigen_data.keys())}")
print(f"数据集特征: {toxigen_data[next(iter(toxigen_data))].features}")

file_path = f'./toxigen'
# 保存数据集
toxigen_data.save_to_disk(file_path)
# 从磁盘加载数据集
datasets = load_from_disk(file_path)
# 查看数据集
print(datasets)