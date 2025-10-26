
## 1.loc和iloc区别
`loc`和`iloc`是两种不同的索引方式，它们的主要区别在于索引的依据：

1. `questions.loc[1, 'Incorrect Answers']`:
    
    - 使用标签（label）进行索引
    - 第一个参数`1`是行索引标签（即行的名称或索引值）
    - 第二个参数`'Incorrect Answers'`是列名
    - 这种方式通过行标签和列名来定位数据
2. `questions.iloc[1, 'Incorrect Answers']`:
    
    - 使用整数位置进行索引
    - 第一个参数`1`是行的整数位置（从0开始计数）
    - 第二个参数应该是列的整数位置，而不是列名
    - 注意：`iloc`通常不接受列名作为参数，如果要使用列名，应该用`loc`

# 2、a\[:,\[1 2]] 和a\[range(len(a)),\[1 2 ]] 区别

outputs\[:, prompt_ids.squeeze(0)].shape 与 outputs\[range(outputs.shape\[0]), prompt_ids.squeeze(0)].shape

这两种PyTorch索引方式在结果形状和含义上有重要区别：

1. **`outputs[:, prompt_ids.squeeze(0)]`**:
    
    - 这是基本的切片索引
    - `:`表示选择所有行
    - `prompt_ids.squeeze(0)`是一个一维张量，包含要选择的列索引
    - 结果形状为`(outputs.shape[0], prompt_ids.squeeze(0).shape[0])`
    - 这会创建一个二维矩阵，其中包含所有行在指定列索引位置的值
    - 例如，如果`outputs`形状为`(5, 10)`，`prompt_ids.squeeze(0)`为`[1, 3, 2]`，则结果形状为`(5, 3)`
2. **`outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]`**:
    
    - 这是高级索引（advanced indexing）
    - `range(outputs.shape[0])`创建一个行索引序列`[0, 1, 2, ..., outputs.shape[0]-1]`
    - `prompt_ids.squeeze(0)`是列索引序列
    - 两个索引序列长度必须相同
    - 结果形状为`(outputs.shape[0],)`，是一维张量
    - 这种索引方式会返回一个一维张量，其中第i个元素是`outputs[i, prompt_ids.squeeze(0)[i]]`
    - 例如，如果`outputs`形状为`(5, 10)`，`range(outputs.shape[0])`为`[0, 1, 2, 3, 4]`，`prompt_ids.squeeze(0)`为`[1, 3, 2, 0, 4]`，则结果形状为`(5,)`，包含元素`[outputs[0,1], outputs[1,3], outputs[2,2], outputs[3,0], outputs[4,4]]`

**关键区别**：

- 第一种方式返回一个二维矩阵，包含所有行在指定列的值
- 第二种方式返回一个一维向量，只包含对角线上的元素（按索引对应关系）


# 3、pd.to_csv方法

```
model_outputs = pd.read_csv('./helpsteer/helpfulness_model_outputs_seed42_n100.csv')
```



```
# iloc[index]这样取只有一个元素就会变成一个series，是竖着的
model_outputs.iloc[0].to_csv('./_outputs_seed42_n100_test.csv', index=False)
# iloc[index:index+1]这样取，还是一个df
model_outputs.iloc[0:1].to_csv('./_outputs_seed42_n100_test.csv', index=False)
# model_outputs.iloc[0]['prompt']
```


