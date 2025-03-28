### 1 .修改csv文件的列名

编码都改为utf-8,

中文:

<img src="file:///C:\Users\张俊杰\AppData\Local\Temp\QQ_1740989024825.png" alt="img" style="zoom:50%;" />

中文相关的C_text.csv,label_C.csv 需要把列名images_name改为 filename,和英文版本保持一致。

英文：<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250303160615959.png" alt="img" style="zoom:50%;" />



### 2、下载语言和视觉模型

```
huggingface-cli download google/vit-base-patch16-224 --local-dir /data1/zjj/wo
rkspace/COT/vit-base-patch16-224/
```



```
huggingface-cli download FacebookAI/xlm-roberta-base --local-dir /data1/zjj/workspace/COT/xlm-roberta-base/
```



3、logging模块

运行日志前，清空日志

file_handler = logging.FileHandler(f"./logs/{logger_name}.log",mode='w')

或者

logging.basicConfig(filemode='w')
