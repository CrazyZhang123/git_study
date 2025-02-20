

安装ChatGLM3-6B模型

```
pip install modelscope
// 下载整个模型repo到指定目录
modelscope download --model ZhipuAI/chatglm3-6b --local_dir ../model
```

#### 其他

解压 .tar.gz 文件

如果文件是以 .tar.gz 或 .tgz 为扩展名，可以使用 *tar* 命令来解压。示例如下：

tar -zxvf FileName.tar.gz

### 1.basic_demo

> 配置MODEL_PATH

在 Linux 或 macOS 上
临时设置（仅对当前终端会话有效）：

```
export MODEL_PATH=/path/to/your/model
```

永久设置（对所有终端会话有效）：

打开你的 shell 配置文件（例如 .bashrc、.bash_profile、.zshrc 等）：

```
vim ~/.bashrc
```

添加以下行：

```
export MODEL_PATH=/path/to/your/model
```

保存并关闭文件，然后运行：

```
source ~/.bashrc
```

