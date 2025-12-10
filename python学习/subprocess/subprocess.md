subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。

使用 subprocess 模块
subprocess 模块首先推荐使用的是它的 run 方法，更高级的用法可以直接使用 Popen 接口。

run 方法语法格式如下：

subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None)
args：表示要执行的命令。必须是一个字符串，字符串参数列表。
stdin、stdout 和 stderr：子进程的标准输入、输出和错误。其值可以是 subprocess.PIPE、subprocess.DEVNULL、一个已经存在的文件描述符、已经打开的文件对象或者 None。subprocess.PIPE 表示为子进程创建新的管道。subprocess.DEVNULL 表示使用 os.devnull。默认使用的是 None，表示什么都不做。另外，stderr 可以合并到 stdout 里一起输出。
timeout：设置命令超时时间。如果命令执行时间超时，子进程将被杀死，并弹出 TimeoutExpired 异常。
check：如果该参数设置为 True，并且进程退出状态码不是 0，则弹 出 CalledProcessError 异常。
encoding: 如果指定了该参数，则 stdin、stdout 和 stderr 可以接收字符串数据，并以该编码方式编码。否则只接收 bytes 类型的数据。
shell：如果该参数为 True，将通过操作系统的 shell 执行指定的命令。
run 方法调用方式返回 CompletedProcess 实例，和直接 Popen 差不多，实现是一样的，实际也是调用 Popen，与 Popen 构造函数大致相同，例如:

```python
#执行ls -l /dev/null 命令
>>> subprocess.run(["ls", "-l", "/dev/null"])
crw-rw-rw-  1 root  wheel    3,   2  5  4 13:34 /dev/null
CompletedProcess(args=['ls', '-l', '/dev/null'], returncode=0)
returncode: 执行完子进程状态，通常返回状态为0则表明它已经运行完毕，若值为负值 "-N",表明子进程被终。
```

简单实例：

```python
import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)


runcmd(["dir","/b"])#序列参数
runcmd("exit 1")#字符串参数
```
**解释：**
>shell=True：在 shell 中执行命令
>stdout=subprocess.PIPE：捕获标准输出
>stderr=subprocess.PIPE：捕获标准错误
>encoding="utf-8"：以 UTF-8 解码输出
>timeout=1：超时 1 秒

输出结果如下：

```bash
(base) root@ubuntu:~/zjj/MAT-Steer# python test_subporcess.py 

success: CompletedProcess(args=['dir', '/b'], returncode=0, stdout='assets\t\t    get_activations  __pycache__\t TruthfulQA\nenvironment.yaml    helpsteer\t     README.md\t\t validation\nevaluate.py\t    MAT-Steer.md     test_intervent.py\t validation_answers.csv\nfeatures\t    MAT-test.py      test_mc2分数.py\nfinetune_gpt.ipynb  others\t     test_subporcess.py\n', stderr='')

error: CompletedProcess(args='exit 1', returncode=1, stdout='', stderr='')
```

**解释**
>命令：dir /b（Windows 列出目录，/b 为简洁格式）
>结果：returncode=0 表示成功
>stdout 内容：目录列表（assets、get_activations、pycache、TruthfulQA 等）
>注意：虽然传入列表 ["dir","/b"]，但 shell=True 时会被拼接为字符串执行。
>