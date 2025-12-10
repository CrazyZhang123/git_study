## 1、os.path.dirname/abspath方法

```
import os
import shutil
import subprocess
from typing import List

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print('SCRIPT_DIR:',SCRIPT_DIR)
# __file__：当前脚本的路径（如 /root/zjj/MAT-Steer/test_subporcess.py）
# os.path.abspath(__file__)：转为绝对路径
# os.path.dirname(...)：取目录部分
# 结果：SCRIPT_DIR = "/root/zjj/MAT-Steer"

# 获取仓库根目录
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
print('REPO_ROOT:',REPO_ROOT)
# 仓库根目录 = SCRIPT_DIR 的父目录
# os.path.join(..., '..')：返回父目录
# 结果：REPO_ROOT = "/root/zjj/MAT-Steer"

```

**os.path.join(..., '..')：返回父目录;**
拼接了..就代表返回了上层目录。