下面我按“模块”帮你把这个 `pyproject.toml` 翻译成人话，你以后看别的项目的也能举一反三。

---

## 一、`pyproject.toml` 是干嘛的？

简单说：
**这是 Python 项目的“配置总表”**，告诉工具：

* 这个包叫什么、版本是多少、作者是谁
* 需要哪些依赖
* 怎么打包、怎么安装
* 命令行里 `lm-eval` 这个命令应该指向哪个函数
* 代码风格检查（ruff）、文档检查（pymarkdown）怎么配

你可以把它看成是以前 `setup.py + setup.cfg + requirements.txt` 的合体升级版。

---

## 二、逐段解释

### 1. `[build-system]` —— 用什么工具来“构建”这个包

```toml
[build-system]
requires = ["setuptools>=40.8.0", "wheel"]
build-backend = "setuptools.build_meta"
```

意思是：

* 构建（打包）这个项目需要先安装 `setuptools` 和 `wheel`
* 构建工具使用 `setuptools.build_meta` 这个后端（就是老牌的 setuptools）

---

### 2. `[project]` —— 项目的基本信息

```toml
[project]
name = "lm_eval"
version = "0.4.9.2"
authors = [
    { name = "EleutherAI", email = "contact@eleuther.ai" }
]
description = "A framework for evaluating language models"
readme = "README.md"
...
requires-python = ">=3.10"
license = { "text" = "MIT" }
dependencies = [
    "accelerate>=0.26.0",
    "evaluate",
    ...
]
```

* `name`：包名，安装后就是 `import lm_eval`
* `version`：包版本
* `authors`：作者信息
* `description`：一句话描述
* `readme`：PyPI 上展示的长描述内容来自哪（README.md）
* `classifiers`：给 PyPI 用的标签（开发状态、支持的 Python 版本、协议）
* `requires-python`：要求 Python 版本 ≥ 3.10
* `license`：许可证，这里是 MIT
* `dependencies`：**安装这个包时会一并安装的依赖**

比如你 `pip install lm_eval` 时，这一堆都会跟着装（`torch`, `transformers`, `datasets` 等）。

---

### 3. `setuptools` 的额外配置

#### 3.1 找哪些包

```toml
[tool.setuptools.packages.find]
include = ["lm_eval*"]
```

告诉 setuptools：

* 在项目里寻找以 `lm_eval` 开头的包目录（`lm_eval`, `lm_eval.tasks` 之类）

#### 3.2 把哪些非 Python 文件也打包进去

```toml
[tool.setuptools.package-data]
lm_eval = ["**/*.yaml", "tasks/**/*"]
```

意思是：

* 在 `lm_eval` 包里，把：

  * 所有 `yaml` 文件
  * `tasks` 下面的所有文件
* 都打进 wheel 里（否则 pip 安装后这些数据文件可能丢失）

---

### 4. `[project.scripts]` —— 命令行可执行程序

```toml
[project.scripts]
lm-eval = "lm_eval.__main__:cli_evaluate"
lm_eval = "lm_eval.__main__:cli_evaluate"
```

这表示：安装这个包后，你可以在终端里直接用：

* `lm-eval`
* 或 `lm_eval`

这两个命令，都会调用 `lm_eval.__main__.py` 里的 `cli_evaluate` 函数。
也就是说这是 **命令行入口** 的配置。

---

### 5. `[project.urls]` —— 项目的链接

```toml
[project.urls]
Homepage = "https://github.com/EleutherAI/lm-evaluation-harness"
Repository = "https://github.com/EleutherAI/lm-evaluation-harness"
```

就是给 PyPI 或文档用的主页、仓库链接。

---

### 6. `[project.optional-dependencies]` —— 可选依赖（extras）

这一段很重要：

```toml
[project.optional-dependencies]
api = ["requests", "aiohttp", "tenacity", "tqdm", "tiktoken"]
...
vllm = ["vllm>=0.4.2"]
wandb = ["wandb>=0.16.3", "pandas", "numpy"]
...
tasks = [
    "lm_eval[acpbench]",
    "lm_eval[discrim_eval]",
    ...
]
```

解释：

* 这些是 **“额外功能”需要的依赖**，不是所有用户都必须装

* 安装时，你可以选择：

  ```bash
  # 安装基础版本
  pip install lm_eval

  # 安装带 api 功能的
  pip install lm_eval[api]

  # 安装带 vllm 支持的
  pip install lm_eval[vllm]

  # 安装一整套与 tasks 相关的依赖
  pip install lm_eval[tasks]
  ```

* 比如：

  * `japanese_leaderboard`：日文榜单相关，需要日文分词等库
  * `math`：数学任务需要 `sympy` 等
  * `vllm`：如果你要用 vLLM 推理，就装这个 extra

`tasks` 这个 extra 有点“合集”的感觉，它本身依赖其它几个 extra（像 `lm_eval[math]`）。

---

### 7. `[tool.pymarkdown]` —— Markdown 规范配置

```toml
[tool.pymarkdown]
plugins.md013.enabled = false # line-length
...
```

* 这是给 `pymarkdown` 工具用的配置
* 比如：关闭行长度检查（md013）、允许重复标题等
  就是团队自己定的一些 Markdown 规则。

---

### 8. `[tool.ruff.lint]` / `...[tool.ruff.lint.isort]` —— Ruff 代码风格 / lint 配置

```toml
[tool.ruff.lint]
extend-select = ["I", "W605"]

[tool.ruff.lint.isort]
lines-after-imports = 2
known-first-party = ["lm_eval"]

[tool.ruff.lint.extend-per-file-ignores]
"__init__.py" = ["F401", "F402", "F403"]
"utils.py" = ["F401"]
```

* `ruff` 是一个非常快的 Python 静态检查+格式化工具
* 这里配置：

  * 开启额外的一些规则（例如导入顺序、转义 warning）
  * `isort` 风格：导入后空两行、`lm_eval` 视作第一方包
  * 某些文件（`__init__.py`, `utils.py`）忽略一些常见告警，比如：

    * `F401`：导入但未使用（`__init__` 里很常见）

---

### 9. `[dependency-groups]` —— 开发环境用的依赖组合

```toml
[dependency-groups]
dev = [
    "api", "dev", "sentencepiece"
]
```

这个部分是给一些更高级的依赖管理工具（如 `uv`、`rye`、`pdm` 等）用的：

* 定义一个 `dev` 组，里面引用了若干 optional deps：

  * `"api"`, `"dev"`, `"sentencepiece"`
    表示开发环境下你可能会一次性装这些组。

> 注意：这里的 `"api"`, `"dev"` 指的是 **上面 `[project.optional-dependencies]` 里的 key**，不是直接的包名。

---

## 三、你怎么用这些信息？

几个常见动作：

1. **安装基础版：**

   ```bash
   pip install .
   # 或者从 GitHub 直接
   pip install "git+https://github.com/EleutherAI/lm-evaluation-harness.git"
   ```

2. **加上某个 extra（例如 vLLM 支持）：**

   ```bash
   pip install "lm_eval[vllm]"
   ```

3. **使用命令行：**

   ```bash
   lm-eval --help
   # 或
   lm_eval --help
   ```

4. **查看包里包含哪些数据文件**：
   由 `package-data` 配置可知，`tasks/` 下的 YAML 和任务定义会一起被安装，这对你自定义任务/阅读任务配置有帮助。
