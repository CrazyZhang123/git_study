
```bash
echo "# LLM_assist" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:CrazyZhang123/LLM_assist.git
git push -u origin main
```


## tag使用
### 2️⃣ 创建 tag

假设要创建一个 **轻量 tag** `version1.0`：

```bash
git tag version1.0
```

- 默认会在 **当前最新 commit** 上创建 tag。
  

如果想加描述（**附注 tag / annotated tag**）：

```bash
git tag -a version1.0 -m "作业版本 1.0"
```
- `-a` 表示附注 tag。
  
- `-m` 后面是 tag 描述信息。
### 3️⃣ 查看已有 tag

```bash
git tag
```

会列出所有 tag，包括刚创建的 `version1.0`。

### 4️⃣ 推送 tag 到 GitHub

默认 tag **不会自动推送**，需要单独推送：

```bash
git push origin version1.0
```

- 如果想一次性推送所有本地 tag：

```bash
git push origin --tags
```

✅ 完成后，你的 GitHub 仓库就会显示 tag `version1.0`，对应你最后一次提交。

## git_filter_repo

### 问题：
- huggingface token等隐秘的文件，不应该被提交到git仓库中。(下面就是huggingface token被拦截的bug)
- 过大的文件也不应该被提交到git仓库中。

```bash
(base) PS D:\workspace\git_study> git push origin main --force  
Enumerating objects: 856, done.
Counting objects: 100% (856/856), done.
Delta compression using up to 12 threads
Compressing objects: 100% (722/722), done.
Writing objects: 100% (856/856), 324.15 MiB | 4.42 MiB/s, done.
Total 856 (delta 123), reused 833 (delta 114), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (123/123), done.
remote: error: GH013: Repository rule violations found for refs/heads/main.
remote:
remote: - GITHUB PUSH PROTECTION
remote:   —————————————————————————————————————————
remote:     Resolve the following violations before pushing again
remote:
remote:     - Push cannot contain secrets
remote:
remote:
remote:      (?) Learn how to resolve a blocked push
remote:      https://docs.github.com/code-security/secret-scanning/working-with-secret-scanning-and-push-protection/working-with-push-protection-from-the-command-line#resolving-a-blocked-push 
remote:
remote:
remote:       —— Hugging Face User Access Token ————————————————————
remote:        locations:
remote:          - blob id: 1b98197d01aaa920b347f3723ad39eec0a3a8089
remote:
remote:        (?) To push, remove secret from commit(s) or follow this URL to allow the secret.
remote:        https://github.com/CrazyZhang123/git_study/security/secret-scanning/unblock-secret/34bVRvM9urs9nWNgFVffinish on time.
remote:      It can still contain undetected secrets.
remote:
remote:      (?) Use the following command to find the path of the detected secret(s):
remote:          git rev-list --objects --all | grep blobid
remote:     ——————————————————————————————————————————————————————
remote:
remote:
To github.com:CrazyZhang123/git_study.git
 ! [remote rejected] main -> main (push declined due to repository rule violations)
error: failed to push some refs to 'github.com:CrazyZhang123/git_study.git'

```

### 命令讲解

```bash
git rev-list --objects --all | findstr 1b98197d01aaa920b347f3723ad39eec0a3a8089 
```

![image-20251027003721695](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20251027003721695.png)

把上面问题的id复制过来，进行查找，**记住一定要在git bash环境下，不如就会乱码**。

```
python -m git_filter_repo --invert-paths --force ^
  --paths "python学习/数据分析/code/pandas/joyfulpandas/data/11-13章数据集.zip" ^
  --paths "python学习/数据分析/code/pandas/joyfulpandas/data/ch4/marine_observation.csv" ^
  --paths "python学习/数据分析/code/pandas/joyfulpandas/data/jetbra_激活/激活教程.mp4"
```
这段代码是使用 `git-filter-repo` 工具对 Git 仓库进行历史修改的命令，主要作用是**从仓库的所有提交历史中永久除指定路径的文件/文件夹**，常用于清理仓库中不需要的大文件、敏感文件或冗余数据。


### 命令参数详解：
1. **`python -m git_filter_repo`**  
   通过 Python 执行 `git-filter-repo` 工具（需提前安装：`pip install git-filter-repo`）。


2. **`--invert-paths`**  
   核心参数，意为“反转路径筛选逻辑”：  
   - 若不加此参数，`--paths` 指定的是“要保留的路径”；  
   - 加上此参数后，`--paths` 指定的是“要删除的路径”。  


3. **`--force`**  
   强制执行命令，忽略一些安全提示（例如仓库有未提交的修改时，仍继续执行）。


4. **`--paths "路径1" --paths "路径2" ...`**  
   多次使用 `--paths` 指定要删除的文件/文件夹路径（**相对于仓库根目录**）：  
   - `"python学习/数据分析/code/pandas/joyfulpandas/data/11-13章数据集.zip"`  
   - `"python学习/数据分析/code/pandas/joyfulpandas/data/ch4/marine_observation.csv"`  
   - `"python学习/数据分析/code/pandas/joyfulpandas/data/jetbra_激活/激活教程.mp4"`  
- 用git bash去看，不要用powershell


### 命令作用：
执行后，Git 仓库的**所有历史提交记录**中，上述三个路径对应的文件会被彻底删除，且无法通过 `git log` 或 `git checkout` 恢复。修改后的仓库会保留其他文件的历史，仅移除指定路径的痕迹。


### 注意事项：
1. 执行前务必备份仓库，此操作会不可逆地修改历史提交。  
2. 若仓库已推送到远程（如 GitHub），执行后需用 **`git push --force` 强制覆盖远程仓库（会影响协作者，需提前沟通）**。  
3. 常用于清理大文件以减小仓库体积，或删除敏感信息（如激活文件、密码等）。

### 后续命令

#### ✅ 最后两步收尾：

#### 🔧 1. 重新添加远程仓库

在项目根目录执行：

```
git remote add origin git@github.com:CrazyZhang123/git_study.git
```

或者如果你用的是 HTTPS（非 SSH）：

```
git remote add origin https://github.com/CrazyZhang123/git_study.git
```

------

#### 🚀 2. 强制推送新历史（已清理干净）

```
git push origin main --force
```

------

执行完这两步后：

- 你就能成功 push；
- GitHub 的 secret 扫描不会再拦截；
- 大文件和敏感信息也都安全清理完毕。