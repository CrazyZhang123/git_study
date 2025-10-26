
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