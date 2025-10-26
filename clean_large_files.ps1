# clean_large_files.ps1
# 功能：从 Git 历史中移除大文件目录，但保留本地文件；并设置 .gitignore

$repoRoot = Get-Location
$excludePath = "python学习/数据分析/code/"

Write-Host "[1/5] 检查是否在 Git 仓库中..." -ForegroundColor Cyan
if (!(Test-Path ".git")){
    Write-Error "错误：当前目录不是 Git 仓库！请在仓库根目录运行此脚本。"
    exit 1
}

Write-Host "[2/5] 安装 git-filter-repo（如果尚未安装）..." -ForegroundColor Cyan
if (!(Get-Command git-filter-repo -ErrorAction SilentlyContinue)) {
    Write-Host "正在通过 pip 安装 git-filter-repo..."
    pip install git-filter-repo
    if ($LASTEXITCODE -ne 0) {
        Write-Error "安装 git-filter-repo 失败，请手动运行：pip install git-filter-repo"
        exit 1
    }
}

# 创建备份分支
Write-Host "[3/5] 创建备份分支 backup/pre-clean ..." -ForegroundColor Cyan
git checkout -b backup/pre-clean 2>$null
git checkout - 2>$null

Write-Host "[4/5] 从 Git 历史中移除路径：$excludePath（保留本地文件）..." -ForegroundColor Cyan
git filter-repo --path "$excludePath" --invert-paths --force

Write-Host "[5/5] 添加 .gitignore 并提交..." -ForegroundColor Cyan
$gitignoreFile = ".gitignore"
if (Test-Path $gitignoreFile) {
    $content = Get-Content $gitignoreFile
    if ($content -notcontains $excludePath) {
        Add-Content $gitignoreFile $excludePath
    }
} else {
    Set-Content $gitignoreFile $excludePath
}
git add .gitignore
git commit -m "chore: ignore large data directory"

Write-Host "`n✅ 清理完成！" -ForegroundColor Green
Write-Host "下一步操作建议：" -ForegroundColor Yellow
Write-Host "  1. 检查本地文件是否完好（$excludePath 应仍在）"
Write-Host "  2. 强制推送：git push origin main --force"
Write-Host "  3. 如果推送成功，可删除备份分支：git branch -D backup/pre-clean"