# 代码部署
hexo clean
hexo generate
hexo deploy

# 代码备份
## 文件加载StagingArea。
git add .
## 查看当前状态
git status
## 提交代码LocalRepository
git commit -m "backup_source"
## 提交代码RemoteRepository
git push origin source