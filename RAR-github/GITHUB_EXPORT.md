# GitHub 发布包说明

本目录由工作区自动导出到 **`../RAR-github`**，仅含适合上传 GitHub 的源码（不含数据与 checkpoint）。

## 重新生成

```bash
bash scripts/export_for_github.sh
```

默认输出：`/home/wuyuncheng/RAR-github`（与主仓库同级）。

## 推送 GitHub

```bash
cd ../RAR-github
cp configs/config.example.yaml configs/config.yaml
git init && git add . && git commit -m "Initial release"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
