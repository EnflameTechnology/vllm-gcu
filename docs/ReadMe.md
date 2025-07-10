# 文档构建与预览指南

本指南介绍如何将 Sphinx 文档构建成 HTML，并使用本地 Web 服务器预览文档。

---

<p align="center">
  <a href="./ReadMe-EN.md">English</a> |
  <a href="./ReadMe.md">简体中文</a> |
</p>

## 前提条件

- Python 3.8 及以上版本
- `pip`（Python 包管理工具）

---

## 第一步：安装所需的 Python 包

使用 `pip` 安装必需的包：

```bash
python3 -m pip install sphinx sphinx_rtd_theme sphinx-multiversion myst-parser
````

> **提示：** 建议使用 Python 虚拟环境隔离依赖，避免影响系统环境：

```bash
python3 -m venv venv
source venv/bin/activate   # Windows 用户请执行：venv\Scripts\activate
python3 -m pip install sphinx sphinx_rtd_theme sphinx-multiversion myst-parser
```

---

## 第二步：构建 HTML 文档

在 `docs/` 目录下（即 `Makefile` 所在目录），运行：

```bash
make html
```

生成的 HTML 文件会输出到 `content/_build/html` 目录。

---

## 第三步：本地启动文档预览服务器

运行简单的 HTTP 服务器，在浏览器中查看文档：

```bash
cd content/_build/html
python3 -m http.server 8000
```

打开浏览器访问：

```
http://localhost:8000
```

---

## 其他说明

* 停止服务器请按 `Ctrl + C`。
* 如果遇到缺少 Python 包的错误，请确认依赖是否已正确安装。
* 需要重新构建文档时，重复执行第二步。

---

## 常见问题排查

* **找不到 `myst_parser` 模块**：请确认已安装 `myst-parser`。
* **找不到主题 `sphinx_rtd_theme`**：请确认已安装 `sphinx_rtd_theme`。
* 执行命令失败时请检查 Python 环境。

---

## 联系与支持

如有问题或建议，欢迎在仓库中提交 issue 或联系维护者。
