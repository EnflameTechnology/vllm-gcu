# Documentation Build & Preview Guide

This guide explains how to build the Sphinx documentation into HTML and run a local web server to preview the docs.

---

<p align="center">
  <a href="./ReadMe-EN.md">English</a> |
  <a href="./ReadMe.md">简体中文</a> |
</p>

## Prerequisites

- Python 3.8 or newer
- `pip` (Python package installer)

---

## Step 1: Install Required Python Packages

Install the necessary packages using `pip`:

```bash
python3 -m pip install sphinx sphinx_rtd_theme sphinx-multiversion myst-parser
````

> **Tip:** It's best to use a Python virtual environment to keep dependencies isolated:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
python3 -m pip install sphinx sphinx_rtd_theme sphinx-multiversion myst-parser
```

---

## Step 2: Build the HTML Documentation

From the `docs/` directory (where the `Makefile` is located), run:

```bash
make html
```

This generates the HTML output in `content/_build/html`.

---

## Step 3: Serve the Documentation Locally

Run a simple HTTP server to view your docs in a browser:

```bash
cd content/_build/html
python3 -m http.server 8000
```

Open your browser and navigate to:

```
http://localhost:8000
```

---

## Additional Notes

* To stop the HTTP server, press `Ctrl + C` in the terminal.
* If you encounter errors about missing Python packages, verify that you installed them in the correct environment.
* You can rebuild the docs anytime by repeating **Step 2**.

---

## Troubleshooting

* **`myst_parser` module not found:** Make sure `myst-parser` is installed.
* **Theme error `sphinx_rtd_theme` not found:** Install `sphinx_rtd_theme` package.
* Check your Python environment if commands fail.

---

## Contact & Support

For questions or issues, please open an issue in this repository or contact the maintainers.

---

