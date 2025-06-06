# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
# please set doc name
DOCNAME      = README

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
# Convert markdown to RST by pandoc
# 将markdown中的所有</br><br/>替换为 空行，写入一个新的markdown文件
# 	@cp source/$(DOCNAME).md source/$(DOCNAME)_new.md
# 	@sed -i -e 's/<\/br>/\n\r/g' -e 's/<br\/>/\n\r/g' source/$(DOCNAME)_new.md
# # 使用pandoc将新的markdown文件转成rst，--wrap=none代表不换行（目的是为了让转成的表格不换行用以生成不会变形的表格）
# 	@pandoc source/$(DOCNAME)_new.md --wrap=none --column=100 -f markdown -t rst -o source/$(DOCNAME).rst
# # 处理pandoc转出来的不符合rst要求的文档内部链接，例：`链接显示名称 <####标题名称>`__  替换为  `标题名称`_
# 	@sed -i -e 's/`.*<#\{1,\}/`/g' -e 's/>`__/`_/g' source/$(DOCNAME).rst
# # 删除过程中生成的新的markdown文件
# 	@rm -r source/$(DOCNAME)_new.md

	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
