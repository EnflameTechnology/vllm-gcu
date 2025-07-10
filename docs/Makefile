###############################################################
#
# Copyright 2020 The Enflame Tech Company. All Rights Reserved.
#
# Brief:
#   Build Installation documents
#
###############################################################

# Makefile - for use inside the docs/ folder
# Assumes Sphinx source is at content/source

SPHINX_BUILD = sphinx-build
SOURCE_DIR   = content/source
BUILD_DIR    = content/_build
HTML_DIR     = $(BUILD_DIR)/html

CCBLUE  = "\033[34;01m"
CCEND   = "\033[0m"

help:
	@echo $(CCBLUE) "Available targets:" $(CCEND)
	@echo "  initdoc       - Initialize a new Sphinx project"
	@echo "  html          - Build HTML docs"
	@echo "  clean         - Remove build artifacts"
	@echo "  gh-pages      - Deploy HTML to GitHub Pages"

initdoc:
	sphinx-quickstart $(SOURCE_DIR)

html:
	$(SPHINX_BUILD) -b html $(SOURCE_DIR) $(HTML_DIR)

clean:
	rm -rf $(BUILD_DIR)

gh-pages: html
	cp -r $(HTML_DIR)/* ../..
	touch ../../.nojekyll
	@echo $(CCBLUE) "Deploying to GitHub Pages..." $(CCEND)
	@cd ../.. && \
		git checkout -B gh-pages && \
		git add . && \
		git commit -m "Deploy to GitHub Pages" && \
		git push origin gh-pages --force && \
		git checkout -


.PHONY: help initdoc html clean gh-pages
