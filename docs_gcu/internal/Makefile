###############################################################
#
# Copyright 2020 The Enflame Tech Company. All Rights Reserved.
#
# Brief:
#   Build Installation documents
#
###############################################################

EFB_PKG_VER := $(EFB_PKG_VER) -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/_static:/web/content/source/_static \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/install:/web/content/source/install \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/introduction:/web/content/source/introduction \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/LLMs:/web/content/source/LLMs \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/usage:/web/content/source/usage \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/VLMs:/web/content/source/VLMs \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/logo.png:/web/content/source/logo.png \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/statement.rst:/web/content/source/statement.rst \
							  -v $(dir $(realpath $(lastword $(MAKEFILE_LIST))))../content/source/vllm_gcu_cn_header.rst:/web/content/source/vllm_gcu_cn_header.rst
include ../makefiles/doc.mk

