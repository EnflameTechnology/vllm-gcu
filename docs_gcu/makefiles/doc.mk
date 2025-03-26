ROOT_DIR         := $(CURDIR)
EFB_PKG_VER      ?=

# change to local image from remote sphinx-server-u:1.0
docs_image       := artifact.enflame.cn/enflame_docker_images/sphinx-server-u:1.6
#2.3.1
docs_image_ver   := 4.1.2
doc_dir          ?= content
port             ?= 8008
docker_env       := -e DOC_ROOT=$(ROOT_DIR) -e EFB_PKG_VER=$(EFB_PKG_VER)
mpath            := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

CCRED  :="\033[0;31m"
CCBLUE :="\033[34;01m"
CCEND  :="\033[0m"

help:
	@echo $(CCBLUE) "Supported Make Targets :" $(CCEND)
	@echo "  -- build-doc-server-image"
	@echo "  -- build-doc-server-image-force"
	@echo "  -- start-doc-server"
	@echo "  -- stop-doc-server"
	@echo "  -- restart-doc-server"
	@echo "  -- attach-doc-server"
	@echo "  -- multi-version-docs"
	@echo "  -- initdoc"
	@echo "  -- html"
	@echo "  -- pdf"
	@echo "  -- man"

initdoc:
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web -it $(docs_image) bash -c "sphinx-quickstart"
html: build-pull-server-image
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make html -C $(doc_dir)"
man:
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make man -C $(doc_dir)"
pdf: build-pull-server-image
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make latexpdf -C $(doc_dir)"
multi-version-docs:
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -v "$(shell pwd)":/web $(docs_image) bash -c "sphinx-versioning build $(doc_dir) $(doc_dir)/_build/"
host-html: 
	make html -C $(doc_dir)
pdf-doxygen-runtime: build-pull-server-image
	@if [ -e content/source/api ]; then rm -r content/source/api; fi
	cp -R ../../sdk/runtime/include ./
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make latexpdf -C $(doc_dir)"
	rm -r include
	rm -r _doxygen
html-doxygen-runtime: build-pull-server-image
	@if [ -e content/source/api ]; then rm -r content/source/api; fi
	cp -R ../../sdk/runtime/include ./
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make html -C $(doc_dir)"
	rm -r include
	rm -r _doxygen
host-html-doxygen-runtime: 
	@if [ -e content/source/api ]; then rm -r content/source/api; fi
	cp -R ../../sdk/runtime/include ./
	make html -C $(doc_dir)
	rm -r include
	rm -r _doxygen
pdf-c++: build-pull-server-image
	@if [ -e content/source/api ]; then rm -r content/source/api; fi
# 将c++代码拷贝至文档的content路径下（为了将代码能够同步映射到docker中）
# CODE_DIR 为tops下的c++代码存放路径，例如，factor，则CODE_DIR=sdk/include/factor
# CODE_FOLDER 为tops下的c++代码文件夹名称，例如，factor，则CODE_FOLDER=factor
	cp -R ../../$(CODE_DIR) ./content
	docker run --network host $(docker_env) -v $(ROOT_DIR):$(ROOT_DIR) -e USER=$(shell id -nu) -u $(shell id -u):$(shell id -g) -v "$(shell pwd)":/web $(docs_image) bash -c "make latexpdf -C $(doc_dir)"
# 删除临时生成的doxygen文件
	rm -r _doxygen
# 将前面拷贝的content下的代码删除
	@if [ -e content/$(CODE_FOLDER) ]; then rm -r content/$(CODE_FOLDER); fi
host-pdf-c++:
	@if [ -e content/source/api ]; then rm -r content/source/api; fi
# 将c++代码拷贝至文档的content路径下（为了将代码能够同步映射到docker中）
# CODE_DIR 为tops下的c++代码存放路径，例如，factor，则CODE_DIR=sdk/include/factor
# CODE_FOLDER 为tops下的c++代码文件夹名称，例如，factor，则CODE_FOLDER=factor
	cp -R ../../$(CODE_DIR) ./content
	make latexpdf -C $(doc_dir)
# 删除临时生成的doxygen文件
	rm -r _doxygen
# 将前面拷贝的content下的代码删除
	@if [ -e content/$(CODE_FOLDER) ]; then rm -r content/$(CODE_FOLDER); fi

start-doc-server: build-doc-server-image
	@xid=$(shell docker container ps -q  -f name=$(docs_image)-$(port)) ; if [ ! -z "$${xid}" ]; then echo $(CCRED) "sphinx doc server:  -- $(docs_image)-$(port) -- is running already, please run 'make start-doc-server port=????'" $(CCEND) && exit 1; fi
	docker run -d $(docker_env) -v $(ROOT_DIR):${ROOT_DIR} -v "$(shell pwd)/${doc_dir}/source":/web -u $(id -u):$(id -g) -p $(port):8000 --name $(docs_image)-$(port) $(docs_image)

	@IP=$(shell hostname -I | awk '{print $$1}') ; echo $(CCBLUE) "please go http://$${IP}:$(port) to checkout html doc" $(CCEND)

stop-doc-server:
	@if [ -e docs ]; then rm docs; fi
	@xid=$(shell docker container ps -q  -f name=$(docs_image)-$(port)) ; if [ ! -z "$${xid}" ]; then docker container kill $${xid} ; fi
	@xid=$(shell docker container ps -qa -f name=$(docs_image)-$(port)) ; if [ ! -z "$${xid}" ]; then docker container   rm $${xid} ; fi

restart-doc-server: stop-doc-server start-doc-server

attach-doc-server:
	docker exec -it $(shell docker ps -q -f name=$(docs_image)-$(port)) bash

docker-env: /usr/bin/docker

/usr/bin/docker :
	@echo "docker engine is not installed, On debian system, please install like this 'sudo apt install docker.io -y'" && exit 1

build-pull-server-image: docker-env
	if [ -z "$(shell docker image ls $(docs_image) -q)" ] ; then docker pull $(docs_image) ;fi

build-doc-server-image:  docker-env
	if [ -z "$(shell docker image ls $(docs_image) -q)" ] ; then docker build --rm --network host -t $(docs_image) -f $(mpath)../docs/sphinx-server/$(docs_image_ver)/Dockerfile $(mpath)../docs/sphinx-server ;fi

build-doc-server-image-force: build-pull-server-image docker-env
	docker build -q --rm --network host -t $(docs_image) -f sphinx-server/${docs_image_ver}/Dockerfile sphinx-server

.PHONY: initdoc html man pdf multi-version-docs start-doc-server
.PHONY: stop-doc-server restart-doc-server stop-doc-server
.PHONY: attach-doc-server docker-env build-doc-server-image
.PHONY: build-doc-server-image-force
