# 安装
## 软硬件需求
* OS：ubuntu 20.04 & 22.04
* Python：3.9 - 3.12
* 加速卡：燧原S60

## 安装内容

以下步骤基于拟使用的 `Python3`版本, 请先安装对应python3版本的所需依赖，需要在**docker**内安装：

* 安装环境：安装过程请参考《TopsRider软件栈安装手册》，请根据手册完成TopsRider软件栈安装;

### vllm安装

首先通过如下命令检查vllm及相关依赖是否已经安装：
```shell
python3 -m pip list | grep vllm_gcu
python3 -m pip list | grep vllm
python3 -m pip list | grep xformers
python3 -m pip list | grep tops-extension
python3 -m pip list | grep torch-gcu
python3 -m pip list | grep torch
python3 -m pip list | grep torchvision
```
如果已经正常安装，在`x86_64`架构下可以显示如下内容：
```
vllm_gcu                          0.8.0+<version>
vllm                              0.8.0
xformers                          <version>
tops-extension                    <version>
torch-gcu                         2.6.0+<version>
torch                             2.6.0+cpu
torchvision                       0.21.0+cpu
```
如果未安装，可以通过以下两种安装方式完成vllm安装：
* 通过TopsRider安装：
```shell
./Topsrider_xxx.run -y -C vllm-gcu
```
* 通过whl包安装：
```shell
# 安装vllm_gcu的依赖库
python3 -m pip install vllm==0.8.0
python3 -m pip install torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu
python3 -m pip install torchvision==0.21.0 -i https://download.pytorch.org/whl/cpu
python3 -m pip install torch_gcu-2.6.0+<version>*.whl
python3 -m pip install tops_extension-<version>*.whl
python3 -m pip install xformers-<version>*.whl

# 安装vllm_gcu库
python3 -m pip install vllm_gcu-0.8.0+<version>*.whl
```
