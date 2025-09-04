## Qwen-VL

### Qwen2.5-VL-3B-Instruct

#### 模型下载
* url: [Qwen2.5-VL-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/files)
* branch: master
* commit id: 1b5a7674

- 将上述url设定的路径下的内容全部下载到`Qwen2.5-VL-3B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
python3 -m pip install transformers==4.53.2 triton==3.1.0
```

#### 环境变量

```
export VLLM_USE_V1=0
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --num-frames 6 \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Image \
 --dtype=bfloat16 \
 --batch-size=1 \
 --input-len=1200 \
 --input-vision-shape="1280,720" \
 --max-output-len=100 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --gpu-memory-utilization 0.9
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为128000；
