## qwen-vl

### Qwen-VL

#### 模型下载
* url: [Qwen-VL](https://www.modelscope.cn/models/Qwen/Qwen-VL/files)
* branch: master
* commit id: 71ab3b10

- 将上述url设定的路径下的内容全部下载到`Qwen-VL`文件夹中。

#### requirements

```shell
python3 -m pip install matplotlib
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen-VL] \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 2048 \
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
 --model=[path of Qwen-VL] \
 --dtype=bfloat16 \
 --num-prompts 4 \
 --batch-size=4 \
 --input-len=1024 \
 --input-vision-shape=448,448 \
 --max-output-len=1024 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 2048 \
 --trust-remote-code \
 --block-size=64
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为2048；

### Qwen2-VL-2B-Instruct

#### 模型下载
* url: [Qwen2-VL-2B-Instruct](https://www.modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct)
* branch: master
* commit id: 103fa047a85cdce37dce2e17a0f00d1ab13ed1f2

- 将上述url设定的路径下的内容全部下载到`Qwen2-VL-2B-Instruct`文件夹中。

#### requirements

```shell
python3 -m pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
--backend vllm \
--demo \
--model=[path of Qwen2-VL-2B-Instruct] \
--model-arch-suffix Image \
--prompt=[your prompt] \
--input-vision-file=[path of your test image] \
--dtype=float16 \
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
 --model=[path of Qwen2-VL-2B-Instruct] \
 --model-arch-suffix Image \
 --dtype=float16 \
 --num-prompts 1 \
 --batch-size=1 \
 --input-len=1200 \
 --input-vision-shape=1280,720 \
 --max-output-len=100 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### Qwen2-VL-7B-Instruct-GPTQ-Int4

#### 模型下载
* url: [Qwen2-VL-7B-Instruct-GPTQ-Int4](https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4)
* branch: master
* commit id: c12b1f54af3eaeff5f9c6fc5707160c825e83cde

- 将上述url设定的路径下的内容全部下载到`Qwen2-VL-7B-Instruct-GPTQ-Int4`文件夹中。

#### requirements

```shell
python3 -m pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
--backend vllm \
--demo \
--model=[path of Qwen2-VL-7B-Instruct-GPTQ-Int4] \
--model-arch-suffix Image \
--prompt=[your prompt] \
--input-vision-file=[path of your test image] \
--dtype=float16 \
--max-output-len=128 \
--device=gcu \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--trust-remote-code \
--block-size=64 \
--quantization gptq
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
--backend vllm \
--perf \
--model=[path of Qwen2-VL-7B-Instruct-GPTQ-Int4] \
--model-arch-suffix Image \
--dtype=float16 \
--num-prompts 1 \
--batch-size=1 \
--input-len=1200 \
--input-vision-shape=1280,720 \
--max-output-len=100 \
--device=gcu \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--trust-remote-code \
--block-size=64 \
--quantization gptq
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### Qwen2-VL-72B-Instruct-GPTQ-Int8

#### 模型下载
* url: [Qwen2-VL-72B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8/files)
* branch: master
* commit id: dbbe45d0

- 将上述url设定的路径下的内容全部下载到`Qwen2-VL-72B-Instruct-GPTQ-Int8`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm --demo \
 --model=[path of Qwen2-VL-72B-Instruct-GPTQ-Int8] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=float16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --quantization gptq
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2-VL-72B-Instruct-GPTQ-Int8] \
 --model-arch-suffix Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --num-frames 12 \
 --dtype=float16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --quantization gptq
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试
##### 单图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of Qwen2-VL-72B-Instruct-GPTQ-Int8] \
 --model-arch-suffix Image \
 --dtype=float16 \
 --batch-size=32 \
 --input-len=1500 \
 --input-vision-shape="448,448" \
 --max-output-len=4096 \
 --device=gcu \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --max-seq-len-to-capture 32768 \
 --trust-remote-code \
 --block-size=64 \
 --quantization gptq \
 --gpu-memory-utilization 0.945
```
##### 多图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of Qwen2-VL-72B-Instruct-GPTQ-Int8] \
 --model-arch-suffix Image \
 --dtype=float16 \
 --batch-size=13 \
 --input-len=1500 \
 --input-vision-shape="448,448;448,448" \
 --mm-per-prompt=2 \
 --max-output-len=8192 \
 --device=gcu \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --max-seq-len-to-capture 32768 \
 --trust-remote-code \
 --block-size=64 \
 --quantization gptq \
 --gpu-memory-utilization 0.9
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### Qwen2.5-VL-3B-Instruct

#### 模型下载
* url: [Qwen2.5-VL-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/files)
* branch: master
* commit id: 1b5a7674

- 将上述url设定的路径下的内容全部下载到`Qwen2.5-VL-3B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers>=4.50.1 opencv-python==4.11.0.86
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

### QVQ-72B-Preview

#### 模型下载
* url: [QVQ-72B-Preview](https://www.modelscope.cn/models/Qwen/QVQ-72B-Preview/files)
* branch: master
* commit id: be7cda0c

- 将上述url设定的路径下的内容全部下载到`QVQ-72B-Preview`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of QVQ-72B-Preview] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=8196 \
 --device=gcu \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of QVQ-72B-Preview] \
 --model-arch-suffix Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --num-frames 12 \
 --dtype=bfloat16 \
 --max-output-len=8196 \
 --device=gcu \
 --tensor-parallel-size 8 \
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
 --model=[path of QVQ-72B-Preview] \
 --model-arch-suffix Image \
 --dtype=bfloat16 \
 --batch-size=1 \
 --input-len=2048 \
 --input-vision-shape="1280,720" \
 --max-output-len=2048 \
 --device=gcu \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --gpu-memory-utilization 0.9
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为128000；

