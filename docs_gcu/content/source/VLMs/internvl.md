## internvl

### internvl-chat-v1-5

#### 模型下载
* url: [internvl-chat-v1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)
* branch: main
* commit id: c443cca6c074f663043b70dbc0e807ddde042124

- 将上述url设定的路径下的内容全部下载到`internvl-chat-v1-5`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of internvl-chat-v1-5] \
 --input-vision-file image1.jpg \
 --tensor-parallel-size 2 \
 --max-model-len=32768 \
 --max-output-len=128 \
 --device gcu \
 --batch-size=1 \
 --block-size=64 \
 --trust-remote-code \
 --max-num-seqs 1 \
 --prompt "请详细描述图片"
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[image1.jpg](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/examples/image1.jpg);

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of internvl-chat-v1-5] \
 --tensor-parallel-size=2 \
 --input-vision-shape 448,448 \
 --max-model-len=32768 \
 --input-len=4096 \
 --max-output-len=12288 \
 --device gcu \
 --batch-size=1 \
 --block-size=64 \
 --max-num-seqs 1 \
 --trust-remote-code
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### InternVL2-8B

#### 模型下载
* url: [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)
* branch: main
* commit id: 6f6d72be3c7a8541d2942691c46fbd075c147352

- 将上述url设定的路径下的内容全部下载到`InternVL2-8B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install decord
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model [path of InternVL2-8B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test image] \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model-arch-suffix Video \
 --model [path of InternVL2-8B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test video] \
 --num-frames 8 \
 --mm-per-prompt 8 \
 --max-output-len 512 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[image1.jpg](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/examples/image1.jpg);

#### 性能测试
##### 单图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2-8B] \
 --batch-size=1 \
 --input-len=1024 \
 --input-vision-shape="448,448" \
 --mm-per-prompt=1 \
 --max-output-len=512 \
 --trust-remote-code \
 --block-size 64 \
 --tensor-parallel-size 1 \
 --max-model-len 32768  \
 --gpu-memory-utilization 0.945 \
 --device gcu
```
##### 多图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2-8B] \
 --batch-size=1 \
 --input-len=1024 \
 --input-vision-shape="448,448;448,448" \
 --mm-per-prompt=2 \
 --max-output-len=512 \
 --trust-remote-code \
 --block-size 64 \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --gpu-memory-utilization 0.945 \
 --device gcu
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### InternVL2_5-2B

#### 模型下载
* url: [InternVL2_5-2B](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-2B/files)
* branch: master
* commit id: 96855603

- 将上述url设定的路径下的内容全部下载到`InternVL2_5-2B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install decord
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model [path of InternVL2_5-2B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test image] \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model-arch-suffix Video \
 --model [path of InternVL2_5-2B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test video] \
 --num-frames 8 \
 --mm-per-prompt 8 \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[image1.jpg](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/examples/image1.jpg);

#### 性能测试
##### 单图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2_5-2B] \
 --batch-size=1 \
 --input-len=1024 \
 --input-vision-shape="448,448" \
 --mm-per-prompt=1 \
 --max-output-len=512 \
 --trust-remote-code \
 --device=gcu \
 --block-size 64 \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --gpu-memory-utilization 0.945
```
##### 多图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2_5-2B] \
 --batch-size=1 \
 --input-len=1024 \
 --input-vision-shape="448,448;448,448" \
 --mm-per-prompt=2 \
 --max-output-len=512 \
 --trust-remote-code \
 --device=gcu \
 --block-size 64  \
 --tensor-parallel-size 1 \
 --max-model-len 32768  \
 --gpu-memory-utilization 0.945
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

### InternVL2_5-78B

#### 模型下载
* url: [InternVL2_5-78B](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-78B/files)
* branch: master
* commit id: 6c2cd57c

- 将上述url设定的路径下的内容全部下载到`InternVL2_5-78B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install decord
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model [path of InternVL2_5-78B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test image] \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --tensor-parallel-size 8
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model-arch-suffix Video \
 --model [path of InternVL2_5-78B] \
 --prompt [your prompt] \
 --input-vision-file [path of your test video] \
 --num-frames 8 \
 --mm-per-prompt 8 \
 --max-output-len 512 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --tensor-parallel-size 8
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[image1.jpg](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/examples/image1.jpg);

#### 性能测试
##### 单图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2_5-78B] \
 --batch-size=8 \
 --input-len=1024 \
 --input-vision-shape="448,448" \
 --mm-per-prompt=1 \
 --max-output-len=512 \
 --trust-remote-code \
 --device=gcu \
 --block-size 64 \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --gpu-memory-utilization 0.945
```
##### 多图性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of InternVL2_5-78B] \
 --batch-size=8 \
 --input-len=1024 \
 --input-vision-shape="448,448;448,448" \
 --mm-per-prompt=2 \
 --max-output-len=512 \
 --trust-remote-code \
 --device=gcu \
 --block-size 64 \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --gpu-memory-utilization 0.945
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；