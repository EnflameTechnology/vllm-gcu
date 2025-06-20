## llava-next

### llama3-llava-next-8b-hf

#### 模型下载
* url: [llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf)
* branch: main
* commit id: 4b8c68ff38c0e76794018154c6df3ca308b0f76c

- 将上述url设定的路径下的内容全部下载到`llama3-llava-next-8b-hf`文件夹中。

#### requirements

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of llama3-llava-next-8b-hf] \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --max-output-len=128 \
 --device=gcu
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--max-output-len`可按需调整；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of llama3-llava-next-8b-hf] \
 --batch-size=1 \
 --input-len=2048 \
 --input-vision-shape="224,224" \
 --max-output-len=1024 \
 --device=gcu \
 --block-size=64
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--batch-size`、`--input-len`、`--input-vision-shape`、`--max-output-len`可按需调整；

### llava-onevision-qwen2-72b-ov-chat-hf

#### 模型下载
* url: [llava-onevision-qwen2-72b-ov-chat-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-chat-hf)
* branch: main
* commit id: 70606ce

- 将上述url设定的路径下的内容全部下载到`llava-onevision-qwen2-72b-ov-chat-hf`文件夹中。

#### requirements

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
```

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --model=[path of llava-onevision-qwen2-72b-ov-chat-hf] \
 --demo --model-arch-suffix=Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --device=gcu \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --gpu-memory-utilization=0.945 \
 --max-output-len=2048
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --model=[path of llava-onevision-qwen2-72b-ov-chat-hf] \
 --demo \
 --model-arch-suffix=Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --device=gcu \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --gpu-memory-utilization=0.945 \
 --max-output-len=2048
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--max-output-len`可按需调整；

#### 性能测试
##### 图像性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --model=[path of llava-onevision-qwen2-72b-ov-chat-hf] \
 --perf \
 --model-arch-suffix=Image \
 --batch-size=1 \
 --input-len=2048 \
 --max-output-len=2048 \
 --input-vision-shape=384,384 \
 --device=gcu \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --block-size=64 \
 --gpu-memory-utilization=0.945
```

##### 视频性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --model=[path of llava-onevision-qwen2-72b-ov-chat-hf] \
 --perf \
 --model-arch-suffix=Video \
 --batch-size=1 \
 --input-len=4096 \
 --max-output-len=2048 \
 --device=gcu \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --block-size=64 \
 --gpu-memory-utilization=0.945
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--batch-size`、`--input-len`、`--input-vision-shape`、`--max-output-len`可按需调整；