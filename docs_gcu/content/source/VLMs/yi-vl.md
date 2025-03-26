## yi-vl

### Yi-VL-6B

#### 模型下载
* url: [Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B/tree/main)
* branch: main
* commit id: dab34dabd32b391e4e870b7985180f90f79ad9a0

- 将上述url设定的路径下的内容全部下载到`Yi-VL-6B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of Yi-VL-6B] \
 --input-vision-file=[path of training_pipelines.jpg] \
 --prompt=[your prompt] \
 --tensor-parallel-size=1 \
 --max-model-len=4096 \
 --max-output-len=128 \
 --dtype=bfloat16 \
 --device gcu \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[training_pipelines.jpg](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/images/training_pipelines.jpg);

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of Yi-VL-6B] \
 --input-vision-shape=448,448 \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --input-len=2048 \
 --max-output-len=2048 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 4 \
 --batch-size 4 \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为4096;
