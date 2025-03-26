## glm

### glm-4v-9b

#### 模型下载
* url: [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b/tree/main)
* branch: main
* commit id: 01328faefe122fe605c1c127b62e6031d3ffebf7

- 将上述url设定的路径下的内容全部下载到`glm-4v-9b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of glm-4v-9b] \
 --prompt=[your prompt] \
 --input-vision-file=training_pipelines.jpg \
 --tensor-parallel-size 1 \
 --max-model-len=8192 \
 --max-output-len=128 \
 --dtype=bfloat16 \
 --device=gcu \
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
 --model=[path of glm-4v-9b] \
 --input-vision-shape=1120,1120 \
 --tensor-parallel-size=1 \
 --max-model-len=8192 \
 --input-len=4096 \
 --max-output-len=4096 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --batch-size=1 \
 --block-size=64 \
 --trust-remote-code \
 --gpu-memory-utilization 0.945
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为8192；