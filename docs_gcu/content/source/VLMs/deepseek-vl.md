## deepseek-vl

### deepseek-vl-7b-chat

#### 模型下载
* url: [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/tree/main)
* branch: main
* commit id: 6f16f00805f45b5249f709ce21820122eeb43556

- 将上述url设定的路径下的内容全部下载到`deepseek-vl-7b-chat`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of deepseek-vl-7b-chat] \
 --prompt=[your prompt] \
 --input-vision-file=[path of training_pipelines.jpg] \
 --tensor-parallel-size=1 \
 --max-model-len=16384 \
 --max-output-len=128 \
 --dtype=bfloat16 \
 --device gcu \
 --block-size=64 \
 --batch-size=1 \
 --max-num-seqs 1 \
 --trust-remote-code
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 示例图片下载地址为[training_pipelines.jpg](https://github.com/deepseek-ai/DeepSeek-VL/blob/main/images/training_pipelines.jpg);

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of deepseek-vl-7b-chat] \
 --input-vision-shape=1024,1024 \
 --tensor-parallel-size 1 \
 --max-model-len=16384 \
 --input-len=4096 \
 --max-output-len=4096 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --batch-size 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --max-num-seqs 1 \
 --trust-remote-code
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为16384；