## phi3v

### phi-3-vision-128k-instruct

#### 模型下载
* url: [phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/tree/main)
* branch: main
* commit id: c45209e90a4c4f7d16b2e9d48503c7f3e83623ed

- 将上述url设定的路径下的内容全部下载到`phi-3-vision-128k-instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language --demo \
 --model-arch-suffix Image \
 --model=[path of phi-3-vision-128k-instruct] \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --max-model-len=32768 \
 --dtype=bfloat16 \
 --gpu-memory-utilization=0.9 \
 --max-output-len=256 \
 --trust-remote-code \
 --device=gcu
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--max-output-len`可按需调整；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language --perf \
 --model-arch-suffix Image \
 --model=[path of phi-3-vision-128k-instruct] \
 --max-model-len=32768 \
 --dtype=bfloat16 \
 --input-vision-shape="1024,1024" \
 --gpu-memory-utilization=0.9 \
 --block-size=64 \
 --input-len=8192 \
 --max-output-len=1024 \
 --batch-size=1 \
 --trust-remote-code \
 --device=gcu
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--batch-size`、`--input-len`、`--input-vision-shape`、`--max-output-len`可按需调整；
* 本模型支持的`max-model-len`为131072，单张S60可以支持32768；