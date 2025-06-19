## qwen-vl

### Qwen2-VL-72B-Instruct

#### 模型下载
* url: [Qwen2-VL-72B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-VL-72B-Instruct)
* branch: master
* commit id: fdcb6167

- 将上述url设定的路径下的内容全部下载到`Qwen2-VL-72B-Instruct`文件夹中。


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2-VL-72B-Instruct] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --device=gcu \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --gpu-memory-utilization 0.945

```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of Qwen2-VL-72B-Instruct] \
 --model-arch-suffix Image \
 --dtype=bfloat16 \
 --num-prompts 1 \
 --batch-size=1 \
 --input-len=1200 \
 --input-vision-shape=1,3,1280,720 \
 --max-output-len=100 \
 --device=gcu \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --gpu-memory-utilization 0.945

```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；
