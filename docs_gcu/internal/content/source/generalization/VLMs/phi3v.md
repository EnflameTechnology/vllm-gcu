## Phi3v

### Phi-3.5-vision-instruct

本模型推理及性能测试需要1张enflame gcu。

#### 测试环境
- S60 daily: 3.3.20250112
- vllm: 0.6.1.post2

#### 模型下载
* url: [Phi-3.5-vision-instruct](https://www.modelscope.cn/models/LLM-Research/Phi-3.5-vision-instruct/files)
* branch: master
* commit id: 1ce2adc1

- 将上述url设定的路径下的内容全部下载到`Phi-3.5-vision-instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Phi-3.5-vision-instruct] \
 --model-arch-suffix=Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=1 \
 --max-output-len=256 \
 --gpu-memory-utilization=0.945 \
 --trust-remote-code \
 --device=gcu
```
S60输出结果
```shell
Prompt: 'What is the content of this image in details?', Generated text:  The image presents a scene set against a backdrop of a starry night sky. Dominating the foreground is a large, mechanical robot with a predominantly gray and black color scheme. The robot's head is equipped with a large, round, blue-tinted eye, giving it a somewhat sentient appearance.



The robot's body is a complex assembly of various mechanical parts, including a large arm and a smaller head. The arm, which is attached to the robot's back, is designed to hold a large, cylindrical object. This object, which is also gray and black, is attached to the robot's back with a strap.

The robot is situated on a rocky surface, suggesting a rugged, outdoor environment. The rocks are of various sizes and shapes, adding to the overall complexity of the scene.

The image does not contain any discernible text or other objects. The relative positions of the objects are such that the robot is in the foreground, with the rocky surface beneath it. The starry sky forms the background of the image. The robot's large eye is positioned towards the top of the image, while the cylindrical object is
```
L40s输出结果
```shell
Prompt: 'What is the content of this image in details?', Generated text:  The image presents a scene set against a backdrop of a starry night sky. Dominating the foreground is a large, mechanical robot with a predominantly gray and black color scheme. The robot's head is equipped with a large, round, blue-tinted eye, giving it a somewhat sentient appearance.



The robot's body is a complex assembly of various mechanical parts, including a large arm and a smaller head. The arm, which is attached to the robot's back, is designed to hold a large, cylindrical object. This object, which is also gray and black, is attached to the robot's back with a strap.

The robot is situated on a rocky surface, suggesting a rugged, outdoor environment. The rocks are of various sizes and shapes, adding to the overall complexity of the scene.

The image does not contain any discernible text or other objects. The relative positions of the objects suggest that the robot is the main subject of the image, with the rocky surface serving as a base. The large, round, blue-tinted eye of the robot is positioned towards the top of the image, drawing the viewer's attention
```

#### 精度测试
- dataset: mmmu
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --acc \
 --dataset-name=MMMU \
 --dataset-file=[path of mmmu] \
 --max-output-len=10 \
 --model-arch-suffix=Image \
 --model=[path of Phi-3.5-vision-instruct] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=1 \
 --batch-size=1 \
 --gpu-memory-utilization=0.945 \
 --trust-remote-code \
 --device=gcu
```
精度对比

|    device    |     mmlu     |
|--------------|--------------|
|      S60     |     0.428    |
|      L40s    |     0.429    |


#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --backend vllm \
 --model=[path of Phi-3.5-vision-instruct] \
 --tokenizer=[path of Phi-3.5-vision-instruct] \
 --model-arch-suffix=Image \
 --input-vision-shape=1024,1024 \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=1 \
 --batch-size=1 \
 --block-size=64 \
 --input-len=8192 \
 --max-output-len=1024 \
 --gpu-memory-utilization=0.945 \
 --trust-remote-code \
 --device=gcu
```

性能对比

|  device  |  card  | max-model-len | input-len | output-len | num-prompts |    TTFT    |    TPS    | prefill latency | decode latency|
|----------|--------|---------------|-----------|------------|-------------|------------|-----------|-----------------|---------------|
|    S60   |    1   |      32k      |    8192   |     1024   |      1      |   1046.49  |   27.75   |     1033.01     |     35.05     |
|    L40s  |    1   |      32k      |    8192   |     1024   |      1      |   464.41   |   43.26   |     802.93      |     22.34     |
|    S60   |    1   |      32k      |    8192   |     1024   |      4      |   7846.19  |   50.09   |     7740.04     |     72.32     |
|    L40s  |    1   |      32k      |    8192   |     1024   |      4      |   2598.19  |   112.76  |     1938.88     |     33.57     |
|    S60   |    1   |      32k      |    31744  |     1024   |      4      |   7024.10  |   13.51   |     6888.30     |     67.26     |
|    L40s  |    1   |      32k      |    31744  |     1024   |      4      |   2720.11  |   29.52   |     2581.75     |     31.27     |

注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--batch-size`、`--input-len`、`--input-vision-shape`、`--max-output-len`可按需调整；
* 本模型支持的`max-model-len`为131072，单张S60可以支持32768；

#### Wiki页面

- http://wiki.enflame.cn/pages/viewpage.action?pageId=260097920
