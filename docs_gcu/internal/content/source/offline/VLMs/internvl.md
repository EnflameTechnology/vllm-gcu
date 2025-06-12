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
