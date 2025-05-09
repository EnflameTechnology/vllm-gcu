## got-ocr

### GOT-OCR-2.0-hf

#### 模型下载
* url: [GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf/tree/main)
* branch: main
* commit id: d3017ef2c2c1395888c8d635c5e0508bcb0ac78d

- 将上述url设定的路径下的内容全部下载到`GOT-OCR-2.0-hf`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of GOT-OCR-2.0-hf] \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=1024 \
 --device=gcu \
 --tensor-parallel-size 1 \
 --max-model-len 2048 \
 --trust-remote-code \
 --block-size=64 \
 --max-num-batched-tokens 2048
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of GOT-OCR-2.0-hf] \
 --num-prompts 1 \
 --batch-size=1 \
 --input-len=1500 \
 --input-vision-shape=1024,1024 \
 --max-output-len=548 \
 --device=gcu \
 --dtype=bfloat16 \
 --tensor-parallel-size 1 \
 --max-model-len 2048 \
 --trust-remote-code \
 --block-size=64 \
 --max-num-batched-tokens 2048
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；