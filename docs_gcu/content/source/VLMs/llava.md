## llava

### llava-1.5-7b-hf

#### 模型下载
* url: [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)
* branch: main
* commit id: fa3dd2809b8de6327002947c3382260de45015d4

- 将上述url设定的路径下的内容全部下载到`llava-1.5-7b-hf`文件夹中。

#### requirements

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 datasets==3.5.0
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of llava-1.5-7b-hf] \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --max-output-len=128 \
 --device=gcu
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--max-output-len`可按需调整；

#### 数据集测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --acc \
 --dataset-name=llava-bench-coco \
 --dataset-file=[path of dataset] \
 --model=[path of llava-1.5-7b-hf] \
 --device=gcu \
 --block-size=64 \
 --batch-size=[batch size]
```
注：
* 需将[llava-bench-coco](https://huggingface.co/datasets/lmms-lab/llava-bench-coco/blob/main/data/train-00000-of-00001.parquet)文件下载到本地，并设置`--dataset`指向其存储路径；
* `--save-output`设置推理结果的保存文件，后缀名为`json`。若不设置该参数，则不保存推理结果；
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；