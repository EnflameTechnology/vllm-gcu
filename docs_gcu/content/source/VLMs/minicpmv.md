## minicpmv

### MiniCPM-V-2_6

#### 模型下载
* url: [MiniCPM-V-2_6](https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-2_6/files)
* branch: main
* commit id: 92e829ff803337da801b16396faad9356ce2ed86

- 将上述url设定的路径下的内容全部下载到`MiniCPM-V-2_6`文件夹中。

#### 批量离线推理
##### 图像推理
```shell
python3.10 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of MiniCPM-V-2_6] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --max-num-seqs 64 \
 --dtype=float16
```
##### 视频推理
```shell
python3.10 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model=[path of MiniCPM-V-2_6] \
 --model-arch-suffix Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --num-frames 64 \
 --max-num-seqs 64 \
 --dtype=float16
```

注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--max-output-len`可按需调整；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language \
 --perf \
 --model=[path of MiniCPM-V-2_6] \
 --model-arch-suffix Image \
 --batch-size 1 \
 --max-output-len 128 \
 --device gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --input-vision-shape 1080,1080 \
 --input-len 1024 \
 --max-num-seqs 64 \
 --block-size 64 \
 --dtype=float16
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为32768；

#### 数据集测试

```shell
python3.10 -m vllm_utils.benchmark_vision_language \
 --acc \
 --model-arch-suffix Video \
 --dataset-name=videomme \
 --dataset-file=[path of Video-MME] \
 --model=[path of MiniCPM-V-2_6] \
 --max-output-len 128 \
 --device=gcu \
 --trust-remote-code \
 --max-model-len 32768 \
 --num-frames 8 \
 --max-num-seqs 32  \
 --dtype=float16
```
注：
* 需将[Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)文件下载到本地，并设置`--dataset-file`指向其存储路径；
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；