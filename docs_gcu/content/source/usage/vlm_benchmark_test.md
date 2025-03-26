## 视觉大语言模型测试

`benchmark_vision_language`是Enflame新增的测试入口，使用方式为`python3 -m vllm_utils.benchmark_vision_language arg1 arg2 ...`。可参考第6章查看各视觉大语言模型的推理指令。

功能上，支持：
* 使用输入的prompt和视觉文件进行离线推理以展示Enflame gcu对视觉大语言模型的推理能力；
* 使用指定长度的伪prompt和像素值全0的视觉文件进行视觉大语言模型的推理性能测试；
* 使用指定的数据集进行视觉大语言模型的推理精度测试；

可以通过`python3 -m vllm_utils.benchmark_vision_language --help`查看各参数含义。

### 离线推理测试
离线推理测试，是使用输入的prompt和视觉输入文件完成指定vision language model的推理，以演示Enflame gcu的推理能力。

使用上，需要：
* 设定`--demo`以启用离线推理测试；
* 通过`--prompt`设定输入prompt，测试中会对其自行应用模型的`chat template`；
* 通过`--input-vision-file`设定输入的视觉文件：
    * 对仅需要单张图像的vision language model，该参数指向图像文件；
    * 对需要多张图像的vision language model，该参数指向存储多张图像文件的文件夹路径；
    * 对需要输入视频的vision language model，该参数指向视频文件；
* 通过`--num-frames`设定每次推理时从视频文件中提取的frame数量，仅video language model需要；
* 通过`--mm-per-prompt`设定每个prompt对应的视觉文件的数量，仅需要多张图像的vision language model需要；

### 性能测试
性能测试，是使用自动生成的、由多个`hi`组成的输入prompt和像素值全0的图像进行推理，过程中忽略停止字符，生成指定长度的输出`token`。推理完成后，统计`TPS(Tokens per second)`、`TTFT(Time to first token)`、`TPOT(Time per output token)`等指标，以验证Enflame gcu的推理性能。

使用上，需要：
* 设定`--perf`以启用性能测试；
* `--input-len`参数设定输入prompt的长度；
* `--max-output-len`参数设定输出token数；
* `--input-vision-shape`给定输入的视觉文件的`shape`信息，其格式应符合模型的要求且输入的视觉文件提取的`image feature token`数量不能大于`--input-len`设定的输入token数；
* 通过`--num-frames`设定每次推理时从视频文件中提取的frame数量，仅video language model需要；
* 通过`--mm-per-prompt`设定每个prompt对应的视觉文件的数量，仅需要多张图像的vision language model需要；

其余相关参数按需设置。

测试完成输出的结果中，`latency_num_prompts`表示本轮推理的总耗时，`latency_per_token`表示每个输出token的latency，`request_per_second`表示以request为单位计算的吞吐，`token_per_second`表示以token为单位计算的吞吐，`prefill_latency_per_token`表示prefill阶段各token的latency，`decode_latency_per_token`表示decode阶段各token的latency，`decode_throughput`表示decode阶段的吞吐。

### 数据集精度测试
数据集精度测试，是使用特定数据集完成指定视觉多模态大模型的推理和结果统计，以验证Enflame gcu的推理精度。

#### 数据集下载
目前，支持`llava-bench-coco`、`MMMU`和`videomme`三个数据集。使用数据集进行精度验证时，需要提前下载数据集。

特别声明：用户可以从huggingface或其他地址自行下载开源数据集，本文仅给出下载链接，不对开源数据集做任何承诺，使用开源数据集产生的一切后果和风险由用户自行承担。

##### llava-bench-coco

数据集下载信息：

  * url:[llava-bench-coco](https://huggingface.co/datasets/lmms-lab/llava-bench-coco/tree/main)；
  * branch:`main`
  * commit:`2533d3eaa6e837c462b1a5486c54b88cde3094fe`

下载完成后将`llava-bench-coco`文件夹拷贝到执行数据集精度验证的设备上。

##### MMMU

数据集下载信息：

  * url:[MMMU](https://huggingface.co/datasets/MMMU/MMMU/tree/main)
  * branch:`main`
  * commit:`171b0ef74cd1704464e6940860968009d8cdd59a`

下载完成后将`MMMU`文件夹拷贝到执行数据集精度验证的设备上。

##### videomme

数据集下载信息：

  * url:[videomme](https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main)；
  * branch:`main`
  * commit:`ead1408f75b618502df9a1d8e0950166bf0a2a0b`

下载完成后将`Video-MME`文件夹拷贝到执行数据集精度验证的设备上。

#### 使用
使用上，需要：
* 设定`--acc`以启用数据集精度测试；
* `--dataset-name`设定使用的数据集名称，目前仅支持`llava-bench-coco`、`MMMU`和`videomme`三种；
* `--dataset-file`设定数据集的本地存储路径，查看下节获取数据集的下载路径；
* `--num-prompts`设定精度测试使用的数据集的prompt数量，默认值`-1`表示使用完整数据集进行测试；
其余参数可按需设置。