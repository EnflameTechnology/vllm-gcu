## Pooling模型测试

`benchmark_embedding_rerank`是Enflame新增的测试入口，使用方式为`python3 -m vllm_utils.benchmark_embedding_rerank arg1 arg2 ...`。可参考第7章和第8章查看Pooling模型的推理指令。

功能上，支持：
* serving mode下，使用指定长度的伪prompt测试Pooling模型的性能；
* offline mode下，使用指定的数据集进行Pooling模型的推理精度测试；

可以通过`python3 -m vllm_utils.benchmark_embedding_rerank --help`查看各参数含义。


### 性能测试
性能测试，是使用随机生成的prompt进行推理。推理完成后，统计`average_response_time(s)`、`avg_latency_per_sample(s)`、`p99(s)`等指标，以验证Enflame gcu的推理性能。

参数说明：
* `--max-concurrency`用于控制最大并发数，默认是`None`，不限制最大并发数；
* `--total-requests`用于控制总的请求数据，在不设置`--max-concurrency`时，会一次将所有请求发送给server端；
* `--input-len`用于设置输入sequence的长度；
* `--query-len`用于reranker模型，query的输入长度；
* `--num-docs`用于reranker模型，query需要对比的docs数量，每个doc长度为`--query-len`；

其余相关参数按需设置。

测试完成输出的结果中，`average_response_time(s)`表示平均每个请求的耗时，`avg_latency_per_sample(s)`表示每个请求的latency，`p99(s)`表示99%以上请求耗时。

### 数据集精度测试
数据集精度测试，是使用特定数据集完成指定Pooling模型的推理和结果统计，以验证Enflame gcu的推理精度。

#### 数据集下载
目前，支持`OnlineShopping-classification`、`fiqa`两个数据集。使用数据集进行精度验证时，需要提前下载数据集。

特别声明：用户可以从huggingface或其他地址自行下载开源数据集，本文仅给出下载链接，不对开源数据集做任何承诺，使用开源数据集产生的一切后果和风险由用户自行承担。

##### llava-bench-coco

数据集下载信息：

  * url: [OnlineShopping-classification](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification/tree/main)；
  * branch: `main`
  * commit: `e610f2ebd179a8fda30ae534c3878750a96db120`

下载完成后将`OnlineShopping-classification`文件夹拷贝到执行数据集精度验证的设备上。

##### fiqa

数据集下载信息：

  * url: [fiqa](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip)

下载完成后将`fiqa`文件夹拷贝到执行数据集精度验证的设备上。


#### 使用
使用上，需要：
* `--dataset`用于reranker模型，当前仅支持`fiqa`；
* `--beir-data-root`用于reranker模型，设置为`fiqa`数据集路径；
* `--eval-task`用于embedding模型，当前仅支持`OnlineShopping`；
* `--dataset-dir`用于embedding模型，设置为`OnlineShopping`数据集路径；
其余参数可按需设置。
