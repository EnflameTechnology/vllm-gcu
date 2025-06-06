## LLM数据集精度测试（内部）
### 设计方法
本节介绍如何在数据集上进行vLLM-gcu推理结果的精度验证。

数据集精度验证是指在有代表性的数据集上使用vLLM-gcu执行语言模型（LLM）的推理，并统计推理结果的精度。当前的数据集验证工具支持ceval、mmlu、cmmlu、HumanEval数据集的精度验证。其中，ceval、cmmlu、mmlu为客观单选数据集，分别对应中文和英文数据集。HumanEval数据集为代码生成能力评估数据集。

本项目仅旨在使用数据集来评估超大型语言模型（LLM）模型，测试将利用开源测评框架opencompass-0.3.1提供的部分执行过程、数据预处理和使用方法，用户需自行安装相应环境的opencompass 0.3.1的python包。特别声明，本文仅使用opencompass的部分功能，不对全部功能做任何承诺。

注：opencompass 0.3.1版本需要使用python 3.10或以上版本运行

### 使用流程
#### 数据集下载

特别声明：用户可以从huggingface或其他地址自行下载开源数据集，本文仅给出下载链接，不对开源数据集做任何承诺，使用开源数据集产生的一切后果和风险由用户自行承担。

##### ceval

数据集下载信息：

  * url:[ceval](https://huggingface.co/datasets/ceval/ceval-exam/tree/main)
  * branch:`main`
  * commit:`3923b51`
  * data file:`ceval-exam.zip`

下载完成后将`ceval-exam.zip`拷贝到执行数据集精度验证的设备上并解压。

##### mmlu

数据集下载信息：

  * url:[mmlu](https://huggingface.co/datasets/cais/mmlu/tree/main)
  * branch:`main`
  * commit:`7a00892`
  * data file:`data.tar`

下载完成后将`data.tar`拷贝到执行数据集精度验证的设备上并解压。

##### cmmlu

数据集下载信息：

  * url:[cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu/tree/main)
  * branch:`main`
  * commit:`efcc940`
  * data file:`cmmlu_v1_0_1.zip`

下载完成后将`cmmlu_v1_0_1.zip`拷贝到执行数据集精度验证的设备上并解压。

##### HumanEval数据集

数据集下载信息：

  * url:[humaneval](https://github.com/openai/human-eval/tree/master/data)
  * branch:`master`
  * commit:`312c5e5`
  * data file:`HumanEval.jsonl.gz`

下载完成后将`HumanEval.jsonl.gz`拷贝到执行数据集精度验证的设备上。

#### 用户界面
在安装前请确保默认的二进制python3已经被设置为正确的版本并运行。

安装opencompass可能会改变如torch等版本，如发生环境改变，请自行将依赖包更新到指定版本，以免运行失败。

通过vllm_utils子模块evaluate_datasets的run接口执行测试程序。请在搭载了enflame gcu的服务器上，执行如下命令：

```
python3 -m vllm_utils.evaluate_datasets.run <arguments>
```
参数设置遵循opencompass的使用习惯，推荐的做法是使用配置的简称。目前，已经添加了包括mmlu、cmmlu、ceval、humaneval在内的数据集配置（数据集名称使用name_gen的形式），以及一些模型相关的配置。用户和开发者可以根据需要逐步进行配置。以下是示例用法：

```
python3 -m vllm_utils.evaluate_datasets.run \
--datasets mmlu_gen \
--models vllm_wizardcoder_15b
```

常见参数信息可以通过输入`--help`查看。如果倾向于跳过推理过程，那么必须将`--reuse`参数设置为"latest"，或者指定预测结果的具体路径。

  ```bash
  位置参数：
    config                 训练配置文件路径
  
  可选参数：
    -h, --help             显示此帮助信息并退出
    --models MODELS [MODELS ...]
                           模型列表
    --datasets DATASETS [DATASETS ...]
                           数据集列表
    --summarizer SUMMARIZER
                           摘要生成器
    --debug                调试模式，在该模式下调度器将在
                           单个进程中运行任务，输出不会被重定向到文件
    -m {all,infer,eval,viz}, --mode {all,infer,eval,viz}
                           运行模式。如果你只想获得推理结果，可以选择"infer"；
                           如果你已经有结果并希望评估它们，可以选择"eval"；
                           如果你想可视化结果，可以选择"viz"。
    -r [REUSE], --reuse [REUSE]
                           重用先前的输出和结果
    -w WORK_DIR, --work-dir WORK_DIR
                           工作路径，所有的输出将保存在此路径，包括slurm日志、
                           评估结果、摘要结果等。
                           如果未指定，工作目录将被设置为./outputs/default。
    --config-dir CONFIG_DIR
                           使用自定义配置目录代替config/来搜索数据集、模型和摘要生成器的配置
    --data-dir             数据集路径
    --max-num-workers MAX_NUM_WORKERS
                           并行运行的最大工作线程数。将被配置中的"max_num_workers"参数覆盖。
    --device               运行设备选择
  ```

运行完成后，会在`--output-default`路径下保存本次的运行结果，对于ceval、mmlu、cmmlu数据集可以得到预测结果的精度，对于humaneval数据集可以得到推理生成的文本。

说明：
*  目前ceval、mmlu、cmmlu数据集默认采用zero shot、no chat、no cot模式，即从推理结果中根据第一个logits中根据`A`、`B`、`C`、`D`四个字符的logit值选择最大的作为推理结果，暂未开放n-shot、chat、cot模式；
*  如果模型路径与配置不一致，请设置参数`--vllm-path`为真实路径，具体细节请参考下一项。

模型参数也可以通过参数来灵活配置,下面的程序演示了通过`--vllm-path`配置模型路径。需要注意，目前仅vLLM模型支持以这种方式配置模型参数。

```
python3 -m vllm_utils.evaluate_datasets.run \
--datasets mmlu_gen \
--vllm-path /home/pretrained_models/chatglm3-6b-32k
```

如果希望了解更多的vLLM模型参数信息，可以通过输入`--help`查看。

  ```bash
  vllm模型参数：
    --vllm-path VLLM_PATH
       vLLM模型的路径
    --max-seq-len MAX_SEQ_LEN
       序列的最大长度
    --model-kwargs MODEL_KWARGS [MODEL_KWARGS ...]
       模型的额外参数
    --generation-kwargs GENERATION_KWARGS [GENERATION_KWARGS ...]
       生成过程的额外参数
    --end-str END_STR
       生成文本的结束字符串
    --max-out-len MAX_OUT_LEN
       输出的最大长度
    --batch-size BATCH_SIZE
       批处理大小
    --tensor-parallel-size TENSOR_PARALLEL_SIZE
       张量并行大小
  ```

  
#### 模型测试示例

- 基本运行方式：
```
python3 -m vllm_utils.evaluate_datasets.run \
--datasets humaneval_gen \
--models vllm_wizardcoder_15b
```
这个命令将使用humaneval_gen数据集和vllm_wizardcoder_15b模型进行评估。这里的`--datasets`参数指定了用于评估的数据集名称，而`--models`参数指定了要使用的模型名称。命令中没有指定模型路径，因此它将查找默认路径下的模型。

- 指定模型路径的运行方式：
```
python3 -m vllm_utils.evaluate_datasets.run \
--datasets humaneval_gen \
--models vllm_wizardcoder_15b \
--vllm-path /home/pretrained_models/WizardCoder-15B-v1.0
```
与第一个命令类似，这个命令同样使用humaneval_gen数据集和vllm_wizardcoder_15b模型进行评估。不同之处在于，它通过`--vllm-path`参数指定了模型文件的具体路径。当模型文件不在默认路径时请使用这种方法。

- 给定vllm模型参数的运行方式：
```
python3 -m vllm_utils.evaluate_datasets.run \
--datasets humaneval_gen_6d1cc2 \
--data-dir tinydata/humaneval/human-eval-v2-20210705.jsonl \
--max-out-len 1024 \
--max-seq-len 2048 \
--batch-size 8 \
--model-kwargs dtype=half \
--vllm-path /home/pretrained_models/WizardCoder-15B-v1.0
```
该命令提供了`--model-kwargs`，给定了`dtype`和`quantization`等vllm模型需要的参数。
