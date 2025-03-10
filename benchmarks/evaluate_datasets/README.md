# Tiny Dataset Evaluation

## Design method

- Project is positioned to evaluate vLLM models with datasets only, never used as general interface of opencompass. Who tends to use full opencompass functionality, please refer to its official project
- Utilizing the execution process, data pre-processing and usage methods provided by [opencompass](https://pypi.org/project/opencompass/), the goal is to provide standard testing methods and credible test results. In order to ensure the accuracy of calculation and adaptability of usage, partial configuration files have been copied from opencompass
- Expand *VLLM* model in opencompass as *VLLMEX*
  - add an efficient method of calculating ppl
  - add supplementary methods of pre-processing prompt text and post-processing generated text to adapt to more models, together with meta_template
- In addition to the solidified configuration files provided by opencompass, add method of configuring model parameters through command line parameters to enhance flexibility

## Requirements

```shell
python3 -m pip install sentence-transformers==2.2.2
python3 -m pip install huggingface-hub==0.25.2
```

## User interface

- Necessary to install requirements recording in the same directory. Worth noting that some dependencies may change local environment, better to install with *--no-deps*
  
- **Caution: Make sure default binary *python3* has been set to expected and executed one**
  
- The execution entry is **run**, simplifiing some parameters of opencompass. For out-of-tree executing, it has been encapsulated as a module of vllm_utils, command as
  
  ```python
  python -m vllm_utils.evaluate_datasets.run <arguments>
  ```
  

- Arguments respect usage habits of opencompass which supports short-name of configurations if exist, which is recommended. Currently, configuration for datasets including *mmlu*, *cmmlu*, *ceval*, *humaneval* and few model configs have been added. Users and developers can incrementally configure as need. Example usage as below,
  
  ```bash
  python -m vllm_utils.evaluate_datasets.run --datasets mmlu_gen --models vllm_wizardcoder_15b
  ```
  
  Common arguments messages could be viewed by *--help. When tend to skip *infer*, *--reuse* must be set as *"latest"* or specific path to predictions.
  
  ```bash
  positional arguments:
    config                Train config file path
  
  optional arguments:
    -h, --help            show this help message and exit
    --models MODELS [MODELS ...]
    --datasets DATASETS [DATASETS ...]
    --summarizer SUMMARIZER
    --debug               Debug mode, in which scheduler will run tasks in the single process, and output will not be redirected to files
    -m {all,infer,eval,viz}, --mode {all,infer,eval,viz}
                          Running mode. You can choose "infer" if you only want the inference results, or "eval" if you already have the
                          results and want to evaluate them, or "viz" if you want to visualize the results.
    -r [REUSE], --reuse [REUSE]
                          Reuse previous outputs & results
    -w WORK_DIR, --work-dir WORK_DIR
                          Work path, all the outputs will be saved in this path, including the slurm logs, the evaluation results, the summary
                          results, etc.If not specified, the work_dir will be set to ./outputs/default.
    --config-dir CONFIG_DIR
                          Use the custom config directory instead of config/ to search the configs for datasets, models and summarizers
    --max-num-workers MAX_NUM_WORKERS
                          Max number of workers to run in parallel. Will be overrideen by the "max_num_workers" argument in the config.
    --device DEVICE
  ```

  Note: if *model path* is not inconsistent with configuration, please set argument *--vllm-path* to path to real, details refer to next item.
  
- Model parameters can also be configured from arguments considering flexibility. Note that only vLLM models are supported in this way.
  
  ```bash
  python -m vllm_utils.evaluate_datasets.run --datasets mmlu_gen --vllm-path /home/pretrained_models/chatglm3-6b-32k
  ```
  
  vllm model arguments messages could be viewed by *--help*
  
  ```bash
  vllm_model_args:
    --vllm-path VLLM_PATH
    --max-seq-len MAX_SEQ_LEN
    --model-kwargs MODEL_KWARGS [MODEL_KWARGS ...]
    --generation-kwargs GENERATION_KWARGS [GENERATION_KWARGS ...]
    --end-str END_STR
    --max-out-len MAX_OUT_LEN
    --batch-size BATCH_SIZE
    --tensor-parallel-size TENSOR_PARALLEL_SIZE

## Example Model tests

```shell
python -m vllm_utils.evaluate_datasets.run --datasets humaneval_gen --models vllm_wizardcoder_15b
python -m vllm_utils.evaluate_datasets.run --datasets humaneval_gen --models vllm_wizardcoder_15b --vllm-path /home/pretrained_models/WizardCoder-15B-v1.0
python -m vllm_utils.evaluate_datasets.run --datasets mmlu_ppl --vllm-path /home/pretrained_models/chatglm3-6b-32k --batch-size 4
```
