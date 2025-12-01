## qwen

### Qwen2.5-Coder-32B-Instruct
本模型推理及性能测试至少需要8张enflame gcu。


#### 批量离线推理
* 测试基于mqa score spec decode，需要按照vllm官网要求，在eager mode下使用flashattn backend进行测试。
* 安装Enflame提供的flashattn gcu版whl包。
```
pip3 install flash_attn-2.6.3+torch.2.5.1.gcu.***-linux_x86_64.whl
```
* 添加环境变量 export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```shell
python3 -m vllm_utils.benchmark_test \
--model [path of qwen2.5-coder-32b-instruct] \
--tensor-parallel-size 8 \
--demo cm \
--template default \
--add-generation-prompt True \
--output-len 9016 \
--dtype float16 \
--speculative-model '[ngram]' \
--num-speculative-tokens 100 \
--ngram-prompt-lookup-max 50 \
--ngram-prompt-lookup-min 10 \
--use-v2-block-manager \
--trust-remote-code \
--enforce-eager \
--max-model-len 20000
```

#### serving模式
* 测试基于mqa score spec decode，需要按照vllm官网要求，在eager mode下使用flashattn backend进行测试。
* 安装Enflame提供的flashattn gcu版whl包。
```
pip3 install flash_attn-2.6.3+torch.2.5.1.gcu.***-linux_x86_64.whl
```
* 添加环境变量 export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
--model [path of qwen2.5-coder-32b-instruct] \
--tensor-parallel-size 8  \
--max-model-len 20000  \
--max-num-seqs 1 \
--speculative-model '[ngram]' \
--num-speculative-tokens 100 \
--ngram-prompt-lookup-max 50 \
--ngram-prompt-lookup-min 10 \
--use-v2-block-manager \
--enforce_eager \


# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend vllm \
--dataset-name="sharegpt" \
--dataset-path=[path of ShareGPT_V3_unfiltered_cleaned_split.json] \
--model=[path of qwen2.5-coder-32b-instruct]\
--trust_remote_code \
--save-result \
--num-prompts 128 \
--sharegpt-output-len 1024
```
注：
* 为测试spec decode效果，数据集使用ShareGPT测试；
* num-prompts、sharegpt-output-len可按需调整；