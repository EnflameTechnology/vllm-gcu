## CodeGeeX

codegeex系列模型，使用vllm 0.6.1.post2及以上版本时，需要手动降级transformers库版本

```
python3 -m pip install transformers==4.43.0
```

### CodeGeeX2-6B

#### 模型下载
* url: [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b/tree/main)
* branch: `main`
* commit id: `3cb3f8fa305c8188c6c997d0be2ccc4b87ba6f7f`

将上述url设定的路径下的内容全部下载到`codegeex2-6b`文件夹中。

#### Tokenizer修改

将**codegeex2-6b/tokenization_chatglm.py**中ChatGLMTokenizer类的__init__函数修改为如下内容：

```python
    def __init__(self,
                 vocab_file,
                 padding_side="left",
                 clean_up_tokenization_spaces=False,
                 **kwargs):
        self.tokenizer = SPTokenizer(vocab_file)
        super().__init__(padding_side=padding_side,
                         clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                         **kwargs)
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        # self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codegeex2-6b] \
 --demo=cc \
 --output-len=100 \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codegeex2-6b] \
 --max-model-len=8192 \
 --tokenizer=[path of codegeex2-6b] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
* 本模型支持的`max-model-len`为8k；
* `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codegeex4-all-9b

#### 模型下载
*  url: [codegeex4-all-9b](https://huggingface.co/THUDM/codegeex4-all-9b/tree/main)

*  branch: `main`

*  commit id: `6ee90cf`

将上述url设定的路径下的内容全部下载到`codegeex4-all-9b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codegeex4-all-9b] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codegeex4-all-9b] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --input-len=2048 \
 --output-len=2048 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

注：
*  本模型支持的`max-model-len`为131072；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

