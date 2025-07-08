## multilora
### 使用方法
```shell
python3 -m vllm_utils.multilora_inference --base-model=[path of base model] --device=[device type] --lora-config=[path of lora config file]
```
各参数含义如下：
* `--base-model`：base model存储路径；
* `--device`：设备类型，默认为`gcu`；
* `--lora-config`:lora配置文件存储路径，配置文件类型需为`json`。格式如下：
  ```
  {
    "lora_models":[
        {"id":[lora model 1 id],"model_path":[lora model 1 path]}，
        {"id":[lora model 2 id],"model_path":[lora model 2 path]}
    ],
    "prompts":[
        {"text":[prompt text 1],"lora_id":[id of lora model]}，
        {"text":[prompt text 2],"lora_id":[id of lora model]}
  }
  ```
  * `lora_models`:设定各lora model的`id`及存储路径；
  * `prompts`:设定各输入prompt的文本信息和对应的lora model id，若不设置`lora id`，则仅使用`--base-model`进行推理；

### 示例
```shell
python3 -m vllm_utils.multilora_inference --base-model=chatglm3-6b/ --device=gcu --lora-config=chatglm_6b_lora.config
```
* `chatglm3-6b/`为模型的本地存储路径，下载自[chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b/commit/e46a14881eae613281abbd266ee918e93a56018f);
* `chatglm_6b_lora.config`为lora配置文件，其内容为：
  ```
  {
    "lora_models":[
        {"id":1,"model_path":"chatglm3-6b-csc-chinese-lora"}
    ],
    "prompts":[
        {"text":"请介绍下你自己,包括你的主要功能和应用场景。"},
        {"text":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: 对下面文本纠错\n\n少先队员因该为老人让坐。 ASSISTANT:",
            "lora_id":"1"},
        {"text":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: 对下面文本纠错\n\n下个星期，我跟我朋唷打算去法国玩儿。 ASSISTANT:",
            "lora_id":"1"}
    ]
  }
  ```
  * `chatglm3-6b-csc-chinese-lora`为lora模型的本地存储路径，下载自[chatglm3-6b-csc-chinese-lora](https://huggingface.co/shibing624/chatglm3-6b-csc-chinese-lora/tree/main);
  * `chatglm3-6b-csc-chinese-lora`在本轮推理时设置其`id`为1；
  *  `prompts`部分设置的三个prompt，第一个仅使用`base model`进行推理，其余使用`base model`和`lora model`共同进行推理；