## 新增模型
vLLM-gcu中，可以按照[vLLM官方手册](https://docs.vllm.ai/en/v0.11.0/contributing/model/registration.html#out-of-tree-models)所示进行模型新增。相比于对vLLM-gcu进行侵入式修改，推荐采用`out-of-tree`方式进行模型新增。

使用`out-of-tree`方式进行模型新增推理，为了方便新增模型可以注册到所有的`worker`，可按照下述方法实现示例模型`yourmodel`的新增。
### 模型实现
在`vllm_gcu/models`目录下创建`yourmodel`目录，目录结构如下所示：
```
vllm_gcu
   └── models
        └── yourmodel
              ├── __init__.py
              ├── configuration_yourmodel.py (非必须)          
              └── yourmodel.py
```
* 说明：
  * __init__.py：空文件，用于打包时可以正常包含`yourmodel`文件夹。
  * configuration_yourmodel.py：添加模型对应的config文件，定义为`YourModelConfig`。
  * yourmodel.py：模型实现代码，模型结构定义为`YourModelForCausalLM`。

### 模型注册

在`vllm_gcu/models/__init__.py`文件内的`register_custom_models`函数添加以下内容。

```python
# 注册config (非必须)
custom_configs = [
    ("yourmodel", "vllm_gcu.models.yourmodel.configuration_yourmodel", "YourModelConfig"),
]
# 注册模型
ModelRegistry.register_model("YourModelForCausalLM", "vllm_gcu.models.yourmodel.yourmodel:YourModelForCausalLM")
```
说明：
* 如果使用的transformers库不包含该模型，需要注册config。
* 在vllm没有实现且transformers库内包含该模型时，vllm会fallback到transformers上推理。
* 在vllm_gcu内注册模型失败且transformers库内包含该模型时，vllm会fallback到transformers上推理。
* 在vllm官方实现中包含该模型时，vllm_gcu内注册的模型会覆盖vllm官方实现的模型。
