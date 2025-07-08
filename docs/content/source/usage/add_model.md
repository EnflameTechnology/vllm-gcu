## 新增模型
vLLM-gcu中，可以按照[vLLM官方手册](https://docs.vllm.ai/en/stable/models/adding_model.html)所示进行模型新增。相比于对vLLM-gcu进行侵入式修改，推荐采用`out-of-tree`方式进行模型新增。

使用`out-of-tree`方式进行模型新增且需要多卡推理时，为了方便新增模型可以注册到所有的`worker`，可按照下述方法实现示例模型`cusllama`的新增。
* 本地新建文件夹A，在其下创建`OOT_utils.py`，内部填写下述内容：
  ```
    import os
    import sys
    import importlib

    def register_OOT_models(OOT_models_path: str):
        for model in os.listdir(OOT_models_path):
            model_path = os.path.join(OOT_models_path, model)
            if not os.path.isdir(model_path) or \
                not os.path.exists(os.path.join(model_path,'register_'+model+'.py')):
                continue

            sys.path.append(model_path)
            module = importlib.import_module('register_'+model)
            attrs = dir(module)
            for attr in attrs:
                if 'register' in attr:
                    func = getattr(module, attr)
                    func()
  ```
* 在A下创建模型名称文件夹，需与模型名称一致，本例中为`cusllama`；
* 在`cusllama`下新建存储模型推理过程的py文件，并在新建的`register_cusllama.py`中实现`register_cusllama`函数，在其中调用`ModelRegistry.register_model`完成模型注册。如下给出`register_cusllama.py`的示例：
  ```
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(current_dir)
    import contextlib
    from transformers import AutoConfig

    def register_cusllama():
        from cusllama import CustomerLlaMAForCausalLM
        from cusllama_config import CustomerLlaMAConfig

        from vllm import ModelRegistry
        ModelRegistry.register_model('CustomerLlaMAForCausalLM', CustomerLlaMAForCausalLM)

        with contextlib.suppress(ValueError):
            AutoConfig.register("cusllama", CustomerLlaMAConfig)
  ```
* 推理时，添加环境变量`VLLM_OOT_MODEL_PATH`指向A的路径。