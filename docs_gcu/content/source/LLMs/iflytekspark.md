## iFlytekSpark
### iflytekspark_13b
本模型推理及性能测试需要两张enflame gcu。
#### 模型下载
*  url: [iflytekspark_megatron_ckpt](https://gitee.com/iflytekopensource/i-flytek-spark-13-b-model-gpu/tree/master)
*  branch: `main`
*  commit id: `55071d6`

将上述url设定的路径下的内容全部下载到`path_to_iflytekspark_megatron_ckpt`文件夹中。
#### megatron到hg模型权重转换
1. 将权重文件转换成hg格式，并保存至path_to_iflytekspark_hg_ckpt
    ```shell
    mkdir path_to_iflytekspark_hg_ckpt
    python3 convert_to_hf.py \
    --mp_states_path=path_to_iflytekspark_megatron_ckpt/\
                     iFlytekSpark_13B_base_fp32/mp_rank_00_model_states.pt \
    --out_path=path_to_iflytekspark_hg_ckpt
    ```
    convert_to_hf.py脚本如下：
    ```python
        import os
        import torch
        from collections import OrderedDict
        import re
        import argparse
        from typing import Dict
        import json
        from safetensors.torch import save_file as safe_save_file
        from huggingface_hub import split_torch_state_dict_into_shards
        from vllm.transformers_utils.config import IFlytekSparkConfig
        def fix_query_key_value_ordering(param,
                                         checkpoint_version,
                                         num_splits,
                                         num_heads,
                                         hidden_size):
            # Permutes layout of param tensor to 
            # [num_splits * num_heads * hidden_size, :]
            # for compatibility with later versions of NVIDIA Megatron-LM.
            # The inverse operation is performed inside Megatron-LM to 
            # read checkpoints:
            # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/
            # megatron/checkpointing.py#L209
            # If param is the weight tensor of the self-attention block, 
            # the returned tensor
            # will have to be transposed one more time to be read by 
            # HuggingFace GPT2.
            input_shape = param.size()
            if checkpoint_version == 1.0:
                # version 1.0 stores [num_heads * hidden_size * num_splits, :]
                saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
                param = param.view(*saved_shape)
                param = param.transpose(0, 2)
                param = param.transpose(1, 2).contiguous()
            elif checkpoint_version >= 2.0:
                # other versions store [num_heads * num_splits * hidden_size, :]
                saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
                param = param.view(*saved_shape)
                param = param.transpose(0, 1).contiguous()
            param = param.view(*input_shape)
            return param
        def convert_megatron_checkpoint(sd_megatron, config):
            """
            Converts a Megatron checkpoint to a HuggingFace GPT-SW3 checkpoint.
            """
            hidden_size = config.hidden_size
            heads = config.num_attention_heads
            intermediate_size = config.intermediate_size
            hidden_size_per_head = config.hidden_size // config.num_attention_heads
            print(f"sd_megatron keys:{sd_megatron.keys()}")
            # Megatron-LM checkpoint version
            if "checkpoint_version" in sd_megatron.keys():
                checkpoint_version = sd_megatron["checkpoint_version"]
            else:
                checkpoint_version = 5.0
            checkpoint_version = 0
            # The model.
            model = sd_megatron["module"]
            # The language model.
            lm = model["language_model"]
            # The embeddings.
            embeddings = lm["embedding"]
            # The word embeddings.
            word_embeddings = embeddings["word_embeddings"]["weight"]
            # Truncate the embedding table to vocab_size rows.
            word_embeddings = word_embeddings[: config.vocab_size, :]
            # The position embeddings.
            # pos_embeddings = embeddings["position_embeddings"]["weight"]
            # The transformer.
            transformer = lm["transformer"] if "transformer" \
                in lm.keys() else lm["encoder"]
            sd_hf = {
                "model.embed_tokens.weight": word_embeddings,
            }
            # The regex to extract layer names.
            layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
            # Keep track of the attention/query/value tensor.
            attention_qkv_weight = None
            # Extract the layers.
            for key, val in transformer.items():
                # Match the name.
                m = layer_re.match(key)
                # Stop if that's not a layer
                if m is None:
                    break
                # The index of the layer.
                layer_idx = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)
                # The name of the layer.
                layer_name = f"model.layers.{layer_idx}"
                # For layernorm(s), simply store the layer norm.
                if op_name.endswith("layernorm"):
                    ln_name = "input_layernorm" if op_name.startswith("input") \
                        else "post_attention_layernorm"
                    sd_hf[layer_name + "." + ln_name + "." + weight_or_bias] = val
                    print(f"process {key} => {ln_name}.{weight_or_bias}")
                # Transpose the QKV matrix.
                elif (
                    op_name == "attention.query_key_value" or \
                        op_name == "self_attention.query_key_value"
                ) and weight_or_bias == "weight":
                    print(f"process {key} => query_key_value.weight")
                    # Make sure the QKV pointer is nil.
                    assert attention_qkv_weight is None, ""
                    out_val = fix_query_key_value_ordering(val,
                                                           checkpoint_version,
                                                           3,
                                                           heads,
                                                           hidden_size_per_head)
                    # Store the tensor as we need the bias as well to interleave QKV and biases.
                    attention_qkv_weight = out_val
                # Transpose the bias.
                elif (
                    op_name == "attention.query_key_value" or \
                        op_name == "self_attention.query_key_value"
                ) and weight_or_bias == "bias":
                    print(f"process {key} => query_key_value.bias")
                    # Make sure we read the weight tensor.
                    assert attention_qkv_weight is not None, ""
                    # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
                    q = attention_qkv_weight[0 * config.hidden_size: \
                        1 * config.hidden_size, :]
                    k = attention_qkv_weight[1 * config.hidden_size: \
                        2 * config.hidden_size, :]
                    v = attention_qkv_weight[2 * config.hidden_size: \
                        3 * config.hidden_size, :]
                    out_val = fix_query_key_value_ordering(val,
                                                          checkpoint_version,
                                                          3,
                                                          heads,
                                                          hidden_size_per_head)
                    # Split the bias.
                    q_bias = out_val[0 * config.hidden_size: \
                        1 * config.hidden_size]
                    k_bias = out_val[1 * config.hidden_size: \
                        2 * config.hidden_size]
                    v_bias = out_val[2 * config.hidden_size: \
                        3 * config.hidden_size]
                    # Store.
                    sd_hf[f"{layer_name}.self_attn.q_proj.weight"] = q
                    sd_hf[f"{layer_name}.self_attn.q_proj.bias"] = q_bias
                    sd_hf[f"{layer_name}.self_attn.k_proj.weight"] = k
                    sd_hf[f"{layer_name}.self_attn.k_proj.bias"] = k_bias
                    sd_hf[f"{layer_name}.self_attn.v_proj.weight"] = v
                    sd_hf[f"{layer_name}.self_attn.v_proj.bias"] = v_bias
                    # Clear the stored tensor.
                    attention_qkv_weight = None
                # Transpose the weights.
                elif op_name == "self_attention.dense" and \
                    weight_or_bias == "weight":
                    print(f"process {key} => dense.weight")
                    # out_name = megatron_to_transformers[op_name]
                    sd_hf[layer_name + ".self_attn.dense." + "weight"] = val
                # Copy the bias.
                elif op_name == "self_attention.dense" and weight_or_bias == "bias":
                    name = layer_name + ".self_attn.dense." + "bias"
                    print(f"process {key} => {name} size:{val.size()}")
                    # out_name = megatron_to_transformers[op_name]
                    # print(val)
                    sd_hf[name] = val
                #dense_4h_to_h
                elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                    print(f"process {key} => mlp.dense_h_to_4h.weight")
                    # out_name = megatron_to_transformers[op_name]
                    tmp_name = layer_name + ".mlp.dense_h_to_4h." + "weight"
                    sd_hf[tmp_name] = val.contiguous()
                # Copy the bias.
                elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "bias":
                    print(f"process {key} => mlp.dense_h_to_4h.bias")
                    # out_name = megatron_to_transformers[op_name]
                    tmp_name = layer_name + ".mlp.dense_h_to_4h." + "bias"
                    sd_hf[tmp_name] = val.contiguous()
                # dense_4h_to_h
                elif op_name == "mlp.dense_4h_to_h" and weight_or_bias == "weight":
                    print(f"process {key} => mlp.dense_4h_to_h.weight")
                    # out_name = megatron_to_transformers[op_name]
                    tmp_name = layer_name + ".mlp.dense_4h_to_h." + "weight"
                    sd_hf[tmp_name] = val.contiguous()
                # Copy the bias.
                elif op_name == "mlp.dense_4h_to_h" and weight_or_bias == "bias":
                    print(f"process {key} => mlp.dense_4h_to_h.bias")
                    # out_name = megatron_to_transformers[op_name]
                    tmp_name = layer_name + ".mlp.dense_4h_to_h." + "bias"
                    sd_hf[tmp_name] = val.contiguous()
            # The final layernorm.
            sd_hf["model.norm.weight"] = transformer["final_layernorm.weight"]
            sd_hf["model.norm.bias"] = transformer["final_layernorm.bias"]
            # For LM head, transformers' wants the matrix to weight embeddings.
            sd_hf["lm_head.weight"] = word_embeddings.clone()
            return sd_hf
        def save_state_dict(state_dict: Dict[str, torch.Tensor], save_directory: str):
            state_dict_split = split_torch_state_dict_into_shards(state_dict)
            for filename, tensors in state_dict_split.filename_to_tensors.items():
                shard = {tensor: state_dict[tensor] for tensor in tensors}
                safe_save_file(
                    shard,
                    os.path.join(save_directory, filename),
                    metadata={"format": "pt"},
                )
            if state_dict_split.is_sharded:
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }
                with open(os.path.join(save_directory,
                                      "model.safetensors.index.json"),
                                      "w") as f:
                    f.write(json.dumps(index, indent=2))
        def save_checkpoint(pt_path, hf_path):
            ckpt = torch.load(pt_path)
            config = IFlytekSparkConfig()
            output_state_dict = convert_megatron_checkpoint(ckpt, config)
            save_state_dict(output_state_dict, hf_path)
            print(f'done')
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(
                description="convert mp model to hf safetensors"
            )
            parser.add_argument(
                "-p",
                "--mp_states_path",
                type=str,
                help="the mp_rank_00_model_states.pt path"
            )
            parser.add_argument(
                "-o",
                "--out_path",
                type=str,
                help="output path"
            )
            code_path = os.path.abspath(os.path.dirname(__file__))
            args = parser.parse_args()
            save_checkpoint(args.mp_states_path, args.out_path)
    ```
2. 新建如下config.json，并保存至path_to_iflytekspark_hg_ckpt
    ```json
        {
        "_name_or_path": null,
        "architectures": [
            "IFlytekSparkForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 5,
        "hidden_act": "gelu_gate",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "iflytekspark",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "pretraining_tp": 1,
        "layer_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "rope_scaling": null,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
        "transformers_version": "4.40.0.dev0",
        "use_cache": true,
        "attention_bias": true,
        "vocab_size": 60000,
        "gated_linear_unit":true
    }
    ```
3. 拷贝tokenizer.model tokenizer.vocab文件至path_to_iflytekspark_hg_ckpt
    ```shell
    cp path_to_iflytekspark_megatron_ckpt/Tokenizer/tokenizer.* \
    path_to_iflytekspark_hg_ckpt
    ```
    iflytekspark_hg_ckpt下的目录结构如下所示
    ```shell
        ├── config.json
        ├── model-00001-of-00010.safetensors
        ├── model-00002-of-00010.safetensors
        ├── model-00003-of-00010.safetensors
        ├── model-00004-of-00010.safetensors
        ├── model-00005-of-00010.safetensors
        ├── model-00006-of-00010.safetensors
        ├── model-00007-of-00010.safetensors
        ├── model-00008-of-00010.safetensors
        ├── model-00009-of-00010.safetensors
        ├── model-00010-of-00010.safetensors
        ├── model.safetensors.index.json
        ├── tokenizer.model
        └── tokenizer.vocab
    ```
#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of iflytekspark_hg_ckpt] \
 --demo=tc \
 --tensor-parallel-size=2 \
 --output-len=32 \
 --trust-remote-code \
 --dtype=float16
```
注：
*  本模型在ecc off模式下双卡支持的`max-model-len`为32k，ecc on模式下双卡支持的`max-model-len`为32k，使能32k需要2卡；
#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
--model=[path of iflytekspark_hg_ckpt] \
--max-model-len=32768 \
--tokenizer=[path of iflytekspark_hg_ckpt] \
--input-len=128 \
--output-len=128 \
--num-prompts=8 \
--block-size=64 \
--dtype=float16 \
--gpu-memory-utilization=0.945 \
--tensor-parallel-size=2
```
注：
*  本模型在ecc off模式下双卡支持的`max-model-len`为32k，ecc on模式下双卡支持的`max-model-len`为32k，使能32k需要2卡；
*  `input-len`、`output-len`和`batch-size`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
