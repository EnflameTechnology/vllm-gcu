add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 1
             NAME test_routing_strategy
             COMMAND "cd tests && python -m pytest -sv kernels/test_routing_strategy.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 2
             NAME test_apply_repetition_penalties
             COMMAND "cd tests && python -m pytest -sv kernels/test_apply_repetition_penalties.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 3
             NAME test_topk_softmax_renormalize
             COMMAND "cd tests && python -m pytest -sv kernels/test_topk_softmax_renormalize.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 4
             NAME test_chunk_delta_h
             COMMAND "cd tests && python -m pytest -sv kernels/test_chunk_delta_h.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 5
             NAME test_top_k_per_row
             COMMAND "cd tests && python -m pytest -sv kernels/test_top_k_per_row.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 6
             NAME test_convert_req_index_to_global_index
             COMMAND "cd tests && python -m pytest -sv kernels/test_convert_req_index_to_global_index.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 7
             NAME test_mrotary_embedding
             COMMAND "cd tests && python -m pytest -sv kernels/test_mrotary_embedding.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 8
             NAME test_topk_softmax_renormalize
             COMMAND "cd tests && python -m pytest -sv kernels/test_topk_softmax_renormalize.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 9
             NAME test_index_add
             COMMAND "cd tests && python -m pytest -sv kernels/test_index_add.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 10 
             NAME test_get_ep_indices
             COMMAND "cd tests && python -m pytest -sv kernels/test_get_ep_indices.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 11
             NAME test_cp_gather_indexer_k_quant_cache
             COMMAND "cd tests && python -m pytest -sv kernels/test_cp_gather_indexer_k_quant_cache.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 12
             NAME test_flashmla_mixed
             COMMAND "cd tests && python -m pytest -sv kernels/test_flashmla_mixed.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 13
             NAME test_linear_quant_per_tensor_fp8
             COMMAND "cd tests && python -m pytest -sv kernels/test_linear_quant_per_tensor_fp8.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 14
             NAME test_correct_attn_cp_out
             COMMAND "cd tests && python -m pytest -sv kernels/test_correct_attn_cp_out.py"
            )
add_py_test (PROJECT libra
             PLATFORM silicon
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev
             ID 15
             NAME test_fused_moe_per_tensor_fp8
             COMMAND "cd tests && python -m pytest -sv kernels/test_fused_moe_per_tensor_fp8.py"
            )