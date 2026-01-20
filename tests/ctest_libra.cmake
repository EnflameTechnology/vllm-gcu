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
 