# add_py_test (PROJECT libra
#              PLATFORM silicon
#              REGRESSION ci daily
#              CATEGORY func
#              OS ubuntu
#              MODULE vllm
#              ID 1
#              NAME test_gcu_models
#              COMMAND "cd tests && \
#              export PYTORCH_EFML_BASED_GCU_CHECK=1 && \
#              export TORCHGCU_INDUCTOR_ENABLE=0 && \
#              export TORCH_ECCL_AVOID_RECORD_STREAMS=1 && \
#              export TORCH_DEVICE_BACKEND_AUTOLOAD=0 && python -m pytest test_offline_inference.py")

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
 