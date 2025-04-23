add_py_test (PROJECT scorpio
             PLATFORM s60
             REGRESSION ci daily
             CATEGORY func
             OS ubuntu
             MODULE vllm
             ID 1
             NAME test_gcu_models
             COMMAND "cd tests && \
             export TORCH_DEVICE_BACKEND_AUTOLOAD=0 && python -m pytest test_offline_inference.py")