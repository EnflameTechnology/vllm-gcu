add_py_test (PROJECT scorpio
             PLATFORM s60
             REGRESSION daily
             CATEGORY func
             OS ubuntu
             MODULE vllm_dev_kernels
             ID 1
             NAME test_mrotary_embedding
             COMMAND "cd tests && python -m pytest -sv kernels/test_mrotary_embedding.py"
            )
