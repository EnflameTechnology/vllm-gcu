add_py_test (PROJECT scorpio
             PLATFORM s60
             REGRESSION ci daily
             CATEGORY func
             OS ubuntu
             MODULE vllm
             ID 1
             NAME test_template
             COMMAND "echo test success")