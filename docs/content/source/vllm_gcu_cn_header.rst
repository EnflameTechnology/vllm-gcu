前言
====

本文档介绍vLLM-gcu的功能和使用方法，包括模型批量离线推理示例、性能评估及在特定数据集上的精度验证。


版本信息
--------

.. table:: 版本信息
   :align: center
   :widths: 15 20 15 10 40
   :width: 100%

   ===========  ============= =========== ============= =======================================================
   日期         版本          作者        原始vllm版本   新增功能
   ===========  ============= =========== ============= =======================================================
   2023-11-09   v0.2.1        Enflame     0.2.1         **1.在gcu上支持了下述模型的推理：**
                                                         - llama2 7b/13b;
   2024-01-25   v0.2.6        Enflame     0.2.6         **1.在gcu上支持了下述模型的推理：**
                                                         - llama2 70b;
                                                         - llama 65b;
                                                         - chatglm2/3 6b;
                                                         - baichuan2 7b;
                                                         - Mistral-7b-v0.1;
                                                         - Qwen-7b、Qwen-14b-Chat、Qwen-72b-Chat;
                                                         - vicuna-13b-v1.5;
                                                        **2.添加了性能测试；**

                                                        **3.添加了数据集精度验证，支持ceval/mmlu/adgen数据集；**
   2024-01-30   v0.2.7        Enflame     0.2.7         **1.在gcu上支持了下述模型的推理：**
                                                         - yi 6b/34b;
                                                         - internlm 7b/20b;
                                                        **2.添加了cmmlu数据集的精度验证；**
   2024-03-07   v0.2.7        Enflame     0.2.7         **1.在gcu上支持了下述模型的推理：**
                                                         - baichuan2 7b/13b;
                                                         - codegeex 6b;
                                                         - codellama 13b/34b;
                                                         - gpt-neox-20b;
                                                         - WizardCoder 15b/33b;
                                                         - Ziya-Coding-34B-v1.0
                                                        **2.添加了humaneval数据集的精度验证；**
    2024-03-14  v0.2.7        Enflame     0.2.7         **1.在gcu上支持了下述模型的推理：**
                                                         - starcoderbase;
                                                         - OPT 13b;
                                                         - deepseek-llm-67b;
                                                         - xuanyuan-70b;
                                                         - mixtral-8x7b;
   2024-03-29   v0.2.7        Enflame     0.3.3         **1.升级至0.3.3**
   2024-04-07   v0.3.3        Enflame     0.3.3         **1.在gcu上支持了下述模型的推理：**
                                                         - bloomz-7b1;
                                                         - bloom-7b1;
                                                         - CharacterGLM-6B;
                                                         - Aquila2-34B;
                                                         - AquilaChat2-34B;
   2024-04-16   v0.2.7        Enflame     0.3.3         **1.在gcu上支持了下述模型的推理：**
                                                         - codellama-70b;
                                                         - vicuna-33b-v1.3;
                                                         - Qwen1.5-7B;
                                                         - Qwen1.5-14B-Chat;
                                                         - deepseek-llm-67b-chat;
   2024-04-28   v0.3.3        Enflame     0.3.3         **1.在gcu上支持了下述模型的推理：**
                                                         - starcoder2-7b;
                                                         - starcoder2-15b;
                                                         - Orion-14b-base;
                                                         - gpt-j-6b;
                                                         - XuanYuan-6B;
                                                         - XuanYuan-13B;
                                                         - Meta-Llama-3-8B;
                                                         - Meta-Llama-3-70B;
   2024-06-05   v0.3.3        Enflame     0.3.3         **1.在gcu上支持了下述模型的推理：**
                                                         - Orion-14B-Chat;
                                                         - Qwen-7B-Chat;
                                                         - Qwen1.5-32B;
                                                         - Qwen1.5-72B-Chat;
                                                         - AquilaChat2-34B-16K;
                                                         - XuanYuan2-70B;
                                                         - iflytekspark_13b;
                                                         - deepseek-moe-16b-base;
                                                         - deepseek-moe-16b-chat;
                                                         - internlm2-7b;
                                                         - internlm2-20b;
                                                         - internlm2-chat-20b;
                                                         - deepseek-coder-6.7b-base;
   2024-06-18   v0.3.3        Enflame     0.3.3         **1.在gcu上支持了下述W8A16模型的推理：**
                                                         - baichuan2-7B-base-w8a16;
                                                         - baichuan2-13B-base-w8a16;
                                                         - bloomz-w8a16;
                                                         - chatglm2-6b-w8a16;
                                                         - chatglm3-6b-w8a16;
                                                         - internlm-7b-w8a16;
                                                         - llama2-7b-w8a16;
                                                         - llama2-13b-w8a16;
                                                         - llama2-70b-w8a16;
                                                         - llama3-8b-w8a16;
                                                         - llama3-70b-w8a16;
                                                         - mixtral-8x7B-v0.1-w8a16;
                                                         - Qwen1.5-14B-Chat-w8a16;
                                                         - Qwen-14B-Chat-w8a16;
                                                         - Qwen-72B-Chat-w8a16;
                                                         - Ziya-Coding-34B-v1.0-w8a16;
   2024-08-08   v0.4.2        Enflame     0.4.2         **1.升级至0.4.2版本;**
   2024-08-13   v0.4.2        Enflame     0.4.2         **1.在gcu上支持了下述模型的推理：**
                                                         - codellama-70b-instruct-w8a16;
                                                         - codegemma-7b;
                                                         - gemma-7b-w8a16;
                                                         - starcoder2-15b-w8a16;
                                                         - Qwen1.5-32B-w8a16;
                                                         - glm-4-9b/chat/glm-4-9b-w8a16;
   2024-08-30   v0.4.2        Enflame     0.4.2         **1.在gcu上支持了下述模型的推理：**
                                                         - Qwen1.5-32B-w4a16;
                                                         - Qwen1.5-32B-w4a16c8;
                                                         - Qwen1.5-MoE-A2.7B;
                                                         - Qwen2-72B-instruct-w4a16c8;
                                                         - dbrx-instruct;
   2024-09-10   v0.4.2        Enflame     0.4.2         **1.在gcu上支持了下述模型的推理：**
                                                         - Qwen2-7B;
                                                         - Qwen-7B-Instruct;
                                                         - Qwen2-72B-padded-w8a16;
                                                         - Qwen2-72B-Instruct;
                                                         - Meta-Llama-3.1-8B-Instruct;
                                                         - Qwen2-1.5B-Instruct;
                                                         - llama2-7b-w4a16;
                                                         - Qwen1.5-4B;
                                                         - Qwen1.5-4B-Chat;
                                                         - Qwen1.5-32B-Chat-w8a16;
                                                         - Qwen1.5-72B-w8a16;
                                                         - Qwen1.5-72B-Chat-w8a16;
                                                         - Meta-Llama-3.1-70B-Instruct;
                                                         - Qwen1.5-72B-w4a16;
                                                         - Qwen2-57B-A14B;
                                                         - deepseek-vl-7b-chat;
                                                         - glm-4v-9b;
                                                         - DeepSeek-V2-Lite-Chat;
                                                         - llama3-70b-w4a16;
                                                         - Mixtral-8x22B-v0.1;
                                                         - Mixtral-8x22B-v0.1-w8a16;
                                                         - qwen2-72b-instruct-gptq-int4;
                                                         - Yi-34B-200K;
                                                         - llama2-7b-w4a16c8;
                                                         - llama2-70b-w4a16c8;
                                                         - Yi-1.5-34B-Chat-GPTQ;
                                                         - SUS-Chat-34B-w8a16;
   2024-11-01   v0.4.2        Enflame     0.4.2         **1.在gcu上支持了下述模型的推理：**
                                                         - internLM-2.5-7B-chat-w4a16
                                                         - Baichuan2-13B-w4a16
                                                         - deepseek-moe-16b-base-w4a16
                                                         - Mixtral-8x7B-v0.1-w4a16
                                                         - Llama-3.1-70B-Instruct-w4a16
                                                         - Llama-2-70b-hf-w8a8c8
                                                         - Llama-2-7b-chat-hf-w8a8c8
                                                         - Qwen2-72B-w8a8c8
                                                         - InternVL-Chat-V1.5
   2024-11-21   v0.6.1.post2  Enflame     0.6.1.post2   **1.升级至0.6.1.post2版本;**
                                                         - 新增支持gptq-int8格式，原w8a16格式不再建议使用
   2024-12-11   v0.6.1.post2  Enflame     0.6.1.post2   **1.在gcu上支持了下述模型的推理：**
                                                         - deepseek-moe-16b-base-w8a8c8
                                                         - qwen1.5-32B-w8a8c8
                                                         - baichuan2_13B-w8a8c8
                                                         - llama3.1_70B_Instruct-w8a8c8
                                                         - qwen2-vl-2b-instruct
                                                         - MiniCPM-v2.6
                                                         - llama3-llava-next-8b
                                                         - Qwen2-VL-7B-Instruct-GPTQ-Int4
                                                         - Phi-3-vision-128k-instruct
   ===========  ============= =========== ============= =======================================================

注：vllm-gcu和原始同版本vllm默认行为有如下差别：
1. 默认关闭vllm统计信息收集
2. 默认关闭async output process功能
3. multi-process executor不支持fork方式，默认使用spawn方式启动
4. topp等后处理使用原精度计算
5. seq 32k以上不默认开启chunked-prefill功能
6. 不支持multi step scheduling功能
7. 默认关闭推理错误时自动dump输入数据功能
