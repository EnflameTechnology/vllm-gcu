已支持的大语言模型列表
====

.. table:: 已支持的大语言模型(注：.表示尚未测试)
   :align: center
   :widths: 16 11 11 11 11 11 11 11 12
   :width: 100%

   ================  =========== =========== =========== =========== =========== =========== =========== ============
   模型                 FP16       BF16      W4A16 GPTQ  W8A16 GPTQ   W4A16 AWQ    W8A16     W8A8 INT8    INT8 KV
   ================  =========== =========== =========== =========== =========== =========== =========== ============
   Baichuan2           Y            .          .            Y             Y         .           Y           Y
   ChatGLM2            Y            .          .            Y             .         .           .           .
   ChatGLM3            Y            .          .            Y             .         .           .           .
   CodeLlama           .            .          .            Y             .         .           .           .
   DBRX                Y            .          .            Y             .         .           .           .
   DeepSeek            Y            .          .            .             .         .           .           .
   DeepSeek-MoE        Y            .          Y            .             .         .           Y           .
   DeepSeek-Coder      Y            .          .            .             .         .           .           .
   DeepSeek-V2-Lite    .            Y          .            .             .         .           .           .
   DeepSeek-Prover-V2  .            Y          .            .             .         .           .           .
   Gemma               Y            .          .            .             .         .           .           .
   codegemma           Y            .          .            .             .         .           .           .
   iFlytekSpark        Y            .          .            .             .         .           .           .
   InternLM2           Y            Y          Y            .             .         .           .           .
   LLaMA2              Y            .          .            Y             Y         Y           Y           Y
   LLaMA3              Y            .          Y            Y             .         .           .           .
   LLaMA3.1            Y            .          Y            .             .         .           .           .
   Mistral             Y            .          .            .             .         .           .           .
   Mixtral             Y            Y          .            Y             .         .           .           .
   Qwen1.5             Y            Y          Y            Y             .         .           Y           .
   Qwen1.5-MoE         .            Y          .            .             .         .           .           .
   Qwen2               Y            Y          Y            Y             Y         .           Y           Y
   Qwen2.5             .            .          .            Y             .         .           .           .
   Qwen3               .            Y          .            .             Y         .           .           .
   Qwen3-MoE           .            Y          .            .             Y         .           .           .
   StarCoder2          Y            .          .            Y             .         .           .           .
   SUS-Chat            .            .          .            Y             .         .           .           .
   WizardCoder         Y            .          .            .             .         .           .           .
   Yi                  Y            .          .            Y             .         .           .           .
   Yi-1.5              .            .          Y            .             .         .           .           .
   Ziya-Coding         Y            .          .            Y             .         .           .           .
   gte-Qwen2           Y            .          .            .             .         .           .           .
   jina-reranker-v2    .            Y          .            .             .         .           .           .
   ================  =========== =========== =========== =========== =========== =========== =========== ============