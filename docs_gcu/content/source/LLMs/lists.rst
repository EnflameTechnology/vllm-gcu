已支持的大语言模型列表
====

.. table:: 已支持的大语言模型(注：.表示尚未测试)
   :align: center
   :widths: 18 11 11 11 11 11 11 11 12
   :width: 100%

   ==================   =========== =========== =========== =========== =========== =========== =========== ============
   模型                    FP16       BF16       W4A16 GPTQ  W8A16 GPTQ   W4A16 AWQ    W8A16     W8A8 INT8    INT8 KV
   ==================   =========== =========== =========== =========== =========== =========== =========== ============
   Baichuan2               Y            .          .            Y             Y         .           Y           Y
   ChatGLM3                Y            .          .            Y             .         .           .           .
   DBRX                    Y            .          .            Y             .         .           .           .
   DeepSeek-MoE            .            .          .            .             .         .           Y           .
   DeepSeek-V3             .            .          .            .             Y         .           .           .
   DeepSeek-Prover-V2      .            Y          .            .             .         .           .           .
   Gemma                   Y            .          .            .             .         .           .           .
   codegemma               Y            .          .            .             .         .           .           .
   InternLM2               .            .          Y            .             .         .           .           .
   LLaMA2                  Y            .          .            Y             Y         Y           Y           Y
   LLaMA3                  Y            .          Y            Y             .         .           .           .
   LLaMA3.1                Y            .          Y            .             .         .           .           .
   Mixtral                 .            Y          .            .             .         .           .           .
   Qwen1.5                 Y            .          Y            Y             .         .           Y           .
   Qwen2                   .            .          .            Y             .         .           .           .
   Qwen2.5                 .            Y          .            Y             .         .           .           .
   Qwen3                   .            Y          .            .             Y         .           .           .
   Qwen3-MoE               .            Y          .            .             Y         .           .           .
   WizardCoder             Y            .          .            .             .         .           .           .
   Yi                      .            .          .            Y             .         .           .           .
   gte-Qwen2               Y            .          .            .             .         .           .           .
   jina-reranker-v2        .            Y          .            .             .         .           .           .
   ==================   =========== =========== =========== =========== =========== =========== =========== ============