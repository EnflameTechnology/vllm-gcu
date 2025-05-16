## vLLM
vLLM是高效易用的大语言模型推理库，基于PagedAttention高效地进行attention key和value的内存管理，提高实时场景下大语言模型服务的吞吐与内存使用效率。

作为一个使用Python编写的开源推理库，vLLM紧跟业界发展趋势快速迭代，支持量化模型推理、multi-lora、chunked prefill、Speculative decoding推理等特性。总之，vLLM因其易用性和先进性得到了广泛的关注和应用。

## vLLM-gcu
vLLM-gcu是适配于燧原S60 gcu的vLLM，用于支持在Enflame gcu上运行大语言模型和视觉大语言模型的推理。

vLLM-gcu维持vLLM中的模型推理、request调度策略，只是在Enflame gcu设备端完成相关算子地高效计算。

## 版本信息与使用注意事项
当前vLLM-gcu是与Enflame gcu适配的vLLM 0.8.0版本，其使用方式与vLLM 0.8.0版本基本一致，但具备如下特性：

- 推理时，需设置`--device=gcu`；
- attention计算，仅支持xformers backend；
- 默认关闭vllm统计信息收集；
- 默认关闭async output process功能；
- multi-process executor不支持fork方式，默认使用spawn方式启动；
- top-p等后处理使用原精度计算；
- seq 32k以上不默认开启chunked-prefill功能；
- 默认关闭推理错误时自动dump输入数据功能；