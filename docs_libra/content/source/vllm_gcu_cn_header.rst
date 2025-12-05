前言
====

本文档介绍vLLM-gcu的功能和使用方法，包括模型批量离线推理示例、性能评估及在特定数据集上的精度验证。


版本信息
--------

.. table:: 版本信息
   :align: center
   :widths: 15 20 15 10 40
   :width: 100%

   ===========  ========= ========= ============ =========================================================================
   日期         版本      作者      原始vllm版本 新增功能
   ===========  ========= ========= ============ =========================================================================
   2025-09-25   v0.9.2     Enflame     0.9.2     | **1.vllm-gcu升级到0.9.2版本**
                                                 | **2.支持DeepSeek-R1模型**
                                                 | • DP4-TP4SP4-EP16并行推理
   2025-12-04   v0.11.0    Enflame     0.11.0    | **1.vllm-gcu升级到0.11.0版本**
                                                 | **2.DeepSeek-R1模型**
                                                 | • 支持 P/D 分离、MTP1 等
                                                 | **3.DeepSeek-V3.1-Terminus 模型**
                                                 | • 支持 DP2-TP4SP4-EP8 并行推理
                                                 | • 支持 DP4-TP4SP4-EP16 并行推理
                                                 | • 支持 P/D 分离、MTP1 等
                                                 | **4.新增特性**
                                                 | • 运行时图模式增强（Runtime Graph Mode Enhancement）
                                                 | • MoE 模型深度优化（MOE deepep ll+deepgemm masked）
                                                 | • 异步调度与计算重叠（Async Scheduling + Overlap Model Execution）
                                                 | • 投机解码异步调度适配（Async Scheduling Supports Eagle Spec Decode）
                                                 | • 投机解码图模式兼容（Eagle Spec Decode Supports Runtime Graph）
                                                 | • KV 缓存卸载（KV Cache Offloading）
                                                 | • 采样器数据并行（Sampler Data Parallel）
                                                 | • 增加性能分析追踪点（Add Profiling Trace Points）
   ===========  ========= ========= ============ =========================================================================

注：vllm-gcu和原始同版本vllm默认行为有如下差别：

1. 默认关闭vllm统计信息收集

2. multi-process executor不支持fork方式，默认使用spawn方式启动