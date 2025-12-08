## fusedmoe
### 功能介绍

融合MoE性能提升：通过优化融合技术，使用混合专家（MoE）架构的模型性能得到了显著提升。这一更新增强了专家层计算的执行效率，减少了开销，并提高了利用大规模MoE模型的任务的吞吐量。

### 使用方法

MoE类模型默认都会使能FusedMoE功能。在多卡推理情况下，默认MoE采用的Tensor Parallelism (TP)的并行方式，可以通过添加参数`enable_expert_parallel=True`使能[Expert Parallelism (EP)](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html#expert-parallelism-ep)。