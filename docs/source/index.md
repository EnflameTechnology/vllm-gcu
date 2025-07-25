# Welcome to vLLM-GCU

:::{figure} ./logos/vllm-gcu.png
:align: center
:alt: vLLM-GCU
:class: no-scaled-link
:width: 70%
:::

:::{raw} html

<p style="text-align:center">
<strong>vLLM-GCU</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-gcu" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-gcu/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-gcu/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

**vLLM-GCU** is a high-performance backend plugin that enables efficient inference for Large Language Models (LLMs) and Vision-Language Models (VLMs) on **Enflame GCU** hardware. Built on the vLLM framework, this plugin leverages GCU-specific operator optimization and runtime integration via TopsRider and GCU-aware extensions.

It is the **recommended integration method** for deploying LLMs on Enflame’s GCU platforms. The plugin conforms to the [vLLM Hardware Pluggable Interface RFC](https://github.com/vllm-project/vllm/issues/11162), enabling hardware decoupling and modular backend development.

With vLLM-GCU, users can deploy a wide range of Transformer-based, Mixture-of-Experts, multi-modal, and quantized models optimized for GCU inference workloads.

---

## Documentation

% How to get started with vLLM-GCU?
:::{toctree}
:caption: Getting Started
:maxdepth: 1
quick_start
installation
tutorials/index.md
faqs
:::

% What models and features are supported on GCU?
:::{toctree}
:caption: User Guide
:maxdepth: 1
user_guide/support_matrix/index
user_guide/configuration/index
user_guide/feature_guide/index
:::

% How to contribute and extend vLLM-GCU?
:::{toctree}
:caption: Developer Guide
:maxdepth: 1
developer_guide/evaluation/index
developer_guide/performance/index
developer_guide/modeling/index
:::
