/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mha_fwd_int8kv.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <tuple>
#include <vector>
#include <cmath>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_int8kv(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const c10::optional<at::Tensor> &k_new_,
        const c10::optional<at::Tensor> &v_new_,
        const c10::optional<at::Tensor> &q_v_,
        const c10::optional<at::Tensor> &out_,
        const c10::optional<at::Tensor> &cu_seqlens_q_,
        const c10::optional<at::Tensor> &cu_seqlens_k_,
        const c10::optional<at::Tensor> &cu_seqlens_k_new_,
        const c10::optional<at::Tensor> &seqused_q_,
        const c10::optional<at::Tensor> &seqused_k_,
        const c10::optional<int64_t> max_seqlen_q_,
        const c10::optional<int64_t> max_seqlen_k_,
        const c10::optional<at::Tensor> &page_table_,
        const c10::optional<at::Tensor> &kv_batch_idx_,
        const c10::optional<at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &rotary_cos_,
        const c10::optional<at::Tensor> &rotary_sin_,
        const c10::optional<at::Tensor> &seqlens_rotary_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const c10::optional<at::Tensor> &k_zp_,
        const c10::optional<at::Tensor> &v_zp_,
        const c10::optional<double> softmax_scale_,
        bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        // int64_t attention_chunk,
        const double softcap,
        bool is_rotary_interleaved,
        const c10::optional<at::Tensor> &scheduler_metadata_,
        const int64_t num_splits,
        const c10::optional<bool> pack_gqa_,
        const int64_t sm_margin,
        const c10::optional<at::Tensor> &s_aux_
) {
    const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();

    bool const is_varlen_q = cu_seqlens_q_.has_value();
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    const bool paged_KV = page_table_.has_value();

    auto q_dtype = q.dtype();
    auto opts = q.options();
    caffe2::TypeMeta out_type;
    out_type = q_dtype;

    at::Tensor k_new;
    if (k_new_.has_value()) {
        k_new = k_new_.value();
    }

    at::Tensor v_new;
    if (v_new_.has_value()) {
        v_new = v_new_.value();
    }

    at::Tensor q_v;
    if (q_v_.has_value()) {
        q_v = q_v_.value();
    }

    at::Tensor cu_seqlens_q;
    if (cu_seqlens_q_.has_value()) {
        cu_seqlens_q = cu_seqlens_q_.value();
    }

    at::Tensor cu_seqlens_k;
    if (cu_seqlens_k_.has_value()) {
        cu_seqlens_k = cu_seqlens_k_.value();
    }

    at::Tensor cu_seqlens_k_new;
    if (cu_seqlens_k_new_.has_value()) {
        cu_seqlens_k_new = cu_seqlens_k_new_.value();
    }

    at::Tensor seqused_q;
    if (seqused_q_.has_value()) {
        seqused_q = seqused_q_.value();
    }

    at::Tensor seqused_k;
    if (seqused_k_.has_value()) {
        seqused_k = seqused_k_.value();
    }

    at::Scalar max_seqlen_q_scalar(max_seqlen_q_.value());
    at::Scalar max_seqlen_k_scalar(max_seqlen_k_.value());

    at::Tensor page_table;
    if (page_table_.has_value()) {
        page_table = page_table_.value();
    }

    at::Tensor kv_batch_idx;
    if (kv_batch_idx_.has_value()) {
        kv_batch_idx = kv_batch_idx_.value();
    }

    at::Tensor leftpad_k;
    if (leftpad_k_.has_value()) {
        leftpad_k = leftpad_k_.value();
    }

    at::Tensor rotary_cos;
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }

    at::Tensor rotary_sin;
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }

    at::Tensor seqlens_rotary;
    if (seqlens_rotary_.has_value()) {
        seqlens_rotary = seqlens_rotary_.value();
    }

    at::Tensor q_descale;
    if (q_descale_.has_value()) {
        q_descale = q_descale_.value();
    }

    at::Tensor k_descale;
    if (k_descale_.has_value()) {
        k_descale = k_descale_.value();
    }

    at::Tensor v_descale;
    if (v_descale_.has_value()) {
        v_descale = v_descale_.value();
    }

    at::Tensor k_zp;
    if (k_zp_.has_value()) {
        k_zp = k_zp_.value();
    }

    at::Tensor v_zp;
    if (v_zp_.has_value()) {
        v_zp = v_zp_.value();
    }

    at::Scalar softmax_scale_scalar;
    if (softmax_scale_.has_value()) {
        softmax_scale_scalar = softmax_scale_.value();
    }

    at::Scalar window_size_left_scalar(window_size_left);
    at::Scalar window_size_right_scalar(window_size_right);

    at::Scalar softcap_scalar(softcap);

    at::Tensor scheduler_metadata;
    if (scheduler_metadata_.has_value()) {
        scheduler_metadata = scheduler_metadata_.value();
    }

    at::Scalar num_splits_scalar(num_splits);

    bool pack_gqa;
    if (pack_gqa_.has_value()) {
        pack_gqa = pack_gqa_.value();
    } else {
        pack_gqa = false;
    }

    at::Scalar sm_margin_scalar(sm_margin);

    at::Tensor s_aux;
    if (s_aux_.has_value()) {
        s_aux = s_aux_.value();
    }

    const int batch_size = !is_varlen_q ? q.size(0) : cu_seqlens_q.size(0) - 1;
    int seqlen_q = !is_varlen_q ? q.size(1) : max_seqlen_q_.value();
    int total_q = !is_varlen_q ? batch_size * q.size(1) : q.size(0);
    int num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const head_size_v = v.size(-1);
    int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
    int const num_pages = !paged_KV ? 0 : k.size(0);
    int const page_size = !paged_KV ? 1 : k.size(1);
    int const seqlen_k = !is_varlen_k ?
                (!paged_KV ? k.size(1) : max_num_pages_per_seq * page_size) :
                max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k = k.size(-2);
    int const batch_size_k = !paged_KV ?
                        (!is_varlen_k ? k.size(0) : cu_seqlens_k.size(0) - 1) :
                        page_table.size(0);

    // create out tensor if needed
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
    } else {
        out = !is_varlen_q
            ? torch::empty({batch_size, seqlen_q, num_heads, head_size_v},
                           opts.dtype(out_type))
            : torch::empty({total_q, num_heads, head_size_v},
                           opts.dtype(out_type));
    }

    // create softmax_lse tensor
    at::Tensor softmax_lse;
    if (!is_varlen_q) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
            opts.dtype(at::kFloat));
    } else {
        softmax_lse = torch::empty({num_heads, total_q},
            opts.dtype(at::kFloat));
    }

    // not used by aten op for now, so empty tensor as placeholder
    at::Tensor out_accum, softmax_lse_accum;

    std::vector<at::Tensor> out_vector = {out, softmax_lse,
                                out_accum, softmax_lse_accum};

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFlashAttnFwdInt8KV)(
        out_vector,
        q, k, v,
        k_new, v_new,
        q_v,
        out,
        cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new,
        seqused_q, seqused_k,
        max_seqlen_q_scalar, max_seqlen_k_scalar,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos, rotary_sin, seqlens_rotary,
        q_descale, k_descale, v_descale,
        k_zp, v_zp,
        softmax_scale_scalar,
        is_causal,
        window_size_left_scalar, window_size_right_scalar,
        // attention_chunk,
        softcap_scalar,
        is_rotary_interleaved,
        scheduler_metadata,
        num_splits_scalar,
        pack_gqa,
        sm_margin_scalar,
        s_aux,
        stream));

    return {out, softmax_lse, out_accum, softmax_lse_accum};
}
}  // namespace vllm_gcu::llm_ops
