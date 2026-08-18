/* Copyright 2026 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/executor/litert/custom_ops/attention_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace litert {
namespace lm {
namespace {

// Reference implementation of Scaled Dot Product Attention (Transposed Layout).
void ComputeReferenceAttention(const std::vector<float>& query,
                               const std::vector<float>& key,
                               const std::vector<float>& value,
                               const std::vector<float>& mask, int gt,
                               int s_limit, int s_stride, int h, int k_ts_idx,
                               int v_ts_idx, std::optional<float> softcap,
                               std::vector<float>& ref_out) {
  ref_out.assign(gt * h, 0.0f);

  for (int q_idx = 0; q_idx < gt; ++q_idx) {
    std::vector<double> logits(s_limit, 0.0);
    double max_logit = -1e30;

    for (int s_idx = 0; s_idx < s_limit; ++s_idx) {
      double dot = 0.0;
      for (int d = 0; d < h; ++d) {
        double q_val = query[q_idx * h + d];
        double k_val =
            (k_ts_idx == 2) ? key[s_idx * h + d] : key[d * s_stride + s_idx];
        dot += q_val * k_val;
      }
      if (softcap.has_value()) {
        dot = std::tanh(dot / softcap.value()) * softcap.value();
      }
      double mask_val = mask[q_idx * s_stride + s_idx];
      dot += mask_val;
      logits[s_idx] = dot;
      if (dot > max_logit) {
        max_logit = dot;
      }
    }

    double sum_exp = 0.0;
    std::vector<double> probs(s_limit, 0.0);
    for (int s_idx = 0; s_idx < s_limit; ++s_idx) {
      probs[s_idx] = std::exp(logits[s_idx] - max_logit);
      sum_exp += probs[s_idx];
    }

    for (int s_idx = 0; s_idx < s_limit; ++s_idx) {
      probs[s_idx] /= sum_exp;
    }

    for (int d = 0; d < h; ++d) {
      double out_val = 0.0;
      for (int s_idx = 0; s_idx < s_limit; ++s_idx) {
        double v_val = (v_ts_idx == 3) ? value[d * s_stride + s_idx]
                                       : value[s_idx * h + d];
        out_val += probs[s_idx] * v_val;
      }
      ref_out[q_idx * h + d] = static_cast<float>(out_val);
    }
  }
}

void TestAttentionPath(int gt, int s, int h, int active_seq_len, int k_ts_idx,
                       int v_ts_idx,
                       std::optional<float> softcap = std::nullopt) {
  const int bk = 1;
  std::mt19937 rng(42 + k_ts_idx * 10 + v_ts_idx);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> query(gt * h);
  for (auto& v : query) v = dist(rng);

  std::vector<float> key(s * h);
  for (auto& v : key) v = dist(rng);

  std::vector<float> value(h * s);
  for (auto& v : value) v = dist(rng);

  std::vector<float> mask(gt * s, 0.0f);
  for (int q = 0; q < gt; ++q) {
    for (int k = 0; k < s; ++k) {
      if (k > q || k >= active_seq_len) {
        mask[q * s + k] = -1e30f;
      }
    }
  }

  std::vector<float> ref_out(gt * h, 0.0f);
  ComputeReferenceAttention(query, key, value, mask, gt, active_seq_len, s, h,
                            k_ts_idx, v_ts_idx, softcap, ref_out);

  TensorRef query_ref = {query.data(), {1, bk, gt, h}};
  TensorRef key_ref = {key.data(), (k_ts_idx == 2)
                                       ? std::vector<int32_t>{1, bk, s, h}
                                       : std::vector<int32_t>{1, bk, h, s}};
  TensorRef value_ref = {value.data(), (v_ts_idx == 3)
                                           ? std::vector<int32_t>{1, bk, h, s}
                                           : std::vector<int32_t>{1, bk, s, h}};
  TensorRef mask_ref = {mask.data(), {1, 1, gt, s}};

  std::vector<float> kernel_out(gt * h, 0.0f);
  MutableTensorRef output_ref = {kernel_out.data(), {1, bk, gt, h}};

  bool status = ComputeTransposedAttentionSingleHead(
      /*h_idx=*/0, query_ref, key_ref, value_ref, mask_ref, softcap, k_ts_idx,
      v_ts_idx, output_ref, active_seq_len, /*gt_range=*/std::nullopt,
      /*num_threads=*/1, /*logits_scratch=*/nullptr, /*out_scratch=*/nullptr,
      /*gemm_context=*/nullptr);

  ASSERT_TRUE(status);

  float max_diff = 0.0f;
  for (size_t i = 0; i < ref_out.size(); ++i) {
    float diff = std::abs(ref_out[i] - kernel_out[i]);
    if (diff > max_diff) max_diff = diff;
    EXPECT_NEAR(ref_out[i], kernel_out[i], 1e-4f) << "At index " << i;
  }
  EXPECT_LT(max_diff, 1e-4f);
}

TEST(AttentionUtilsTest, VerifyDecode_K2_V3) {
  TestAttentionPath(/*gt=*/1, /*s=*/128, /*h=*/128, /*active_seq_len=*/16, 2,
                    3);
}

TEST(AttentionUtilsTest, VerifyDecode_K2_V2) {
  TestAttentionPath(/*gt=*/1, /*s=*/128, /*h=*/128, /*active_seq_len=*/16, 2,
                    2);
}

TEST(AttentionUtilsTest, VerifyDecode_K3_V3) {
  TestAttentionPath(/*gt=*/1, /*s=*/128, /*h=*/128, /*active_seq_len=*/16, 3,
                    3);
}

TEST(AttentionUtilsTest, VerifyDecode_K3_V2) {
  TestAttentionPath(/*gt=*/1, /*s=*/128, /*h=*/128, /*active_seq_len=*/16, 3,
                    2);
}

TEST(AttentionUtilsTest, VerifyPrefill_K2_V3) {
  TestAttentionPath(/*gt=*/17, /*s=*/128, /*h=*/128, /*active_seq_len=*/17, 2,
                    3);
}

TEST(AttentionUtilsTest, VerifyPrefill_K2_V2) {
  TestAttentionPath(/*gt=*/17, /*s=*/128, /*h=*/128, /*active_seq_len=*/17, 2,
                    2);
}

TEST(AttentionUtilsTest, VerifyPrefill_K3_V3) {
  TestAttentionPath(/*gt=*/17, /*s=*/128, /*h=*/128, /*active_seq_len=*/17, 3,
                    3);
}

TEST(AttentionUtilsTest, VerifyPrefill_K3_V2) {
  TestAttentionPath(/*gt=*/17, /*s=*/128, /*h=*/128, /*active_seq_len=*/17, 3,
                    2);
}

TEST(AttentionUtilsTest, VerifyDecode_H256) {
  TestAttentionPath(/*gt=*/1, /*s=*/512, /*h=*/256, /*active_seq_len=*/64, 2,
                    3);
}

TEST(AttentionUtilsTest, VerifyDecode_H64) {
  TestAttentionPath(/*gt=*/1, /*s=*/256, /*h=*/64, /*active_seq_len=*/32, 2, 3);
}

TEST(AttentionUtilsTest, VerifySoftcap) {
  TestAttentionPath(/*gt=*/1, /*s=*/128, /*h=*/128, /*active_seq_len=*/16, 2, 3,
                    /*softcap=*/50.0f);
}

TEST(AttentionUtilsTest, VerifyMultiHeadComputeTransposedAttention) {
  const int bk = 8;
  const int gt = 17;
  const int s = 128;
  const int h = 128;
  const int active_seq_len = 17;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> query(bk * gt * h);
  for (auto& v : query) v = dist(rng);

  std::vector<float> key(bk * s * h);
  for (auto& v : key) v = dist(rng);

  std::vector<float> value(bk * h * s);
  for (auto& v : value) v = dist(rng);

  std::vector<float> mask(gt * s, 0.0f);
  for (int q = 0; q < gt; ++q) {
    for (int k = 0; k < s; ++k) {
      if (k > q || k >= active_seq_len) {
        mask[q * s + k] = -1e30f;
      }
    }
  }

  TensorRef query_ref = {query.data(), {1, bk, gt, h}};
  TensorRef key_ref = {key.data(), {1, bk, s, h}};
  TensorRef value_ref = {value.data(), {1, bk, h, s}};
  TensorRef mask_ref = {mask.data(), {1, 1, gt, s}};

  std::vector<float> kernel_out(bk * gt * h, 0.0f);
  MutableTensorRef output_ref = {kernel_out.data(), {1, bk, gt, h}};

  bool status = ComputeTransposedAttention(
      query_ref, key_ref, value_ref, mask_ref,
      /*logit_cap=*/std::nullopt, /*k_ts_idx=*/2, /*v_ts_idx=*/3, output_ref,
      /*active_seq_len=*/active_seq_len);

  ASSERT_TRUE(status);
}

}  // namespace
}  // namespace lm
}  // namespace litert
