// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/fast_cpu_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

#if defined(LITERT_LM_ENABLE_FAST_CPU_OPS)
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/core/options.h"  // from @litert
#include "tflite/c/c_api_types.h"  // from @litert
#include "tflite/c/common.h"  // from @litert
#include "tflite/mutable_op_resolver.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert
#endif  // defined(LITERT_LM_ENABLE_FAST_CPU_OPS)

namespace litert::lm {
namespace {

TEST(FastCpuOpsTest, ConfigureFastCpuOpResolverSucceeds) {
  litert::Expected<litert::Options> options = litert::Options::Create();
  ASSERT_TRUE(options.HasValue());
  EXPECT_OK(MaybeConfigureFastCpuOpResolver(*options));
}

TEST(FastCpuOpsTest, FastSliceNdBytesCoalescedAndStridedMatchesReference) {
  for (bool force_portable : {false, true}) {
    // 1. Contiguous trailing dims (> 256 bytes for unrolled prefetched AVX2):
    //    [1, 16, 2, 64] -> [1, 15, 2, 64] (1920 bytes)
    {
      const int64_t in_dims[4] = {1, 16, 2, 64};
      const int64_t begins[4] = {0, 1, 0, 0};
      const int64_t sizes[4] = {1, 15, 2, 64};
      std::vector<uint8_t> src(16 * 2 * 64);
      for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
      }
      std::vector<uint8_t> dst(15 * 2 * 64, 0);
      internal::FastSliceNdBytes(src.data(), in_dims, begins, sizes,
                                 /*rank=*/4, /*elem_bytes=*/1, dst.data(),
                                 force_portable);
      for (size_t i = 0; i < dst.size(); ++i) {
        EXPECT_EQ(dst[i], src[128 + i]);
      }
    }

    // 2. Fused outer loop + non-32-multiple inner tail (32 < 69 < 256):
    //    [1, 2, 8, 70] -> [1, 2, 8, 69]
    {
      const int64_t in_dims[4] = {1, 2, 8, 70};
      const int64_t begins[4] = {0, 0, 0, 1};
      const int64_t sizes[4] = {1, 2, 8, 69};
      std::vector<uint8_t> src(2 * 8 * 70);
      for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>((i * 29 + 5) & 0xFF);
      }
      std::vector<uint8_t> dst(2 * 8 * 69, 0);
      internal::FastSliceNdBytes(src.data(), in_dims, begins, sizes,
                                 /*rank=*/4, /*elem_bytes=*/1, dst.data(),
                                 force_portable);
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 69; ++c) {
          EXPECT_EQ(dst[r * 69 + c], src[r * 70 + 1 + c]);
        }
      }
    }

    // 3. Multi-dimensional strided recursion (dim >= 1) with < 32-byte inner
    //    block (7 * 4 = 28 bytes) and size-1 middle dim:
    //    [3, 4, 5, 9] -> [2, 1, 3, 7]
    {
      const int64_t in_dims[4] = {3, 4, 5, 9};
      const int64_t begins[4] = {1, 2, 1, 2};
      const int64_t sizes[4] = {2, 1, 3, 7};
      std::vector<uint32_t> src(3 * 4 * 5 * 9);
      for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint32_t>(i + 1000);
      }
      std::vector<uint32_t> dst(2 * 1 * 3 * 7, 0);
      internal::FastSliceNdBytes(reinterpret_cast<const uint8_t*>(src.data()),
                                 in_dims, begins, sizes, /*rank=*/4,
                                 /*elem_bytes=*/sizeof(uint32_t),
                                 reinterpret_cast<uint8_t*>(dst.data()),
                                 force_portable);
      size_t out_idx = 0;
      for (int i0 = 1; i0 < 3; ++i0) {
        for (int i2 = 1; i2 < 4; ++i2) {
          for (int i3 = 2; i3 < 9; ++i3) {
            const size_t in_idx = ((i0 * 4 + 2) * 5 + i2) * 9 + i3;
            EXPECT_EQ(dst[out_idx++], src[in_idx]);
          }
        }
      }
    }
  }
}

TEST(FastCpuOpsTest, FastSliceNdBytesHandlesOverlapSelfCopyAndInvalidArgs) {
  for (bool force_portable : {false, true}) {
    // Identical `dst == src` and overlapping `dst < src` / `dst > src` buffers.
    const int64_t dims[1] = {128};
    const int64_t begins_zero[1] = {0};
    const int64_t begins_shift[1] = {8};
    const int64_t sizes_overlap[1] = {96};

    std::vector<uint8_t> buf(128);
    for (size_t i = 0; i < buf.size(); ++i) {
      buf[i] = static_cast<uint8_t>(i + 1);
    }
    const std::vector<uint8_t> orig = buf;

    // Self-copy (`dst == src`) is a safe no-op.
    internal::FastSliceNdBytes(buf.data(), dims, begins_zero, sizes_overlap,
                               /*rank=*/1, /*elem_bytes=*/1, buf.data(),
                               force_portable);
    EXPECT_EQ(buf, orig);

    // Overlapping `dst < src`: copy `buf[8..104)` to `buf[0..96)`.
    internal::FastSliceNdBytes(buf.data(), dims, begins_shift, sizes_overlap,
                               /*rank=*/1, /*elem_bytes=*/1, buf.data(),
                               force_portable);
    for (size_t i = 0; i < 96; ++i) {
      EXPECT_EQ(buf[i], orig[8 + i]);
    }

    // Overlapping `dst > src`: copy `buf[0..96)` to `buf[16..112)`.
    buf = orig;
    internal::FastSliceNdBytes(buf.data(), dims, begins_zero, sizes_overlap,
                               /*rank=*/1, /*elem_bytes=*/1, buf.data() + 16,
                               force_portable);
    for (size_t i = 0; i < 96; ++i) {
      EXPECT_EQ(buf[16 + i], orig[i]);
    }

    // Guarded invalid / empty calls must safely no-op without modifying `dst`.
    std::vector<uint8_t> untouched(16, 0xAB);
    const int64_t rank9_dims[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int64_t rank9_zeros[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t zero_size[1] = {0};
    internal::FastSliceNdBytes(orig.data(), dims, begins_zero, sizes_overlap,
                               /*rank=*/1, /*elem_bytes=*/0, untouched.data(),
                               force_portable);
    internal::FastSliceNdBytes(orig.data(), dims, begins_zero, sizes_overlap,
                               /*rank=*/0, /*elem_bytes=*/1, untouched.data(),
                               force_portable);
    internal::FastSliceNdBytes(orig.data(), rank9_dims, rank9_zeros, rank9_dims,
                               /*rank=*/internal::kMaxTensorRank + 1,
                               /*elem_bytes=*/1, untouched.data(),
                               force_portable);
    internal::FastSliceNdBytes(orig.data(), dims, begins_zero, zero_size,
                               /*rank=*/1, /*elem_bytes=*/1, untouched.data(),
                               force_portable);
    internal::FastSliceNdBytes(nullptr, dims, begins_zero, sizes_overlap,
                               /*rank=*/1, /*elem_bytes=*/1, untouched.data(),
                               force_portable);
    EXPECT_EQ(untouched, std::vector<uint8_t>(16, 0xAB));
  }
}

#if defined(LITERT_LM_ENABLE_FAST_CPU_OPS)

struct ScopedIntArray {
  TfLiteIntArray* ptr = nullptr;
  explicit ScopedIntArray(const std::vector<int>& dims) {
    ptr = TfLiteIntArrayCreate(static_cast<int>(dims.size()));
    for (size_t i = 0; i < dims.size(); ++i) {
      ptr->data[i] = dims[i];
    }
  }
  ~ScopedIntArray() { TfLiteIntArrayFree(ptr); }
  ScopedIntArray(const ScopedIntArray&) = delete;
  ScopedIntArray& operator=(const ScopedIntArray&) = delete;
};

const tflite::MutableOpResolver* ExtractConfiguredResolver() {
  litert::Expected<litert::Environment> env = litert::Environment::Create({});
  if (!env.HasValue()) return nullptr;
  litert::Expected<litert::Options> opts = litert::Options::Create();
  if (!opts.HasValue()) return nullptr;
  if (!MaybeConfigureFastCpuOpResolver(*opts).ok()) return nullptr;
  auto built =
      litert::internal::LiteRtOptionsPtrBuilder::Build(*opts, env->GetHolder());
  if (!built.HasValue()) return nullptr;
  return (*built)->op_resolver;
}

TEST(FastCpuOpsTest, RegisteredSliceKernelEvalsAllElementWidthsAndIndexTypes) {
  const tflite::MutableOpResolver* resolver = ExtractConfiguredResolver();
  ASSERT_NE(resolver, nullptr);

  const TfLiteRegistration* slice_v1 =
      resolver->FindOp(tflite::BuiltinOperator_SLICE, 1);
  const TfLiteRegistration* slice_v8 =
      resolver->FindOp(tflite::BuiltinOperator_SLICE, 8);
  ASSERT_NE(slice_v1, nullptr);
  ASSERT_NE(slice_v8, nullptr);

  ScopedIntArray in_dims({1, 4, 2, 16});
  ScopedIntArray bs_dims({4});
  ScopedIntArray out_dims({1, 3, 2, 16});
  ScopedIntArray inputs({0, 1, 2});
  ScopedIntArray outputs({3});

  // Allocate enough bytes for up to 8-byte elements (4 * 2 * 16 * 8 = 1024 B).
  std::vector<uint8_t> in_bytes(4 * 2 * 16 * 8);
  for (size_t i = 0; i < in_bytes.size(); ++i) {
    in_bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
  }
  std::vector<uint8_t> out_bytes(3 * 2 * 16 * 8, 0);
  std::vector<int32_t> begin32 = {0, 1, 0, 0};
  std::vector<int32_t> size32 = {1, -1, 2, 16};
  std::vector<int64_t> begin64 = {0, 1, 0, 0};
  std::vector<int64_t> size64 = {1, 3, -1, -1};

  TfLiteTensor tensors[4] = {};
  tensors[0].allocation_type = kTfLiteArenaRw;
  tensors[0].dims = in_dims.ptr;
  tensors[0].data.raw = reinterpret_cast<char*>(in_bytes.data());
  tensors[1].dims = bs_dims.ptr;
  tensors[2].dims = bs_dims.ptr;
  tensors[3].allocation_type = kTfLiteArenaRw;
  tensors[3].dims = out_dims.ptr;
  tensors[3].data.raw = reinterpret_cast<char*>(out_bytes.data());

  TfLiteContext ctx = {};
  ctx.tensors_size = 4;
  ctx.tensors = tensors;
  TfLiteNode node = {};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

  struct TypeCase {
    TfLiteType type;
    size_t elem_bytes;
    bool use_int64_indices;
  };
  const TypeCase kCases[] = {
      {kTfLiteInt8, 1, false},
      {kTfLiteInt16, 2, true},
      {kTfLiteFloat32, 4, false},
      {kTfLiteInt64, 8, true},
  };

  for (const TypeCase& tc : kCases) {
    std::fill(out_bytes.begin(), out_bytes.end(), 0);
    tensors[0].type = tc.type;
    tensors[3].type = tc.type;
    if (tc.use_int64_indices) {
      tensors[1].type = kTfLiteInt64;
      tensors[1].data.raw = reinterpret_cast<char*>(begin64.data());
      tensors[2].type = kTfLiteInt64;
      tensors[2].data.raw = reinterpret_cast<char*>(size64.data());
      EXPECT_EQ(slice_v1->invoke(&ctx, &node), kTfLiteOk);
    } else {
      tensors[1].type = kTfLiteInt32;
      tensors[1].data.raw = reinterpret_cast<char*>(begin32.data());
      tensors[2].type = kTfLiteInt32;
      tensors[2].data.raw = reinterpret_cast<char*>(size32.data());
      EXPECT_EQ(slice_v8->invoke(&ctx, &node), kTfLiteOk);
    }
    const size_t offset = 32 * tc.elem_bytes;
    const size_t total = 3 * 2 * 16 * tc.elem_bytes;
    for (size_t i = 0; i < total; ++i) {
      EXPECT_EQ(out_bytes[i], in_bytes[offset + i]);
    }
  }
}

void NoOpReportError(TfLiteContext* /*context*/, const char* /*fmt*/, ...) {}

TEST(FastCpuOpsTest, RegisteredSliceKernelHandlesEmptySlicesAndFallbacks) {
  const tflite::MutableOpResolver* resolver = ExtractConfiguredResolver();
  ASSERT_NE(resolver, nullptr);
  const TfLiteRegistration* slice_op =
      resolver->FindOp(tflite::BuiltinOperator_SLICE, 8);
  ASSERT_NE(slice_op, nullptr);

  ScopedIntArray in_dims({2, 4});
  ScopedIntArray bs_dims({2});
  ScopedIntArray empty_out_dims({0, 4});
  ScopedIntArray valid_out_dims({1, 4});
  ScopedIntArray wrong_out_dims({2, 4});
  ScopedIntArray inputs({0, 1, 2});
  ScopedIntArray outputs({3});

  std::vector<float> in_data(8, 1.5f);
  std::vector<float> out_data(4, 0.0f);
  std::vector<int32_t> begin32 = {1, 0};
  std::vector<int32_t> size32_empty = {0, 4};

  TfLiteTensor tensors[4] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[0].allocation_type = kTfLiteArenaRw;
  tensors[0].dims = in_dims.ptr;
  tensors[0].data.raw = reinterpret_cast<char*>(in_data.data());
  tensors[1].type = kTfLiteInt32;
  tensors[1].dims = bs_dims.ptr;
  tensors[1].data.raw = reinterpret_cast<char*>(begin32.data());
  tensors[2].type = kTfLiteInt32;
  tensors[2].dims = bs_dims.ptr;
  tensors[2].data.raw = reinterpret_cast<char*>(size32_empty.data());
  tensors[3].type = kTfLiteFloat32;
  tensors[3].allocation_type = kTfLiteArenaRw;
  tensors[3].dims = empty_out_dims.ptr;
  tensors[3].data.raw = nullptr;

  TfLiteContext ctx = {};
  ctx.tensors_size = 4;
  ctx.tensors = tensors;
  ctx.ReportError = &NoOpReportError;
  TfLiteNode node = {};
  node.inputs = inputs.ptr;
  node.outputs = outputs.ptr;

  // Empty slice (`sizes[0] == 0`) succeeds immediately even with null output
  // data pointer.
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteOk);

  // Fallback on 0-D scalar slice (`rank == 0`): base TFLite SLICE copies the
  // scalar element cleanly.
  {
    ScopedIntArray scalar_dims({});
    ScopedIntArray zero_len_dims({0});
    float scalar_in = 42.5f;
    float scalar_out = 0.0f;
    int32_t dummy_idx = 0;
    tensors[0].dims = scalar_dims.ptr;
    tensors[0].data.raw = reinterpret_cast<char*>(&scalar_in);
    tensors[1].dims = zero_len_dims.ptr;
    tensors[1].data.raw = reinterpret_cast<char*>(&dummy_idx);
    tensors[2].dims = zero_len_dims.ptr;
    tensors[2].data.raw = reinterpret_cast<char*>(&dummy_idx);
    tensors[3].dims = scalar_dims.ptr;
    tensors[3].data.raw = reinterpret_cast<char*>(&scalar_out);
    EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteOk);
    EXPECT_EQ(scalar_out, 42.5f);
  }

  // Restore 2-D tensors to exercise remaining fallback guards cleanly (base
  // TFLite SLICE rejects invalid bounds/types via `ReportError` -> error).
  tensors[0].dims = in_dims.ptr;
  tensors[0].data.raw = reinterpret_cast<char*>(in_data.data());
  tensors[1].dims = bs_dims.ptr;
  tensors[2].dims = bs_dims.ptr;
  tensors[3].dims = valid_out_dims.ptr;
  tensors[3].data.raw = reinterpret_cast<char*>(out_data.data());

  // Negative `begin` triggers fallback (`kTfLiteError`).
  std::vector<int32_t> neg_begin = {-1, 0};
  std::vector<int32_t> valid_size = {1, 4};
  tensors[1].data.raw = reinterpret_cast<char*>(neg_begin.data());
  tensors[2].data.raw = reinterpret_cast<char*>(valid_size.data());
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);

  // Invalid `size < -1` triggers fallback (`kTfLiteError`).
  tensors[1].data.raw = reinterpret_cast<char*>(begin32.data());
  std::vector<int32_t> bad_neg_size = {-2, 4};
  tensors[2].data.raw = reinterpret_cast<char*>(bad_neg_size.data());
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);

  // Out-of-bounds `size > dim - begin` triggers fallback (`kTfLiteError`).
  std::vector<int32_t> oob_size = {2, 4};
  tensors[2].data.raw = reinterpret_cast<char*>(oob_size.data());
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);

  // Mismatched `output->dims` triggers fallback (`kTfLiteError`).
  tensors[2].data.raw = reinterpret_cast<char*>(valid_size.data());
  tensors[3].dims = wrong_out_dims.ptr;
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);
  tensors[3].dims = valid_out_dims.ptr;

  // Optional/negative tensor index in `node->inputs` exercises `GetNodeTensor`
  // bounds guard and triggers fallback (`kTfLiteError`).
  inputs.ptr->data[0] = kTfLiteOptionalTensor;
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);
  inputs.ptr->data[0] = 0;

  // Unsupported element type (`kTfLiteResource`, `elem_bytes == 0`) triggers
  // fallback (`kTfLiteError`).
  tensors[0].type = kTfLiteResource;
  tensors[3].type = kTfLiteResource;
  EXPECT_EQ(slice_op->invoke(&ctx, &node), kTfLiteError);
}

#endif  // defined(LITERT_LM_ENABLE_FAST_CPU_OPS)

}  // namespace
}  // namespace litert::lm
