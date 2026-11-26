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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert

#if defined(LITERT_LM_ENABLE_FAST_CPU_OPS)
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/core/options.h"  // from @litert
#include "tflite/c/c_api_types.h"  // from @litert
#include "tflite/c/common.h"  // from @litert
#include "tflite/kernels/register.h"  // from @litert
#include "tflite/mutable_op_resolver.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert
#endif  // defined(LITERT_LM_ENABLE_FAST_CPU_OPS)

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "runtime/executor/fast_cpu_ops_inl.h"
#include "hwy/foreach_target.h"  // from @highway  // IWYU pragma: keep
#include "hwy/highway.h"  // from @highway
#include "runtime/executor/fast_cpu_ops_inl.h"
// clang-format on

namespace litert::lm {
namespace internal {

HWY_EXPORT(RunStridedSliceSimd);

namespace {

inline bool BuffersOverlap(const uint8_t* a, const uint8_t* b, size_t n) {
  const auto ua = reinterpret_cast<uintptr_t>(a);
  const auto ub = reinterpret_cast<uintptr_t>(b);
  return (ua < ub) ? (ua + n > ub) : (ub + n > ua);
}

void RunStridedSlicePortable(const uint8_t* src, uint8_t* dst,
                             const size_t* counts, const size_t* src_strides,
                             const size_t* dst_strides, int dim,
                             size_t block_bytes) {
  if (dim < 0) {
    if (dst != src && block_bytes > 0) {
      if (BuffersOverlap(dst, src, block_bytes)) {
        std::memmove(dst, src, block_bytes);
      } else {
        std::memcpy(dst, src, block_bytes);
      }
    }
    return;
  }
  const size_t n = counts[dim];
  const size_t ss = src_strides[dim];
  const size_t ds = dst_strides[dim];
  for (size_t i = 0; i < n; ++i) {
    RunStridedSlicePortable(src + i * ss, dst + i * ds, counts, src_strides,
                            dst_strides, dim - 1, block_bytes);
  }
}

}  // namespace

void FastSliceNdBytes(const uint8_t* src, const int64_t* in_dims,
                      const int64_t* begins, const int64_t* sizes, int rank,
                      size_t elem_bytes, uint8_t* dst, bool force_portable) {
  if (elem_bytes == 0 || rank <= 0 || rank > kMaxTensorRank || src == nullptr ||
      dst == nullptr || in_dims == nullptr || begins == nullptr ||
      sizes == nullptr) {
    return;
  }
  for (int d = 0; d < rank; ++d) {
    if (sizes[d] <= 0) {
      return;
    }
  }

  size_t in_byte_strides[kMaxTensorRank];
  in_byte_strides[rank - 1] = elem_bytes;
  for (int d = rank - 2; d >= 0; --d) {
    in_byte_strides[d] =
        in_byte_strides[d + 1] * static_cast<size_t>(in_dims[d + 1]);
  }

  size_t src_base_offset = 0;
  for (int d = 0; d < rank; ++d) {
    src_base_offset += static_cast<size_t>(begins[d]) * in_byte_strides[d];
  }
  const uint8_t* src_base = src + src_base_offset;

  size_t block_bytes = elem_bytes;
  size_t dst_accum_bytes = elem_bytes;
  size_t loop_counts[kMaxTensorRank];
  size_t loop_src_strides[kMaxTensorRank];
  size_t loop_dst_strides[kMaxTensorRank];
  int top = -1;

  for (int d = rank - 1; d >= 0; --d) {
    const size_t extent = static_cast<size_t>(sizes[d]);
    if (extent == 1) {
      continue;
    }
    const size_t s_stride = in_byte_strides[d];
    if (top < 0 && s_stride == block_bytes) {
      block_bytes *= extent;
    } else if (top >= 0 &&
               s_stride == loop_counts[top] * loop_src_strides[top]) {
      loop_counts[top] *= extent;
    } else {
      ++top;
      loop_counts[top] = extent;
      loop_src_strides[top] = s_stride;
      loop_dst_strides[top] = dst_accum_bytes;
    }
    dst_accum_bytes *= extent;
  }

  if (!force_portable) {
    HWY_DYNAMIC_DISPATCH(RunStridedSliceSimd)
    (src_base, dst, loop_counts, loop_src_strides, loop_dst_strides, top,
     block_bytes);
    return;
  }
  RunStridedSlicePortable(src_base, dst, loop_counts, loop_src_strides,
                          loop_dst_strides, top, block_bytes);
}

}  // namespace internal

#if defined(LITERT_LM_ENABLE_FAST_CPU_OPS)
namespace {

const TfLiteTensor* GetNodeTensor(const TfLiteContext* context,
                                  const TfLiteIntArray* array, int index) {
  if (context == nullptr || context->tensors == nullptr || array == nullptr ||
      index < 0 || index >= array->size) {
    return nullptr;
  }
  const int tensor_index = array->data[index];
  if (tensor_index < 0 ||
      static_cast<size_t>(tensor_index) >= context->tensors_size) {
    return nullptr;
  }
  return &context->tensors[tensor_index];
}

TfLiteTensor* GetMutableNodeTensor(TfLiteContext* context,
                                   const TfLiteIntArray* array, int index) {
  return const_cast<TfLiteTensor*>(GetNodeTensor(context, array, index));
}

size_t GetElementBytes(TfLiteType type) {
  switch (type) {
    case kTfLiteBool:
    case kTfLiteInt8:
    case kTfLiteUInt8:
      return 1;
    case kTfLiteInt16:
    case kTfLiteUInt16:
    case kTfLiteFloat16:
    case kTfLiteBFloat16:
      return 2;
    case kTfLiteInt32:
    case kTfLiteUInt32:
    case kTfLiteFloat32:
      return 4;
    case kTfLiteInt64:
    case kTfLiteUInt64:
    case kTfLiteFloat64:
      return 8;
    default:
      return 0;
  }
}

TfLiteRegistration g_base_slice = {};

TfLiteStatus FastSliceEval(TfLiteContext* context, TfLiteNode* node) {
  if (context == nullptr || node == nullptr || node->inputs == nullptr ||
      node->outputs == nullptr || node->inputs->size != 3 ||
      node->outputs->size != 1) {
    return g_base_slice.invoke(context, node);
  }

  const TfLiteTensor* input = GetNodeTensor(context, node->inputs, 0);
  const TfLiteTensor* begin = GetNodeTensor(context, node->inputs, 1);
  const TfLiteTensor* size = GetNodeTensor(context, node->inputs, 2);
  TfLiteTensor* output = GetMutableNodeTensor(context, node->outputs, 0);
  if (input == nullptr || begin == nullptr || size == nullptr ||
      output == nullptr || output->allocation_type == kTfLiteDynamic ||
      input->type != output->type || begin->type != size->type ||
      (begin->type != kTfLiteInt32 && begin->type != kTfLiteInt64)) {
    return g_base_slice.invoke(context, node);
  }

  const size_t elem_bytes = GetElementBytes(output->type);
  if (elem_bytes == 0 || input->dims == nullptr || output->dims == nullptr ||
      begin->dims == nullptr || size->dims == nullptr) {
    return g_base_slice.invoke(context, node);
  }

  const int rank = input->dims->size;
  if (rank <= 0 || rank > internal::kMaxTensorRank ||
      output->dims->size != rank || begin->dims->size != 1 ||
      begin->dims->data[0] != rank || size->dims->size != 1 ||
      size->dims->data[0] != rank || begin->data.raw == nullptr ||
      size->data.raw == nullptr) {
    return g_base_slice.invoke(context, node);
  }

  int64_t in_dims[internal::kMaxTensorRank];
  int64_t begins[internal::kMaxTensorRank];
  int64_t sizes[internal::kMaxTensorRank];
  bool is_empty = false;

  if (begin->type == kTfLiteInt32) {
    for (int d = 0; d < rank; ++d) {
      begins[d] = begin->data.i32[d];
      sizes[d] = size->data.i32[d];
    }
  } else {
    for (int d = 0; d < rank; ++d) {
      begins[d] = begin->data.i64[d];
      sizes[d] = size->data.i64[d];
    }
  }

  for (int d = 0; d < rank; ++d) {
    const int64_t dim_val = input->dims->data[d];
    if (dim_val < 0 || begins[d] < 0 || begins[d] > dim_val) {
      return g_base_slice.invoke(context, node);
    }
    int64_t sz = sizes[d];
    if (sz < 0) {
      if (sz != -1) {
        return g_base_slice.invoke(context, node);
      }
      sz = dim_val - begins[d];
    } else if (sz > dim_val - begins[d]) {
      return g_base_slice.invoke(context, node);
    }
    if (sz != output->dims->data[d]) {
      return g_base_slice.invoke(context, node);
    }
    in_dims[d] = dim_val;
    sizes[d] = sz;
    if (sz == 0) {
      is_empty = true;
    }
  }

  if (is_empty) {
    return kTfLiteOk;
  }
  if (input->data.raw == nullptr || output->data.raw == nullptr) {
    return g_base_slice.invoke(context, node);
  }

  internal::FastSliceNdBytes(reinterpret_cast<const uint8_t*>(input->data.raw),
                             in_dims, begins, sizes, rank, elem_bytes,
                             reinterpret_cast<uint8_t*>(output->data.raw));
  return kTfLiteOk;
}

const tflite::MutableOpResolver* GetFastCpuOpResolver() {
  static const tflite::MutableOpResolver* const kResolver =
      []() -> const tflite::MutableOpResolver* {
    auto* resolver =
        new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
    if (const TfLiteRegistration* base_slice =
            resolver->FindOp(tflite::BuiltinOperator_SLICE, 1);
        base_slice != nullptr && base_slice->invoke != nullptr) {
      g_base_slice = *base_slice;
      TfLiteRegistration fast_slice = g_base_slice;
      fast_slice.invoke = &FastSliceEval;
      resolver->AddBuiltin(tflite::BuiltinOperator_SLICE, &fast_slice,
                           /*min_version=*/1, /*max_version=*/8);
    }
    return resolver;
  }();
  return kResolver;
}

}  // namespace
#endif  // defined(LITERT_LM_ENABLE_FAST_CPU_OPS)

absl::Status MaybeConfigureFastCpuOpResolver(litert::Options& options) {
#if defined(LITERT_LM_ENABLE_FAST_CPU_OPS)
  const tflite::MutableOpResolver* resolver = GetFastCpuOpResolver();
  LITERT_RETURN_IF_ERROR(options.AddBuildAction(
      [resolver](auto* /*runtime*/,
                 LiteRtOptions litert_options) -> LiteRtStatus {
        auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(litert_options);
        if (options_impl == nullptr) {
          return kLiteRtStatusErrorRuntimeFailure;
        }
        options_impl->op_resolver = resolver;
        return kLiteRtStatusOk;
      }));
#else
  (void)options;
#endif
  return absl::OkStatus();
}

}  // namespace litert::lm
