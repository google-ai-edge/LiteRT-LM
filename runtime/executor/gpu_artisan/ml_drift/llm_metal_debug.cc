// Copyright 2026 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/llm_metal_debug.h"

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/util/status_macros.h"

namespace ml_drift {
namespace metal {
namespace llm_debug {
namespace {

constexpr char kDebugTensorsDirEnvVar[] = "LLM_METAL_DEBUG_MLDRIFT_TENSORS_DIR";
constexpr char kDebugTensorsDefaultDir[] = "/tmp/llm_metal_debug_tensors";
constexpr int kDebugTensorsPrintLimit = 15;
constexpr float kF16Max = 65504.0f;

// Set once the prefill debug tensors have been dumped, so that only the first
// prefill of the process is captured (later prefills would otherwise overwrite
// the files).
std::atomic<bool> prefill_tensors_dumped{false};

std::string GetDebugTensorsDir() {
  const char* dir = std::getenv(kDebugTensorsDirEnvVar);
  return (dir != nullptr && dir[0] != '\0')
             ? std::string(dir)
             : std::string(kDebugTensorsDefaultDir);
}

// Logs the shape, min/max/mean, and first few values of `tensor`, and flags
// values that would overflow F16.
void LogDebugTensor(const std::string& name, const TensorFloat32& tensor) {
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  double sum_val = 0.0;
  bool has_f16_overflow = false;
  std::string vals;
  for (int i = 0; i < tensor.data.size(); ++i) {
    const float value = tensor.data[i];
    min_val = std::min(min_val, value);
    max_val = std::max(max_val, value);
    sum_val += value;
    has_f16_overflow |= (value > kF16Max || value < -kF16Max);
    if (i < kDebugTensorsPrintLimit) {
      absl::StrAppend(&vals, value, ", ");
    }
  }
  const double mean_val =
      tensor.data.empty() ? 0.0 : sum_val / tensor.data.size();
  ABSL_LOG(INFO) << "Debugging tensor " << name << " of shape "
                 << ToString(tensor.shape) << "\nmin: " << min_val
                 << ", max: " << max_val << ", mean: " << mean_val
                 << "\nwith values:\n"
                 << vals;
  if (has_f16_overflow) {
    ABSL_LOG(ERROR) << "OVERFLOW FOUND FOR " << name;
  }
}

// Writes `num_bytes` bytes from `data` to `<dir>/<name>`, creating `dir` (but
// not its parents) if needed.
absl::Status SaveRawBytesToDisk(const std::string& dir, const std::string& name,
                                const void* data, size_t num_bytes) {
  if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
    return absl::InternalError(absl::StrCat("Failed to create directory ", dir,
                                            ": ", std::strerror(errno)));
  }
  const std::string path = absl::StrCat(dir, "/", name);
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return absl::InternalError(
        absl::StrCat("Failed to open ", path, ": ", std::strerror(errno)));
  }
  const size_t written = std::fwrite(data, 1, num_bytes, file);
  std::fclose(file);
  if (written != num_bytes) {
    return absl::InternalError(absl::StrCat("Failed to write ", path,
                                            ": wrote ", written, " of ",
                                            num_bytes, " bytes."));
  }
  ABSL_LOG(INFO) << "Wrote " << written << " bytes to " << path;
  return absl::OkStatus();
}

absl::Status SaveDebugTensorToDisk(const std::string& dir,
                                   const std::string& name,
                                   const TensorFloat32& tensor) {
  return SaveRawBytesToDisk(dir, name, tensor.data.data(),
                            tensor.data.size() * sizeof(float));
}

// Waits for all work on `command_queue` to finish, then downloads, logs, and
// saves every tensor in `debug_tensors` from `context` into `dir`.
absl::Status DumpDebugTensors(id<MTLCommandQueue> command_queue,
                              id<MTLDevice> device, InferenceContext& context,
                              const DebugTensors& debug_tensors,
                              const std::string& dir) {
  // Command buffers on a queue execute in order, so waiting on an empty one
  // guarantees that all previously committed work has completed.
  id<MTLCommandBuffer> sync_command_buffer = [command_queue commandBuffer];
  [sync_command_buffer commit];
  [sync_command_buffer waitUntilCompleted];

  for (const auto& [name, handle] : debug_tensors) {
    const MetalSpatialTensor* gpu_tensor = context.GetTensor(handle.id);
    RET_CHECK(gpu_tensor != nullptr) << "Missing debug tensor " << name;
    TensorDescriptor desc;
    RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device));
    TensorFloat32 cpu_tensor;
    desc.DownloadData(&cpu_tensor);
    LogDebugTensor(name, cpu_tensor);
    RETURN_IF_ERROR(SaveDebugTensorToDisk(dir, name, cpu_tensor));
  }
  return absl::OkStatus();
}

// Decode-step debugging: if LLM_METAL_DEBUG_MLDRIFT_DECODE_POS is set to a
// sequence position, the debug tensors of the single-token model are dumped
// right after the decode step at that position.
constexpr char kDebugDecodePosEnvVar[] = "LLM_METAL_DEBUG_MLDRIFT_DECODE_POS";

// Returns the requested decode position, or -1 if unset or invalid.
int GetDebugDecodePos() {
  const char* pos = std::getenv(kDebugDecodePosEnvVar);
  int value = -1;
  if (pos == nullptr || !absl::SimpleAtoi(pos, &value)) {
    return -1;
  }
  return value;
}

// Mirrors IsGlobalLayer() in transformer.cc.
bool DebugIsGlobalLayer(const TransformerConfig& config, int layer) {
  const int offset =
      config.global_layer_offset.value_or(config.num_local_layers_per_global);
  return layer % (config.num_local_layers_per_global + 1) == offset;
}

// Saves the raw (native dtype) contents of every KV cache into `dir` as
// `k_<i>.raw` / `v_<i>.raw`, plus a `meta.txt` with the config values needed to
// decode them. The caches use the custom layouts written by
// TransformerBuilder::AddValuesToCache.
absl::Status DumpKvCaches(
    id<MTLDevice> device, const TransformerConfig& config,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& key_caches,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& value_caches,
    int position, const std::string& dir) {
  RET_CHECK_EQ(key_caches.size(), value_caches.size());
  std::string meta = absl::StrCat(
      "position ", position, "\n",                             //
      "number_of_heads ", config.number_of_heads, "\n",        //
      "number_of_kv_heads ", config.number_of_kv_heads, "\n",  //
      "global_number_of_kv_heads ",
      config.global_number_of_kv_heads.value_or(-1), "\n",  //
      "head_dimension ", config.head_dimension, "\n",       //
      "global_head_dimension ", config.global_head_dimension.value_or(-1),
      "\n",                                    //
      "cache_size ", config.cache_size, "\n",  //
      "num_local_layers_per_global ", config.num_local_layers_per_global,
      "\n",  //
      "global_layer_offset ", config.global_layer_offset.value_or(-1), "\n",
      "sliding_window_size ", config.sliding_window_size.value_or(-1), "\n",
      "local_ring_cache_size ", config.local_ring_cache_size.value_or(-1),
      "\n",  //
      "kv_cache_shared_start_layer ",
      config.kv_cache_shared_start_layer.value_or(-1), "\n",     //
      "attn_soft_cap_value ", config.attn_soft_cap_value, "\n",  //
      "stack_size ", config.stack_size, "\n");
  for (int i = 0; i < key_caches.size(); ++i) {
    const bool is_global = DebugIsGlobalLayer(config, i);
    const int cache_size = TransformerBuilder::GetCacheSize(config, is_global);
    auto save = [&](const char* kind,
                    const MetalSpatialTensor* tensor) -> absl::Status {
      RET_CHECK(tensor != nullptr) << "Missing " << kind << " cache " << i;
      TensorDescriptor desc;
      RETURN_IF_ERROR(tensor->ToDescriptor(&desc, device));
      const absl::Span<const uint8_t> data = desc.GetData();
      absl::StrAppend(&meta, "cache ", kind, " ", i, " dtype ",
                      ToString(desc.GetDataType()), " bytes ", data.size(),
                      " is_global ", is_global ? 1 : 0, " cache_size ",
                      cache_size, "\n");
      return SaveRawBytesToDisk(dir, absl::StrCat(kind, "_", i, ".raw"),
                                data.data(), data.size());
    };
    RETURN_IF_ERROR(save("k", key_caches[i].get()));
    RETURN_IF_ERROR(save("v", value_caches[i].get()));
  }
  return SaveRawBytesToDisk(dir, "meta.txt", meta.data(), meta.size());
}

// Logits debugging: the full-vocabulary logits fed into TopK (after the final
// softcap and constraint mask) are saved for the first N sampling steps, where
// N defaults to kDebugLogitsDefaultSteps and can be overridden with the
// LLM_METAL_DEBUG_MLDRIFT_LOGITS_STEPS environment variable.
constexpr char kDebugLogitsStepsEnvVar[] =
    "LLM_METAL_DEBUG_MLDRIFT_LOGITS_STEPS";
constexpr int kDebugLogitsDefaultSteps = 64;
constexpr int kDebugLogitsLogTopK = 5;

// Tensor ids of the extra post-processing outputs used for logits debugging.
GpuModelBuilder::ValueId pre_softmax_id = 0;
GpuModelBuilder::ValueId topk_tokens_id = 0;
bool post_processing_ids_valid = false;
std::atomic<int> logits_steps_dumped{0};

int GetDebugLogitsSteps() {
  const char* steps = std::getenv(kDebugLogitsStepsEnvVar);
  int value = 0;
  if (steps != nullptr && absl::SimpleAtoi(steps, &value)) {
    return value;
  }
  return kDebugLogitsDefaultSteps;
}

}  // namespace

absl::Status MaybeDumpPrefillTensors(id<MTLCommandQueue> command_queue,
                                     id<MTLDevice> device,
                                     InferenceContext& context,
                                     const DebugTensors& debug_tensors,
                                     int batch_index, int num_batches) {
  // Dump the debug tensors for the first batch of the first prefill in this
  // process only, so that the files (and any overflow errors) reflect the
  // earliest point of failure.
  // NOTE: For models with shared KV cache layers, earlier batches skip the
  // optional shared-KV layers, leaving their debug tensors unwritten. To debug
  // those layers, change `batch_index != 0` to
  // `batch_index != num_batches - 1` to dump the last batch of the first
  // prefill instead, which runs all layers.
  if (batch_index != 0 || debug_tensors.empty() ||
      prefill_tensors_dumped.exchange(true)) {
    return absl::OkStatus();
  }
  return DumpDebugTensors(command_queue, device, context, debug_tensors,
                          GetDebugTensorsDir());
}

void RegisterPostProcessingOutputs(const PostProcessingOutputs& outputs) {
  pre_softmax_id = outputs.pre_softmax.id;
  topk_tokens_id = outputs.topk_tokens.id;
  post_processing_ids_valid = true;
}

absl::Status MaybeDumpPostProcessingLogits(id<MTLCommandBuffer> command_buffer,
                                           id<MTLDevice> device,
                                           InferenceContext& post_processing,
                                           int position) {
  if (!post_processing_ids_valid) {
    return absl::OkStatus();
  }
  const int step = logits_steps_dumped.fetch_add(1);
  if (step >= GetDebugLogitsSteps()) {
    return absl::OkStatus();
  }
  [command_buffer waitUntilCompleted];

  TensorFloat32 logits;
  {
    const MetalSpatialTensor* gpu_tensor =
        post_processing.GetTensor(pre_softmax_id);
    RET_CHECK(gpu_tensor != nullptr) << "Missing pre-softmax logits tensor";
    TensorDescriptor desc;
    RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device));
    desc.DownloadData(&logits);
  }
  Tensor<BHWC, DataType::kInt32> topk_tokens;
  {
    const MetalSpatialTensor* gpu_tensor =
        post_processing.GetTensor(topk_tokens_id);
    RET_CHECK(gpu_tensor != nullptr) << "Missing TopK tokens tensor";
    TensorDescriptor desc;
    RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device));
    desc.DownloadData(&topk_tokens);
  }

  // Rank batch 0's logits on the host, which also exposes exact ties (e.g. two
  // candidates rounding to the same F16 value) that TopK had to break.
  const int vocab_size = logits.shape.c;
  const int log_k = std::min(kDebugLogitsLogTopK, vocab_size);
  std::vector<int> order(vocab_size);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(
      order.begin(), order.begin() + log_k, order.end(),
      [&logits](int a, int b) { return logits.data[a] > logits.data[b]; });
  std::string top;
  for (int i = 0; i < log_k; ++i) {
    char entry[64];
    std::snprintf(entry, sizeof(entry), "%d:%.6f ", order[i],
                  logits.data[order[i]]);
    absl::StrAppend(&top, entry);
  }
  const bool top1_tie =
      log_k >= 2 && logits.data[order[0]] == logits.data[order[1]];
  // Printed directly to stderr rather than through ABSL_LOG(INFO), because
  // LiteRT-LM tools raise the absl minimum log level (e.g. to kError), which
  // would silently drop this output.
  std::fprintf(stderr,
               "Logits step %d pos %d: GPU top-1 token %d; host top-%d "
               "(token:logit) %s%s\n",
               step, position,
               topk_tokens.data.empty() ? -1 : topk_tokens.data[0], log_k,
               top.c_str(), top1_tie ? "[TIE at top-1]" : "");
  std::fflush(stderr);

  const std::string name =
      absl::StrCat("logits_step", absl::Dec(step, absl::kZeroPad3), "_pos",
                   absl::Dec(position, absl::kZeroPad5));
  return SaveDebugTensorToDisk(GetDebugTensorsDir(), name, logits);
}

absl::Status MaybeDumpDecodeStepTensors(
    id<MTLCommandQueue> command_queue, id<MTLDevice> device,
    InferenceContext& context, const DebugTensors& debug_tensors,
    const TransformerConfig& config,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& key_caches,
    const std::vector<std::unique_ptr<MetalSpatialTensor>>& value_caches,
    int position) {
  if (debug_tensors.empty() || position != GetDebugDecodePos()) {
    return absl::OkStatus();
  }
  const std::string base_dir = GetDebugTensorsDir();
  if (mkdir(base_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    return absl::InternalError(absl::StrCat(
        "Failed to create directory ", base_dir, ": ", std::strerror(errno)));
  }
  const std::string dir = absl::StrCat(base_dir, "/decode_pos", position);
  std::fprintf(stderr,
               "Dumping %zu debug tensors and %zu KV cache pairs for decode "
               "position %d to %s\n",
               debug_tensors.size(), key_caches.size(), position, dir.c_str());
  std::fflush(stderr);
  // DumpDebugTensors waits for all queued GPU work, so the caches are current.
  RETURN_IF_ERROR(
      DumpDebugTensors(command_queue, device, context, debug_tensors, dir));
  // The KV caches go in a subdirectory so compare_tensors.py, which only reads
  // top-level files, does not treat them as float32 tensors.
  return DumpKvCaches(device, config, key_caches, value_caches, position,
                      absl::StrCat(dir, "/kv"));
}

}  // namespace llm_debug
}  // namespace metal
}  // namespace ml_drift
