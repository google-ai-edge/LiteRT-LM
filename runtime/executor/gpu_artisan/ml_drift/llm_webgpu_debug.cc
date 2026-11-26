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

#include "runtime/executor/gpu_artisan/ml_drift/llm_webgpu_debug.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "runtime/util/status_macros.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

EM_JS_DEPS(emscripten_deps_llm_webgpu_fs, "$FS");
#endif  // __EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {
namespace llm_debug {
namespace {

// Print the first 15 entries of each tensor.
constexpr int kPrintLimit = 15;
constexpr float kF16Max = 65504.0f;

// Set once the prefill debug tensors have been dumped, so that only the first
// prefill of the process is captured (later prefills would otherwise overwrite
// the files).
std::atomic<bool> prefill_tensors_dumped{false};

absl::StatusOr<TensorFloat32> DownloadTensor(
    ExecutionEnvironment& env, InferenceContext& context,
    const GpuModelBuilder::TensorHandle& handle) {
  SpatialTensor* gpu_tensor = context.GetTensor(handle.id);
  if (gpu_tensor == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("No GPU tensor for value id ", handle.id));
  }
  TensorDescriptor desc;
  RETURN_IF_ERROR(gpu_tensor->ToDescriptor(env.device(), &desc));
  TensorFloat32 cpu_tensor;
  desc.DownloadData(&cpu_tensor);
  return cpu_tensor;
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
    if (i < kPrintLimit) {
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

// Writes `tensor` as raw float32 to the file `name` in the current directory
// (Emscripten's in-memory FS on the web) and, on the web, adds a download link
// for it to the page, for offline inspection/comparison.
absl::Status SaveTensorToDisk(const std::string& name,
                              const TensorFloat32& tensor) {
  FILE* tensor_file = std::fopen(name.c_str(), "wb");
  if (tensor_file == nullptr) {
    return absl::InternalError(absl::StrCat("Failed to open ", name));
  }
  const size_t num_written = std::fwrite(tensor.data.data(), sizeof(float),
                                         tensor.data.size(), tensor_file);
  std::fclose(tensor_file);
  if (num_written != tensor.data.size()) {
    return absl::InternalError(absl::StrCat("Failed to write ", name));
  }
  ABSL_LOG(INFO) << "Successfully wrote " << num_written << " floats to "
                 << name;

#ifdef __EMSCRIPTEN__
  static int index = 0;
  // clang-format off
  // (clang-format would split the JS `=>` token into `= >`.)
  EM_ASM({
    const name = UTF8ToString($0);
    const content = FS.readFile(name);
    const a = document.createElement('a');
    a.download = name;
    a.href = URL.createObjectURL(
        new Blob([content], {type: 'application/octet-stream'}));
    a.textContent = name;
    // a.style.display = 'none';
    const br = document.createElement('br');
    document.body.appendChild(br);
    document.body.appendChild(a);
    setTimeout(() => {
      // a.click();
    }, 500 * $1);
    // window.open(a.href, '_blank');
  }, name.c_str(), index);
  // clang-format on
  index++;
#endif  // __EMSCRIPTEN__
  return absl::OkStatus();
}

}  // namespace

absl::Status MaybeDumpPrefillTensors(ExecutionEnvironment& env,
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
  for (const auto& [name, handle] : debug_tensors) {
    ASSIGN_OR_RETURN(TensorFloat32 cpu_tensor,
                     DownloadTensor(env, context, handle));
    LogDebugTensor(name, cpu_tensor);
    RETURN_IF_ERROR(SaveTensorToDisk(name, cpu_tensor));
  }
  return absl::OkStatus();
}

}  // namespace llm_debug
}  // namespace webgpu
}  // namespace ml_drift
