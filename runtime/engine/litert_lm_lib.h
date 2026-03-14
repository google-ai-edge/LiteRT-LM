// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_

#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/log_entry.h"  // from @com_google_absl
#include "absl/log/log_sink.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert {
namespace lm {

class FileLogSink : public absl::LogSink {
 public:
  explicit FileLogSink(const std::string& filename) {
    std::filesystem::path path(filename);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    file_.open(filename, std::ios_base::app);
  }

  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(&mutex_);
    file_ << entry.text_message_with_prefix_and_newline();
  }

 private:
  absl::Mutex mutex_;
  std::ofstream file_;
};

// Input data type for GPU Convolution and Fully Connected operations.
enum class ConvType {
  kAuto,   // Either float32/16 or int8 depending on the model.
  kFloat,  // Either float32 or float16 depending on the activation data type.
  kInt8,   // int8 quantized. Better latency with risk of less accuracy.
};

struct LiteRtLmSettings {
  std::string backend = "gpu";
  std::optional<std::string> vision_backend = std::nullopt;
  std::optional<std::string> audio_backend = std::nullopt;
  std::string sampler_backend = "";
  std::string model_path;
  bool load_model_from_descriptor = false;
  std::string input_prompt = "What is the tallest building in the world?";
  std::optional<std::string> expected_output = std::nullopt;
  std::optional<std::string> log_sink_file = std::nullopt;
  int max_num_tokens = 0;
  int max_output_tokens = -1;
  int max_num_images = 0;
  absl::LogSeverity min_log_level = absl::LogSeverity::kInfo;
  std::set<int> prefill_batch_sizes;
  int num_output_candidates = 1;
  bool benchmark = false;
  int benchmark_prefill_tokens = 0;
  int benchmark_decode_tokens = 0;
  bool async = true;
  bool report_peak_memory_footprint = false;
  std::optional<ActivationDataType> precision;
  bool multi_turns = false;
  int num_cpu_threads = 0;
  // Set external tensor mode false by default since it runs slightly faster
  // during decode as the layout changes optimized for GPU inference is done by
  // GPU, not by CPU.
  bool gpu_external_tensor_mode = false;
  bool configure_magic_numbers = true;
  bool verify_magic_numbers = false;
  bool clear_kv_cache_before_prefill = true;
  int num_logits_to_print_after_decode = 0;
  std::optional<std::string> score_target_text = std::nullopt;
  bool gpu_madvise_original_shared_tensors = true;
  bool disable_cache = false;
  std::string cache_dir = "";
  int prefill_chunk_size = -1;
  std::string preferred_device_substr = "";
  int num_threads_to_upload = -1;
  int num_threads_to_compile = -1;
  bool convert_weights_on_gpu = true;
  bool wait_for_weights_conversion_complete_in_benchmark = true;
  bool optimize_shader_compilation = true;
  bool share_constant_tensors = true;
  // If true, use Session instead of Conversation to run the inference.
  // Note that session does not add necessary prompt templates.
  bool use_session = false;
  int num_iterations = 1;
  std::string litert_dispatch_lib_dir = "";
  bool sampler_handles_input = true;
  ConvType conv_type = ConvType::kAuto;
  bool cache_compiled_shaders_only = false;
  std::string constraint_regex = "";
  bool use_submodel = false;
};

struct LitertLmMetrics {
  std::optional<BenchmarkInfo> benchmark_info;
  float peak_mem_mb = 0.0f;
  float peak_private_mb = 0.0f;
};

// Runs the LLM inference with the given settings.
// If metrics is not null, the metrics will be populated with the metrics from
// the inference. Results from each iteration is saved in the vector.
absl::Status RunLiteRtLm(const LiteRtLmSettings& settings,
                         std::vector<LitertLmMetrics>* metrics = nullptr);

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
