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

#include "omni/base/model_utils.h"

#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/scoped_file.h"

namespace litert::omni {

absl::Status CheckFileReadable(absl::string_view path) {
  std::string path_str(path);
  std::ifstream in(path_str);
  if (!in.good()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> LoadFile(absl::string_view path) {
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  std::string path_str(path);
  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string content(size, '\0');
  if (size > 0 && !file.read(content.data(), size)) {
    return absl::InternalError(absl::StrCat("Failed to read file: ", path));
  }
  return content;
}

absl::StatusOr<std::string> LoadFile(absl::string_view model_dir,
                                     absl::string_view filename) {
  return LoadFile(absl::StrCat(model_dir, "/", filename));
}

absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  LITERT_ASSIGN_OR_RETURN(auto comp_options, Options::Create());
  bool target_gpu = options.backend == lm::Backend::GPU;

  if (target_gpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            comp_options.GetGpuOptions());
    gpu_compilation_options.EnableInfiniteFloatCapping(true);
    gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
#if defined(__APPLE__)
    // Metal argument buffers setting.
    gpu_compilation_options.SetUseMetalArgumentBuffers(true);
#endif  // !__APPLE__
    gpu_compilation_options.EnableConstantTensorSharing(true);
    gpu_compilation_options.SetMadviseOriginalSharedTensors(true);
    gpu_compilation_options.SetConvertWeightsOnGpu(true);
    gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
    comp_options.SetHardwareAccelerators(HwAccelerators::kGpu);
  } else {
    comp_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, comp_options.GetCpuOptions());
    ABSL_RETURN_IF_ERROR(lm::SetCpuOptions(cpu_options, options.num_threads));

    if (!options.cache_dir.empty()) {
      std::string cache_path = absl::StrCat(options.cache_dir, "/",
                                            model_filename, ".xnnpack_cache");
      absl::StatusOr<std::variant<std::string, std::shared_ptr<lm::ScopedFile>>>
          cache_variant(cache_path);
      ABSL_RETURN_IF_ERROR(
          lm::SetCpuCacheOptions(cache_variant, model_filename, cpu_options));
    }
  }

  std::string path = absl::StrCat(options.model_dir, "/", model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, path, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                               target_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

absl::StatusOr<size_t> ResolveInputIndex(const CompiledModel& model,
                                         absl::string_view tensor_name) {
  auto names_res = model.GetSignatureInputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == tensor_name) {
        return i;
      }
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "Input tensor '", tensor_name, "' not found in model signatures"));
}

absl::StatusOr<size_t> ResolveOutputIndex(const CompiledModel& model,
                                          absl::string_view tensor_name) {
  auto names_res = model.GetSignatureOutputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == tensor_name) {
        return i;
      }
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "Output tensor '", tensor_name, "' not found in model signatures"));
}

}  // namespace litert::omni
