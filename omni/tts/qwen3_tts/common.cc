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

#include "omni/tts/qwen3_tts/common.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/scoped_file.h"

namespace litert::omni::tts {

absl::Status CheckFileReadable(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> LoadFileOrAsset(
    const Qwen3StageOptions& options, absl::string_view filename) {
  if (options.model_resources != nullptr) {
    auto direct_or = options.model_resources->GetFile(filename);
    if (direct_or.ok()) {
      return std::string(*direct_or);
    }

    for (absl::string_view entry : options.model_resources->ListFiles()) {
      if (absl::EndsWith(entry, filename)) {
        ABSL_ASSIGN_OR_RETURN(absl::string_view buf,
                              options.model_resources->GetFile(entry));
        return std::string(buf);
      }
    }
    return absl::NotFoundError(absl::StrCat(
        "Could not find asset '", filename,
        "' in model_resources. Available files: ",
        absl::StrJoin(options.model_resources->ListFiles(), ", ")));
  }

  std::string path = absl::StrCat(options.model_dir, "/", filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  std::ifstream file(path, std::ios::binary | std::ios::ate);
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

absl::StatusOr<int> GetLanguageId(absl::string_view language) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, int>>
      kLanguageMap({
          {"chinese", 2055},  {"english", 2050},    {"german", 2053},
          {"italian", 2070},  {"portuguese", 2071}, {"spanish", 2054},
          {"japanese", 2064}, {"korean", 2064},     {"french", 2061},
          {"russian", 2069},
      });
  std::string lang_lower = std::string(language);
  absl::c_transform(lang_lower, lang_lower.begin(), ::tolower);
  auto it = kLanguageMap->find(lang_lower);
  if (it == kLanguageMap->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported language: ", language));
  }
  return it->second;
}

absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const Qwen3StageOptions& options,
    const std::string& model_filename, int num_threads, bool use_gpu) {
  LITERT_ASSIGN_OR_RETURN(auto comp_options, Options::Create());
  if (use_gpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            comp_options.GetGpuOptions());
    gpu_compilation_options.EnableInfiniteFloatCapping(true);
    gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
#if defined(__APPLE__)
    // TODO b/538727793 - See if this is actually needed for qwen3 tts. Gemma3
    // 1B is actually slower with metal argument buffers enabled.
    gpu_compilation_options.SetUseMetalArgumentBuffers(true);
    // gpu_compilation_options.EnableMetalResidencySet(true);
#endif  // !__APPLE__
    gpu_compilation_options.EnableConstantTensorSharing(true);
    gpu_compilation_options.SetMadviseOriginalSharedTensors(true);
    gpu_compilation_options.SetConvertWeightsOnGpu(true);
    gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
    comp_options.SetHardwareAccelerators(HwAccelerators::kGpu);
  } else {
    comp_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, comp_options.GetCpuOptions());
    ABSL_RETURN_IF_ERROR(lm::SetCpuOptions(cpu_options, num_threads));

    if (!options.cache_dir.empty()) {
      std::string cache_path = absl::StrCat(options.cache_dir, "/",
                                            model_filename, ".xnnpack_cache");
      absl::StatusOr<std::variant<std::string, std::shared_ptr<lm::ScopedFile>>>
          cache_variant(cache_path);
      ABSL_RETURN_IF_ERROR(
          lm::SetCpuCacheOptions(cache_variant, model_filename, cpu_options));
    }
  }
  if (options.model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::string model_buf,
                          LoadFileOrAsset(options, model_filename));
    BufferRef<uint8_t> buffer_ref(
        reinterpret_cast<const uint8_t*>(model_buf.data()), model_buf.size());
    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model,
        CompiledModel::Create(env, buffer_ref, comp_options));
    ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                                 use_gpu ? "GPU" : "CPU", " backend");
    return std::move(compiled_model);
  }
  std::string path = absl::StrCat(options.model_dir, "/", model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, path, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                               use_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

}  // namespace litert::omni::tts
