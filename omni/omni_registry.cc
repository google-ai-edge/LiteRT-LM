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

#include "omni/omni_registry.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "omni/base/model_cache.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::omni {
namespace {

constexpr int kDefaultCpuThreads = 4;

bool FileExists(absl::string_view path) {
  if (path.empty()) {
    return false;
  }
  std::ifstream f(std::string(path).c_str());
  return f.good();
}

std::string NormalizeForMatching(absl::string_view str) {
  std::string result;
  result.reserve(str.size());
  for (char c : str) {
    if (c != '-' && c != '_') {
      result.push_back(absl::ascii_tolower(c));
    }
  }
  return result;
}

absl::StatusOr<std::shared_ptr<lm::ModelResources>> OpenLitertLmResources(
    absl::string_view model_path) {
  if (model_path.empty()) {
    return absl::InvalidArgumentError("model_path must not be empty.");
  }
  const std::string resolved_path = ResolveLitertLmPath(model_path);
  if (resolved_path.empty() || !FileExists(resolved_path)) {
    return absl::NotFoundError(
        absl::StrCat("No .litertlm file found at: ", model_path));
  }
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(resolved_path));
  ABSL_ASSIGN_OR_RETURN(auto loader,
                        lm::LitertLmLoader::Create(std::move(scoped_file)));
  ABSL_ASSIGN_OR_RETURN(auto resources,
                        lm::ModelResourcesLitertLm::Create(std::move(loader)));
  return std::shared_ptr<lm::ModelResources>(std::move(resources));
}

OmniEngine::Settings ResolveEngineSettings(
    absl::string_view model_name, const OmniEngine::Settings& settings) {
  OmniEngine::Settings resolved = settings;
  // `kNoCacheDir` only disables compiled-model caches; model files are still
  // resolved and downloaded under the default directory.
  const std::string effective_cache_dir =
      (resolved.cache_dir.empty() || resolved.cache_dir == kNoCacheDir)
          ? "/tmp/omni_models"
          : resolved.cache_dir;
  if (resolved.model_path.empty()) {
    if (!resolved.model_remote_url.empty()) {
      std::string ext =
          std::filesystem::path(resolved.model_remote_url).extension().string();
      if (ext.empty()) {
        ext = ".tflite";
      }
      resolved.model_path = (std::filesystem::path(effective_cache_dir) /
                             absl::StrCat(model_name, ext))
                                .string();
    } else {
      resolved.model_path = (std::filesystem::path(effective_cache_dir) /
                             absl::StrCat(model_name, ".litertlm"))
                                .string();
    }
  }
  if (resolved.tokenizer_path.empty() &&
      !resolved.tokenizer_remote_url.empty()) {
    resolved.tokenizer_path = (std::filesystem::path(effective_cache_dir) /
                               absl::StrCat(model_name, "_tokenizer.json"))
                                  .string();
  }
  return resolved;
}

template <typename MetadataT>
void Merge(MetadataT& dst, const MetadataT& src) {
  dst.MergeFrom(src);
}

template <>
void Merge<lm::proto::Speech2TextMetadata>(
    lm::proto::Speech2TextMetadata& dst,
    const lm::proto::Speech2TextMetadata& src) {
  if (src.decode_start_token_ids_size() > 0) {
    dst.clear_decode_start_token_ids();
  }
  dst.MergeFrom(src);
}

template <typename MetadataT>
const MessageLite* absl_nullable MergeMetadataProto(
    absl::StatusOr<const MetadataT*> embedded,
    const MessageLite* absl_nullable caller_metadata,
    std::unique_ptr<MessageLite>& merged_metadata) {
  if (!embedded.ok() || *embedded == nullptr) {
    return caller_metadata;
  }
  auto merged = std::make_unique<MetadataT>(**embedded);
  if (caller_metadata != nullptr) {
    Merge(*merged, *static_cast<const MetadataT*>(caller_metadata));
  }
  merged_metadata = std::move(merged);
  return merged_metadata.get();
}

const MessageLite* absl_nullable MergeMetadata(
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view metadata_proto_type_id,
    const MessageLite* absl_nullable caller_metadata,
    std::unique_ptr<MessageLite>& merged_metadata) {
  if (lm_resources == nullptr) {
    return caller_metadata;
  }
  if (metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata") {
    return MergeMetadataProto(lm_resources->GetSpeech2TextMetadata(),
                              caller_metadata, merged_metadata);
  }
  if (metadata_proto_type_id == "litert.lm.proto.Text2SpeechMetadata") {
    return MergeMetadataProto(lm_resources->GetText2SpeechMetadata(),
                              caller_metadata, merged_metadata);
  }
  if (metadata_proto_type_id == "litert.lm.proto.Text2ImageMetadata") {
    return MergeMetadataProto(lm_resources->GetText2ImageMetadata(),
                              caller_metadata, merged_metadata);
  }
  return caller_metadata;
}

// Note: every registered session factory compiles at most one `CompiledModel`
// from `options`, so a single per-model cache key is sufficient. A factory that
// compiles several models from the same `options` must use distinct keys.
absl::Status ConfigureLiteRtOptions(absl::string_view model_name,
                                    const OmniEngine::Settings& settings,
                                    ::litert::Options& options) {
  uint32_t accelerators = static_cast<uint32_t>(::litert::HwAccelerators::kCpu);
  ABSL_ASSIGN_OR_RETURN(
      const ModelCacheConfig cache,
      ResolveModelCache(model_name, settings.model_path, settings.cache_dir));
  LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                          options.GetOptions<::litert::CpuOptions>());
  const int num_threads =
      settings.num_threads > 0 ? settings.num_threads : kDefaultCpuThreads;
  LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(num_threads));
  auto set_flags = cpu_options.GetXNNPackFlags();
  // Fall back to the delegate defaults (QS8|QU8, ...) when flags were never
  // set.
  uint32_t xnn_flags = (set_flags.HasValue() && *set_flags != 0)
                           ? *set_flags
                           : TfLiteXNNPackDelegateOptionsDefault().flags;
  xnn_flags |= TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED |
               TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  LITERT_RETURN_IF_ERROR(cpu_options.SetXNNPackFlags(xnn_flags));
  ABSL_RETURN_IF_ERROR(ConfigureCpuModelCache(model_name, cache, cpu_options));
  if (settings.backend == OmniEngine::Settings::Backend::kGpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kGpu);
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            options.GetOptions<::litert::GpuOptions>());
    gpu_options.SetPrecision(settings.force_fp32_on_gpu
                                 ? ::litert::GpuOptions::Precision::kFp32
                                 : ::litert::GpuOptions::Precision::kFp16);
    gpu_options.EnableInfiniteFloatCapping(true);
    gpu_options.EnableConstantTensorSharing(true);
    gpu_options.EnableAllowSrcQuantizedFcConvOps(true);
#if defined(__ANDROID__)
    gpu_options.SetPreferTextureWeights(true);
#endif  // defined(__ANDROID__)
    gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
    LrtSetGpuOptionsUseIrModel(gpu_options.Get(), false);
    ABSL_RETURN_IF_ERROR(
        ConfigureGpuModelCache(model_name, cache, gpu_options));
  } else if (settings.backend == OmniEngine::Settings::Backend::kNpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kNpu);
    LITERT_ASSIGN_OR_RETURN(
        auto& qnn_options,
        options.GetOptions<::litert::qualcomm::QualcommOptions>());
    qnn_options.SetHtpPerformanceMode(::litert::qualcomm::QualcommOptions::
                                          HtpPerformanceMode::kHighPerformance);
  }
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(
      static_cast<::litert::HwAccelerators>(accelerators)));
  return absl::OkStatus();
}

}  // namespace

OmniRegistry& OmniRegistry::Get() {
  static absl::NoDestructor<OmniRegistry> kInstance;
  return *kInstance;
}

absl::StatusOr<std::string> OmniRegistry::ResolveSessionConfigProtoTypeId(
    absl::string_view session_config_type) const {
  if (session_config_type.empty()) {
    return absl::InvalidArgumentError("session_config_type must not be empty.");
  }
  if (absl::StrContains(session_config_type, '.')) {
    return std::string(session_config_type);
  }
  return absl::StrCat("litert.lm.proto.", session_config_type, "SessionConfig");
}

absl::StatusOr<std::string> OmniRegistry::ResolveMetadataProtoTypeId(
    absl::string_view session_config_type,
    absl::string_view metadata_type) const {
  if (session_config_type.empty()) {
    return absl::InvalidArgumentError("session_config_type must not be empty.");
  }
  if (metadata_type.empty()) {
    if (absl::StrContains(session_config_type, '.')) {
      return absl::InvalidArgumentError(
          absl::StrCat("metadata_type must be explicitly provided when "
                       "session_config_type is a full proto type name: ",
                       session_config_type));
    }
    return absl::StrCat("litert.lm.proto.", session_config_type, "Metadata");
  }
  if (absl::StrContains(metadata_type, '.')) {
    return std::string(metadata_type);
  }
  return absl::StrCat("litert.lm.proto.", metadata_type, "Metadata");
}

absl::Status OmniRegistry::RegisterSessionFactory(RegistrySpec spec) {
  ABSL_ASSIGN_OR_RETURN(
      std::string resolved_session_id,
      ResolveSessionConfigProtoTypeId(spec.session_config_type_id));
  ABSL_ASSIGN_OR_RETURN(
      std::string resolved_metadata_id,
      ResolveMetadataProtoTypeId(spec.session_config_type_id,
                                 spec.metadata_type_id.value_or("")));

  if (spec.model_names.empty()) {
    return absl::InvalidArgumentError(
        "RegistrySpec.model_names must not be empty");
  }
  for (const auto& model_name : spec.model_names) {
    if (model_name.empty()) {
      return absl::InvalidArgumentError(
          "RegistrySpec.model_names must not contain empty model_name");
    }
  }
  if (!spec.get_session_factory) {
    return absl::InvalidArgumentError(
        absl::StrCat("RegistrySpec '", spec.model_names.front(),
                     "' get_session_factory must not be null"));
  }

  FactoryKey key{
      .session_config_proto_type_id = std::move(resolved_session_id),
      .metadata_proto_type_id = std::move(resolved_metadata_id),
  };
  absl::MutexLock lock(mutex_);
  auto shared_factory = std::make_shared<SessionFactoryCreator>(
      std::move(spec.get_session_factory));
  factories_[key] = shared_factory;
  const std::string canonical_model_name = spec.model_names.front();
  const std::string model_remote_url = spec.model_remote_url.value_or("");
  for (auto& model_name : spec.model_names) {
    ModelEntry entry;
    entry.factory_key = key;
    entry.canonical_model_name = canonical_model_name;
    entry.model_remote_url = model_remote_url;
    entry.factory_creator = shared_factory;
    models_[std::move(model_name)] = std::move(entry);
  }
  return absl::OkStatus();
}

bool OmniRegistry::UnregisterSessionFactory(
    absl::string_view session_config_type, absl::string_view metadata_type) {
  auto resolved_session_id =
      ResolveSessionConfigProtoTypeId(session_config_type);
  auto resolved_metadata_id =
      ResolveMetadataProtoTypeId(session_config_type, metadata_type);
  if (!resolved_session_id.ok() || !resolved_metadata_id.ok()) {
    return false;
  }
  FactoryKey key{
      .session_config_proto_type_id = *std::move(resolved_session_id),
      .metadata_proto_type_id = *std::move(resolved_metadata_id),
  };
  absl::MutexLock lock(mutex_);
  bool removed = factories_.erase(key) > 0;
  for (auto it = models_.begin(); it != models_.end();) {
    if (it->second.factory_key == key) {
      models_.erase(it++);
      removed = true;
    } else {
      ++it;
    }
  }
  return removed;
}

bool OmniRegistry::UnregisterModel(absl::string_view model_name) {
  absl::MutexLock lock(mutex_);
  return models_.erase(model_name) > 0;
}

bool OmniRegistry::IsModelRegistered(absl::string_view model_name) const {
  absl::MutexLock lock(mutex_);
  return models_.contains(model_name);
}

std::vector<std::string> OmniRegistry::GetRegisteredModels() const {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> names;
  names.reserve(models_.size());
  for (const auto& [name, _] : models_) {
    names.push_back(name);
  }
  return names;
}

std::string OmniRegistry::FindLongestContainedModelName(
    absl::string_view normalized_target) const {
  std::string best;
  size_t best_len = 0;
  for (const auto& [name, _] : models_) {
    const std::string normalized_name = NormalizeForMatching(name);
    if (normalized_name.size() > best_len &&
        absl::StrContains(normalized_target, normalized_name)) {
      best = name;
      best_len = normalized_name.size();
    }
  }
  return best;
}

absl::StatusOr<std::string> OmniRegistry::FindModelNameForPath(
    absl::string_view model_path) const {
  const std::filesystem::path path{std::string(model_path)};
  const std::string normalized_stem = NormalizeForMatching(
      path.has_stem() ? path.stem().string() : path.string());
  absl::MutexLock lock(mutex_);
  std::string best = FindLongestContainedModelName(normalized_stem);
  if (best.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No registered Omni model matches path: ", model_path));
  }
  return best;
}

absl::StatusOr<OmniRegistry::RegisteredModel> OmniRegistry::GetRegisteredModel(
    absl::string_view model_name) const {
  absl::MutexLock lock(mutex_);
  auto it = models_.find(model_name);
  if (it == models_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unsupported or unknown Omni model: ", model_name));
  }
  RegisteredModel result;
  result.model_name = it->first;
  result.canonical_model_name = it->second.canonical_model_name;
  result.model_remote_url = it->second.model_remote_url;
  result.session_config_proto_type_id =
      it->second.factory_key.session_config_proto_type_id;
  result.metadata_proto_type_id = it->second.factory_key.metadata_proto_type_id;
  return result;
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
OmniRegistry::CreateSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata) const {
  std::string resolved_model_name(model_name);
  std::string default_model_remote_url;
  std::shared_ptr<SessionFactoryCreator> creator;
  FactoryKey factory_key;
  bool was_registered_name = false;

  {
    absl::MutexLock lock(mutex_);
    auto model_it = models_.find(resolved_model_name);
    if (model_it == models_.end()) {
      const std::string lower_name = absl::AsciiStrToLower(resolved_model_name);
      model_it = models_.find(lower_name);
      if (model_it != models_.end()) {
        resolved_model_name = lower_name;
      }
    }
    was_registered_name = (model_it != models_.end());

    if (model_it == models_.end() &&
        (!settings.model_path.empty() || absl::StrContains(model_name, '/'))) {
      const std::string matched_name =
          FindLongestContainedModelName(NormalizeForMatching(
              resolved_model_name.empty() ? settings.model_path
                                          : resolved_model_name));
      if (!matched_name.empty()) {
        model_it = models_.find(matched_name);
        resolved_model_name = matched_name;
      }
    }

    if (model_it == models_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Unsupported or unknown Omni model: ", model_name));
    }
    factory_key = model_it->second.factory_key;
    default_model_remote_url = model_it->second.model_remote_url;
    creator = model_it->second.factory_creator;
    if (creator == nullptr) {
      auto factory_it = factories_.find(factory_key);
      if (factory_it == factories_.end()) {
        return absl::NotFoundError(
            absl::StrCat("No registered SessionFactoryCreator for model: ",
                         resolved_model_name));
      }
      creator = factory_it->second;
    }
  }

  if (session_config != nullptr &&
      session_config->GetTypeName() !=
          factory_key.session_config_proto_type_id) {
    return absl::InvalidArgumentError(
        absl::StrCat("session_config type '", session_config->GetTypeName(),
                     "' does not match registered session_config_type '",
                     factory_key.session_config_proto_type_id, "' for model '",
                     resolved_model_name, "'."));
  }
  if (metadata != nullptr &&
      metadata->GetTypeName() != factory_key.metadata_proto_type_id) {
    return absl::InvalidArgumentError(
        absl::StrCat("metadata type '", metadata->GetTypeName(),
                     "' does not match registered metadata_type '",
                     factory_key.metadata_proto_type_id, "' for model '",
                     resolved_model_name, "'."));
  }

  OmniEngine::Settings settings_with_defaults = settings;
  if (!was_registered_name && settings_with_defaults.model_path.empty() &&
      !model_name.empty()) {
    settings_with_defaults.model_path = std::string(model_name);
  }
  if (settings_with_defaults.model_remote_url.empty()) {
    settings_with_defaults.model_remote_url =
        std::move(default_model_remote_url);
  }
  OmniEngine::Settings resolved_settings =
      ResolveEngineSettings(resolved_model_name, settings_with_defaults);

  std::shared_ptr<lm::ModelResources> lm_resources;
  auto lm_resources_or = OpenLitertLmResources(resolved_settings.model_path);
  if (lm_resources_or.ok()) {
    lm_resources = *std::move(lm_resources_or);
  } else if (!absl::IsNotFound(lm_resources_or.status())) {
    return lm_resources_or.status();
  }

  std::unique_ptr<MessageLite> merged_metadata;
  const MessageLite* effective_metadata =
      MergeMetadata(lm_resources.get(), factory_key.metadata_proto_type_id,
                    metadata, merged_metadata);

  LITERT_ASSIGN_OR_RETURN(::litert::Environment env,
                          ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(::litert::Options options,
                          ::litert::Options::Create());
  ABSL_RETURN_IF_ERROR(
      ConfigureLiteRtOptions(resolved_model_name, resolved_settings, options));

  return (*creator)(resolved_model_name, resolved_settings, session_config,
                    effective_metadata, env, options, std::move(lm_resources));
}

}  // namespace litert::omni
