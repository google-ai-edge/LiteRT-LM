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
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/asr_model_type.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"

namespace litert::omni {
namespace {

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
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(model_path));
  ABSL_ASSIGN_OR_RETURN(auto loader,
                        lm::LitertLmLoader::Create(std::move(scoped_file)));
  ABSL_ASSIGN_OR_RETURN(auto resources,
                        lm::ModelResourcesLitertLm::Create(std::move(loader)));
  return std::shared_ptr<lm::ModelResources>(std::move(resources));
}

OmniEngine::Settings ResolveEngineSettings(
    absl::string_view model_name, const OmniEngine::Settings& settings) {
  OmniEngine::Settings resolved = settings;
  const std::string effective_cache_dir =
      resolved.cache_dir.empty() ? "/tmp/asr_models" : resolved.cache_dir;
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

absl::Status ConfigureLiteRtOptions(const OmniEngine::Settings& settings,
                                    ::litert::Options& options) {
  uint32_t accelerators = static_cast<uint32_t>(::litert::HwAccelerators::kCpu);
  if (settings.num_threads > 0) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            options.GetOptions<::litert::CpuOptions>());
    cpu_options.SetNumThreads(settings.num_threads);
  }
  if (settings.backend == OmniEngine::Settings::Backend::kGpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kGpu);
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            options.GetOptions<::litert::GpuOptions>());
    gpu_options.SetPrecision(::litert::GpuOptions::Precision::kFp32);
    gpu_options.EnableConstantTensorSharing(true);
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
  std::shared_ptr<lm::ModelResources> lm_resources;
  std::string candidate_path = settings.model_path.empty()
                                   ? std::string(model_name)
                                   : settings.model_path;
  if (absl::EndsWith(candidate_path, ".litertlm") &&
      FileExists(candidate_path)) {
    ABSL_ASSIGN_OR_RETURN(lm_resources, OpenLitertLmResources(candidate_path));
  }

  std::string resolved_model_name(model_name);
  std::string default_model_remote_url;
  std::shared_ptr<SessionFactoryCreator> creator;
  FactoryKey factory_key;

  {
    absl::MutexLock lock(mutex_);
    auto model_it = models_.find(resolved_model_name);
    if (model_it == models_.end() &&
        (lm_resources != nullptr || metadata != nullptr ||
         !settings.model_path.empty())) {
      // 1. Try matching by embedded/caller AsrMetadata model_type.
      lm::proto::AsrMetadata effective_asr_meta;
      bool has_asr_meta = false;
      if (lm_resources != nullptr) {
        auto asr_meta = lm_resources->GetAsrMetadata();
        if (asr_meta.ok() && *asr_meta != nullptr) {
          effective_asr_meta.CheckTypeAndMergeFrom(**asr_meta);
          has_asr_meta = true;
        }
      }
      if (metadata != nullptr &&
          metadata->GetTypeName() == "litert.lm.proto.AsrMetadata") {
        lm::proto::AsrMetadata caller_asr;
        caller_asr.CheckTypeAndMergeFrom(*metadata);
        if (caller_asr.decode_start_token_ids_size() > 0) {
          effective_asr_meta.clear_decode_start_token_ids();
        }
        effective_asr_meta.MergeFrom(caller_asr);
        has_asr_meta = true;
      }
      if (has_asr_meta &&
          effective_asr_meta.asr_model_type().model_type_case() !=
              lm::proto::AsrModelType::MODEL_TYPE_NOT_SET) {
        std::string candidate_name;
        switch (effective_asr_meta.asr_model_type().model_type_case()) {
          case lm::proto::AsrModelType::kParakeet: {
            const bool is_tdt =
                (lm_resources != nullptr &&
                 lm_resources
                     ->GetTFLiteModelBuffer(
                         lm::proto::AsrMetadata::TF_LITE_ENCODER_DECODER)
                     .ok()) ||
                effective_asr_meta.decode_start_token_ids_size() > 0;
            candidate_name =
                is_tdt ? "parakeet-tdt-0.6b-v3" : "parakeet-ctc-0.6b";
            break;
          }
          case lm::proto::AsrModelType::kMoonshine:
            candidate_name = "moonshine-tiny";
            break;
          case lm::proto::AsrModelType::kWhisper:
            candidate_name = "whisper-tiny";
            break;
          case lm::proto::AsrModelType::kQwen3Asr:
            candidate_name = "qwen3-asr-0.6b";
            break;
          default:
            break;
        }
        if (!candidate_name.empty()) {
          model_it = models_.find(candidate_name);
          if (model_it != models_.end()) {
            resolved_model_name = candidate_name;
          }
        }
      }
      // 2. Fallback: match by normalized substring of model_name or model_path.
      if (model_it == models_.end()) {
        const std::string norm_target = NormalizeForMatching(
            resolved_model_name.empty() ? settings.model_path
                                        : resolved_model_name);
        if (!norm_target.empty()) {
          size_t best_match_len = 0;
          for (auto it = models_.begin(); it != models_.end(); ++it) {
            const std::string norm_candidate = NormalizeForMatching(it->first);
            if (norm_candidate.size() > best_match_len &&
                absl::StrContains(norm_target, norm_candidate)) {
              model_it = it;
              resolved_model_name = it->first;
              best_match_len = norm_candidate.size();
            }
          }
        }
      }
      // 3. Fallback for legacy LLM-based ASR .litertlm packages.
      if (model_it == models_.end() && lm_resources != nullptr) {
        auto llm_meta = lm_resources->GetLlmMetadata();
        if (llm_meta.ok() && *llm_meta != nullptr) {
          const std::string fallback_name =
              effective_asr_meta.log_mel_spectrogram_config().normalize_mel()
                  ? "tinygemma-asr"
                  : "qwen3-asr-0.6b";
          model_it = models_.find(fallback_name);
          if (model_it != models_.end()) {
            resolved_model_name = fallback_name;
          }
        }
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
  if (settings_with_defaults.model_remote_url.empty()) {
    settings_with_defaults.model_remote_url =
        std::move(default_model_remote_url);
  }
  OmniEngine::Settings resolved_settings =
      ResolveEngineSettings(resolved_model_name, settings_with_defaults);

  if (lm_resources == nullptr &&
      absl::EndsWith(resolved_settings.model_path, ".litertlm") &&
      FileExists(resolved_settings.model_path)) {
    ABSL_ASSIGN_OR_RETURN(lm_resources,
                          OpenLitertLmResources(resolved_settings.model_path));
  }

  std::unique_ptr<MessageLite> merged_metadata;
  const MessageLite* effective_metadata = metadata;
  if (lm_resources != nullptr &&
      factory_key.metadata_proto_type_id == "litert.lm.proto.AsrMetadata") {
    auto asr_metadata = lm_resources->GetAsrMetadata();
    if (asr_metadata.ok() && *asr_metadata != nullptr) {
      auto merged_asr =
          std::make_unique<lm::proto::AsrMetadata>(**asr_metadata);
      if (metadata != nullptr) {
        lm::proto::AsrMetadata caller_asr;
        caller_asr.CheckTypeAndMergeFrom(*metadata);
        if (caller_asr.decode_start_token_ids_size() > 0) {
          merged_asr->clear_decode_start_token_ids();
        }
        merged_asr->MergeFrom(caller_asr);
      }
      merged_metadata = std::move(merged_asr);
      effective_metadata = merged_metadata.get();
    }
  }

  LITERT_ASSIGN_OR_RETURN(::litert::Environment env,
                          ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(::litert::Options options,
                          ::litert::Options::Create());
  ABSL_RETURN_IF_ERROR(ConfigureLiteRtOptions(resolved_settings, options));

  return (*creator)(resolved_model_name, resolved_settings, session_config,
                    effective_metadata, env, options, std::move(lm_resources));
}

}  // namespace litert::omni
