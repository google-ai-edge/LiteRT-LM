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
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/tts_metadata.pb.h"
#include "runtime/util/file_util.h"
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

// Sentinel `cache_dir` value that disables compiled-model caches.
constexpr absl::string_view kNoCacheDir = ":nocache";

// Where compiled-model caches go and how they are named. Empty `dir` and `key`
// mean caching is disabled.
struct ModelCacheConfig {
  std::string dir;
  // `<model_name>_<model file id>`, where the id is `<mtime>_<size>` of the
  // model file (see `lm::GetFileCacheIdentifier`).
  std::string key;
};

// Resolves the compiled-model cache location for `model_name`:
// - `cache_dir` is ":nocache": caching is disabled.
// - `cache_dir` is empty: caches go next to the model file (LiteRT-LM
//   convention); skipped silently if that directory is not writable.
// - Otherwise caches go to `cache_dir`, which is created if missing. If it is
//   not writable, caching is skipped when it is the model file's directory
//   (read-only model directories are common); otherwise this is a
//   `PermissionDeniedError`.
// Caching is also skipped when the model file does not exist yet; the session
// factory reports a missing model.
absl::StatusOr<ModelCacheConfig> ResolveModelCache(
    absl::string_view model_name, const OmniEngine::Settings& settings) {
  if (settings.cache_dir == kNoCacheDir) {
    return ModelCacheConfig{};
  }
  absl::StatusOr<std::string> model_id =
      lm::GetFileCacheIdentifier(settings.model_path);
  if (!model_id.ok()) {
    return ModelCacheConfig{};
  }
  const bool explicit_dir = !settings.cache_dir.empty();
  std::string dir =
      explicit_dir
          ? settings.cache_dir
          : std::filesystem::path(settings.model_path).parent_path().string();
  if (explicit_dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
  }
  if (dir.empty() || !lm::IsDirectoryWritable(dir)) {
    if (!explicit_dir) {
      return ModelCacheConfig{};
    }
    // `cache_dir` also holds the model file, e.g. a read-only directory of
    // pre-installed models (the Kotlin API has no separate model path), so
    // load without caches instead of failing.
    std::error_code ec;
    if (std::filesystem::equivalent(
            std::filesystem::path(settings.model_path).parent_path(), dir,
            ec)) {
      ABSL_LOG(WARNING) << "cache_dir " << dir
                        << " is not writable; compiled-model caches disabled.";
      return ModelCacheConfig{};
    }
    return absl::PermissionDeniedError(absl::StrCat(
        "cache_dir is not a writable directory: ", dir,
        ". Set cache_dir to \":nocache\" to disable compiled-model caches."));
  }
  return ModelCacheConfig{
      .dir = std::move(dir),
      .key = absl::StrCat(model_name, "_", *model_id),
  };
}

// Returns true if `s` is `<digits>_<digits>`, the format of
// `lm::GetFileCacheIdentifier`.
bool IsFileCacheIdentifier(absl::string_view s) {
  const size_t sep = s.find('_');
  if (sep == absl::string_view::npos || sep == 0 || sep + 1 == s.size()) {
    return false;
  }
  const auto all_digits = [](absl::string_view part) {
    return absl::c_all_of(part, [](char c) { return absl::ascii_isdigit(c); });
  };
  return all_digits(s.substr(0, sep)) && all_digits(s.substr(sep + 1));
}

// Returns `<cache.dir>/<cache.key><suffix>`. Also deletes stale caches of the
// same model: files named `<model_name>_<mtime>_<size><suffix>` whose
// identifier differs from the current one were built from an older version of
// the model file. Requiring a well-formed identifier keeps caches of other
// models whose names merely start with `<model_name>_` untouched.
std::string GetCachePathAndCleanStale(absl::string_view model_name,
                                      const ModelCacheConfig& cache,
                                      absl::string_view suffix) {
  const std::string current = absl::StrCat(cache.key, suffix);
  const std::string prefix = absl::StrCat(model_name, "_");
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::directory_iterator(cache.dir, ec)) {
    const std::string name = entry.path().filename().string();
    absl::string_view id = name;
    if (name == current || !absl::ConsumePrefix(&id, prefix) ||
        !absl::ConsumeSuffix(&id, suffix) || !IsFileCacheIdentifier(id)) {
      continue;
    }
    std::error_code remove_ec;
    std::filesystem::remove(entry.path(), remove_ec);
  }
  return (std::filesystem::path(cache.dir) / current).string();
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
  // ":nocache" only disables compiled-model caches; model files are still
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
void Merge<lm::proto::AsrMetadata>(lm::proto::AsrMetadata& dst,
                                   const lm::proto::AsrMetadata& src) {
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
  if (metadata_proto_type_id == "litert.lm.proto.AsrMetadata") {
    return MergeMetadataProto(lm_resources->GetAsrMetadata(), caller_metadata,
                              merged_metadata);
  }
  if (metadata_proto_type_id == "litert.lm.proto.TtsMetadata") {
    return MergeMetadataProto(lm_resources->GetTtsMetadata(), caller_metadata,
                              merged_metadata);
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
  ABSL_ASSIGN_OR_RETURN(const ModelCacheConfig cache,
                        ResolveModelCache(model_name, settings));
  if (settings.num_threads > 0 || !cache.key.empty()) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            options.GetOptions<::litert::CpuOptions>());
    if (settings.num_threads > 0) {
      cpu_options.SetNumThreads(settings.num_threads);
    }
    if (!cache.key.empty()) {
      // Persist XNNPack-packed weights in a file-backed cache: they are then
      // mmap'ed (clean, reclaimable pages) instead of living in the native
      // heap, and later loads skip re-packing.
      const std::string weight_cache_path =
          GetCachePathAndCleanStale(model_name, cache, ".xnnpack_cache");
      LITERT_RETURN_IF_ERROR(
          cpu_options.SetXNNPackWeightCachePath(weight_cache_path.c_str()));
    }
  }
  if (settings.backend == OmniEngine::Settings::Backend::kGpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kGpu);
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            options.GetOptions<::litert::GpuOptions>());
    gpu_options.SetPrecision(::litert::GpuOptions::Precision::kFp32);
    gpu_options.EnableConstantTensorSharing(true);
    if (!cache.key.empty()) {
      // Serialize compiled GPU programs and GPU-layout weights so later loads
      // skip shader compilation and weight conversion. The GPU delegate names
      // the files `<cache.key>_mldrift_{program,weight}_cache.bin`.
      for (absl::string_view suffix :
           {"_mldrift_program_cache.bin", "_mldrift_weight_cache.bin"}) {
        GetCachePathAndCleanStale(model_name, cache, suffix);
      }
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetSerializationDir(cache.dir.c_str()));
      LITERT_RETURN_IF_ERROR(gpu_options.SetModelCacheKey(cache.key.c_str()));
      LITERT_RETURN_IF_ERROR(gpu_options.SetSerializeProgramCache(true));
      LITERT_RETURN_IF_ERROR(gpu_options.SetSerializeExternalTensors(true));
    }
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
      const std::string norm_target = NormalizeForMatching(
          resolved_model_name.empty() ? settings.model_path
                                      : resolved_model_name);
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
