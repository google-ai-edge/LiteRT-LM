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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_REGISTRY_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"

namespace litert::omni {

// Process-wide singleton registry mapping model names to
// `OmniSessionFactory` creators. Allows clients to selectively link only the
// models and dependencies they need into the final binary.
class OmniRegistry {
 public:
  // Factory callback that constructs an `OmniSessionFactory` for `model_name`
  // using `settings`, optional `session_config`, optional `metadata`,
  // initialized `env` and `options`, and optional opened `lm_resources`.
  using SessionFactoryCreator =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<OmniSessionFactory>>(
          absl::string_view model_name, const OmniEngine::Settings& settings,
          const MessageLite* absl_nullable session_config,
          const MessageLite* absl_nullable metadata,
          ::litert::Environment& env, ::litert::Options& options,
          std::shared_ptr<lm::ModelResources> absl_nullable lm_resources)
                             const>;

  // Specification of one or more models registered with `OmniRegistry`.
  struct RegistrySpec {
    // First element is the canonical model name; remaining elements are
    // aliases.
    std::vector<std::string> model_names;
    SessionFactoryCreator get_session_factory;
    std::string session_config_type_id;
    std::optional<std::string> metadata_type_id;
    std::optional<std::string> model_remote_url;
    // TODO(b/524681030): Add an optional `matches(const lm::ModelResources*,
    // absl::string_view path)` predicate so each registration can own its
    // model detection from a path or `.litertlm` container.
  };

  // Read-only view of a registered `RegistrySpec` and its resolved proto type
  // IDs.
  struct RegisteredModel {
    std::string model_name;
    std::string canonical_model_name;
    std::string session_config_proto_type_id;
    std::string metadata_proto_type_id;
    std::string model_remote_url;
  };

  // Returns the process-wide `OmniRegistry` singleton instance.
  static OmniRegistry& Get();

  // Resolves a `session_config_type` registration key to a full Protobuf
  // message type name:
  //   - Short key `"Xxx"` (no `.`): resolves to
  //     `"litert.lm.proto.XxxSessionConfig"`.
  //   - Full type name `"d.e.f"` (contains `.`): used as-is `"d.e.f"`.
  absl::StatusOr<std::string> ResolveSessionConfigProtoTypeId(
      absl::string_view session_config_type) const;

  // Resolves a `metadata_type` registration key (given `session_config_type`)
  // to a full Protobuf message type name:
  //   - Empty `""` when `session_config_type` is a short key `"Xxx"`:
  //     derives `"litert.lm.proto.XxxMetadata"`.
  //   - Short key `"Yyy"` (no `.`): resolves to
  //     `"litert.lm.proto.YyyMetadata"`.
  //   - Full type name `"a.b.c"` (contains `.`): used as-is `"a.b.c"`.
  //   - Empty `""` when `session_config_type` is a full type name `"d.e.f"`:
  //     returns `InvalidArgumentError`.
  absl::StatusOr<std::string> ResolveMetadataProtoTypeId(
      absl::string_view session_config_type,
      absl::string_view metadata_type = "") const;

  // Registers a `RegistrySpec` under the resolved
  // `(session_config_type_id, metadata_type_id)` proto type IDs.
  absl::Status RegisterSessionFactory(RegistrySpec spec);

  // Unregisters the `SessionFactoryCreator` for the resolved
  // `(session_config_type, metadata_type)` pair and removes any models mapped
  // to that pair. Returns true if anything was removed.
  bool UnregisterSessionFactory(absl::string_view session_config_type,
                                absl::string_view metadata_type = "");

  // Unregisters a single `model_name` from the model table. Returns true if
  // `model_name` was removed.
  bool UnregisterModel(absl::string_view model_name);

  // Returns true if `model_name` is currently registered.
  bool IsModelRegistered(absl::string_view model_name) const;

  // Returns the list of currently registered `model_name`s.
  std::vector<std::string> GetRegisteredModels() const;

  // Returns the longest registered model name contained in the file name of
  // `model_path`, ignoring case, '-' and '_' (e.g. "kokoro" for
  // "/x/kokoro_int8_gpu.litertlm"), or `NotFoundError` if none matches.
  // Published model files carry variant suffixes, so their stem is usually not
  // a registered name itself.
  absl::StatusOr<std::string> FindModelNameForPath(
      absl::string_view model_path) const;

  // Returns a `RegisteredModel` view of the registration for `model_name`, or
  // `NotFoundError` if `model_name` is not registered.
  absl::StatusOr<RegisteredModel> GetRegisteredModel(
      absl::string_view model_name) const;

  // Creates an `OmniSessionFactory` for a registered `model_name` by validating
  // `session_config` and `metadata` proto types, initializing
  // `litert::Environment`, `litert::Options`, and `lm::ModelResources`, and
  // invoking the registered `get_session_factory`.
  absl::StatusOr<std::unique_ptr<OmniSessionFactory>> CreateSessionFactory(
      absl::string_view model_name, const OmniEngine::Settings& settings = {},
      const MessageLite* absl_nullable session_config = nullptr,
      const MessageLite* absl_nullable metadata = nullptr) const;

 private:
  struct FactoryKey {
    std::string session_config_proto_type_id;
    std::string metadata_proto_type_id;

    bool operator==(const FactoryKey& other) const {
      return session_config_proto_type_id ==
                 other.session_config_proto_type_id &&
             metadata_proto_type_id == other.metadata_proto_type_id;
    }

    template <typename H>
    friend H AbslHashValue(H h, const FactoryKey& key) {
      return H::combine(std::move(h), key.session_config_proto_type_id,
                        key.metadata_proto_type_id);
    }
  };

  struct ModelEntry {
    FactoryKey factory_key;
    std::string canonical_model_name;
    std::string model_remote_url;
    std::shared_ptr<SessionFactoryCreator> factory_creator;
  };

  friend class absl::NoDestructor<OmniRegistry>;
  OmniRegistry() = default;

  // Returns the longest registered model name whose normalized form (lowercase,
  // without '-' and '_') is contained in `normalized_target`, or an empty
  // string if none matches.
  std::string FindLongestContainedModelName(absl::string_view normalized_target)
      const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<FactoryKey, std::shared_ptr<SessionFactoryCreator>>
      factories_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, ModelEntry> models_ ABSL_GUARDED_BY(mutex_);
};

using RegistrySpec = OmniRegistry::RegistrySpec;

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_REGISTRY_H_
