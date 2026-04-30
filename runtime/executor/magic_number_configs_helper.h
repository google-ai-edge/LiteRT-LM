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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_

#include <memory>
#include <vector>

#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/options/litert_magic_number_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Helper to build magic number configs for LiteRT environment options when the
// model contains magic numbers for context length and prefill length.
class MagicNumberConfigsHelper {
 public:
  MagicNumberConfigsHelper() = default;
  ~MagicNumberConfigsHelper() = default;

  // Builds LiteRT environment options including magic number configs and
  // verifications according to the model and executor settings.
  // Note that the returned options contain pointers to the internal memory of
  // this class, and the caller should make sure the helper outlives the usage
  // of the returned options.
  std::vector<EnvironmentOptions::Option> GetLiteRtEnvOptions(
      ModelResources& resources, const LlmExecutorSettings& executor_settings);

  const LiteRtMagicNumberConfigs* magic_number_configs() const {
    return magic_number_configs_.get();
  }
  const LiteRtMagicNumberVerifications* magic_number_verifications() const {
    return magic_number_verifications_.get();
  }

 private:
  litert::options::MagicNumberConfigsPtr magic_number_configs_;
  litert::options::MagicNumberVerificationsPtr magic_number_verifications_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_
