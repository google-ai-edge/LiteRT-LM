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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CONFIG_H_

#include <variant>

#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/fst_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"

namespace litert::lm {

using ConstraintProviderConfig =
    std::variant<ExternalConstraintConfig, LlGuidanceConfig, FstConfig>;

using ConstraintArg = std::variant<ExternalConstraintArg,
                                   LlGuidanceConstraintArg, FstConstraintArg>;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CONFIG_H_
