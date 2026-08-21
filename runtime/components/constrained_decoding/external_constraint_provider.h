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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_EXTERNAL_CONSTRAINT_PROVIDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_EXTERNAL_CONSTRAINT_PROVIDER_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"

namespace litert::lm {

class ExternalConstraintProvider : public ConstraintProvider {
 public:
  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      ConstraintArg constraint_arg) const override;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_EXTERNAL_CONSTRAINT_PROVIDER_H_
