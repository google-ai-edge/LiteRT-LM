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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_CUSTOM_MODEL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_CUSTOM_MODEL_UTILS_H_

#include <optional>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::omni {

// Packs `default_proto` into `model_type.custom_model` when the `model_type`
// oneof has no arm set (`MODEL_TYPE_NOT_SET`). Leaves `model_type` untouched
// when any built-in or `custom_model` arm is already populated.
//
// `ModelTypeT` must be one of the Omni `*ModelType` protos defining a
// `oneof model_type` with a `google.protobuf.Any custom_model` field (e.g.,
// `Any2AnyModelType`, `ImageGenModelType`, `Speech2TextModelType`, or
// `Text2SpeechModelType`).
//
// args
// - model_type: Target `*ModelType` proto message to populate if unset.
// - default_proto: Custom proto payload to pack when `model_type` is unset.
//
// returns
// - absl::OkStatus() on success (including when already populated), or
//   absl::InternalError if packing fails.
template <typename ProtoT, typename ModelTypeT>
absl::Status EnsureDefaultCustomModel(ModelTypeT& model_type,
                                      const ProtoT& default_proto = {}) {
  if (model_type.model_type_case() != ModelTypeT::MODEL_TYPE_NOT_SET) {
    return absl::OkStatus();
  }
  if (!model_type.mutable_custom_model()->PackFrom(default_proto)) {
    model_type.clear_custom_model();
    return absl::InternalError(absl::StrCat(
        "Failed to pack default ", default_proto.GetTypeName(), " proto into ",
        model_type.GetTypeName(), ".custom_model."));
  }
  return absl::OkStatus();
}

// Unpacks `model_type.custom_model` as `ProtoT`.
//
// `ModelTypeT` must be one of the Omni `*ModelType` protos defining a
// `oneof model_type` with a `google.protobuf.Any custom_model` field.
//
// args
// - model_type: Source `*ModelType` proto message to inspect.
//
// returns
// - `std::nullopt` if no `model_type` arm is set (`MODEL_TYPE_NOT_SET`).
// - Unpacked `ProtoT` if `custom_model` is set and holds a valid `ProtoT`.
// - `absl::InvalidArgumentError` if a different `model_type` arm is set or if
//   unpacking `custom_model` as `ProtoT` fails.
template <typename ProtoT, typename ModelTypeT>
absl::StatusOr<std::optional<ProtoT>> UnpackCustomModel(
    const ModelTypeT& model_type) {
  if (model_type.model_type_case() == ModelTypeT::MODEL_TYPE_NOT_SET) {
    return std::optional<ProtoT>(std::nullopt);
  }
  ProtoT proto;
  if (!model_type.has_custom_model()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", model_type.GetTypeName(),
                     ".custom_model packing ", proto.GetTypeName(), "."));
  }
  if (!model_type.custom_model().UnpackTo(&proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to unpack ", model_type.GetTypeName(), ".custom_model as ",
        proto.GetTypeName(), " (got type_url='",
        model_type.custom_model().type_url(), "')."));
  }
  return std::optional<ProtoT>(std::move(proto));
}

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_CUSTOM_MODEL_UTILS_H_
