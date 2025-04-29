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

#include "runtime/executor/llm_executor_io_types.h"

#include <atomic>
#include <ios>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/logging_tensor_buffer.h"

namespace litert::lm {

constexpr char kFieldIndent[] = "  ";

ExecutorTextData::ExecutorTextData(::litert::TensorBuffer&& token_ids)
    : token_ids_(std::move(token_ids)) {}

const ::litert::TensorBuffer& ExecutorTextData::GetTokenIds() const {
  return token_ids_;
}

::litert::TensorBuffer& ExecutorTextData::GetMutableTokenIds() {
  return token_ids_;
}

void ExecutorTextData::SetTokenIds(::litert::TensorBuffer&& token_ids) {
  token_ids_ = std::move(token_ids);
}

std::ostream& operator<<(std::ostream& os, const ExecutorTextData& text_data) {
  os << "ExecutorTextData: {\n"
     << kFieldIndent << "TokenIds: " << text_data.GetTokenIds() << "\n"
     << "}";
  return os;
}

ExecutorVisionData::ExecutorVisionData(
    std::optional<::litert::TensorBuffer>&& embeddings,
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings)
    : embeddings_(std::move(embeddings)),
      per_layer_embeddings_(std::move(per_layer_embeddings)) {}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorVisionData::GetEmbeddingsPtr() const {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError("ExecutorVisionData::embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorVisionData::GetMutableEmbeddingsPtr() {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::embeddings_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorVisionData::GetPerLayerEmbeddingsPtr() const {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::per_layer_embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorVisionData::GetMutablePerLayerEmbeddingsPtr() {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::per_layer_embeddings_ is not set.");
}

void ExecutorVisionData::SetEmbeddings(
    std::optional<::litert::TensorBuffer>&& embeddings) {
  embeddings_ = std::move(embeddings);
}

void ExecutorVisionData::SetPerLayerEmbeddings(
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
  per_layer_embeddings_ = std::move(per_layer_embeddings);
}

// Helper function to print a field from StatusOr<const TensorBuffer*>
static void PrintOptionalTensorBufferFieldFromStatusOr(
    std::ostream& os, const std::string& field_name,
    const absl::StatusOr<const ::litert::TensorBuffer*>& opt_buffer_status,
    const std::string& indent) {
  os << indent << field_name << ": ";
  if (opt_buffer_status.ok()) {
    const ::litert::TensorBuffer* buffer_ptr = opt_buffer_status.value();
    if (buffer_ptr) {  // Should always be true.
      os << *buffer_ptr;
    } else {  // Should not happen if status is ok and value is a pointer
      os << "null (unexpected)";
    }
  } else {
    os << "nullopt (" << opt_buffer_status.status().message() << ")";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorVisionData& vision_data) {
  os << "ExecutorVisionData: {\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "Embeddings", vision_data.GetEmbeddingsPtr(), kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "PerLayerEmbeddings", vision_data.GetPerLayerEmbeddingsPtr(),
      kFieldIndent);
  os << "\n"
     << "}";
  return os;
}

ExecutorAudioData::ExecutorAudioData(
    std::optional<::litert::TensorBuffer>&& embeddings,
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings)
    : embeddings_(std::move(embeddings)),
      per_layer_embeddings_(std::move(per_layer_embeddings)) {}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorAudioData::GetEmbeddingsPtr() const {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError("ExecutorAudioData::embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorAudioData::GetMutableEmbeddingsPtr() {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::embeddings_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorAudioData::GetPerLayerEmbeddingsPtr() const {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::per_layer_embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorAudioData::GetMutablePerLayerEmbeddingsPtr() {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::per_layer_embeddings_ is not set.");
}

void ExecutorAudioData::SetEmbeddings(
    std::optional<::litert::TensorBuffer>&& embeddings) {
  embeddings_ = std::move(embeddings);
}

void ExecutorAudioData::SetPerLayerEmbeddings(
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
  per_layer_embeddings_ = std::move(per_layer_embeddings);
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorAudioData& audio_data) {
  os << "ExecutorAudioData: {\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "Embeddings", audio_data.GetEmbeddingsPtr(), kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "PerLayerEmbeddings", audio_data.GetPerLayerEmbeddingsPtr(),
      kFieldIndent);
  os << "\n"
     << "}";
  return os;
}

ExecutorInputs::ExecutorInputs(std::optional<ExecutorTextData>&& text_data,
                               std::optional<ExecutorVisionData>&& vision_data,
                               std::optional<ExecutorAudioData>&& audio_data)
    : text_data_(std::move(text_data)),
      vision_data_(std::move(vision_data)),
      audio_data_(std::move(audio_data)) {}

absl::StatusOr<const ExecutorTextData*> ExecutorInputs::GetTextDataPtr() const {
  if (text_data_.has_value()) {
    return &text_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::text_data_ is not set.");
}

absl::StatusOr<ExecutorTextData*> ExecutorInputs::GetMutableTextDataPtr() {
  if (text_data_.has_value()) {
    return &text_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::text_data_ is not set.");
}

absl::StatusOr<const ExecutorVisionData*> ExecutorInputs::GetVisionDataPtr()
    const {
  if (vision_data_.has_value()) {
    return &vision_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::vision_data_ is not set.");
}

absl::StatusOr<ExecutorVisionData*> ExecutorInputs::GetMutableVisionDataPtr() {
  if (vision_data_.has_value()) {
    return &vision_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::vision_data_ is not set.");
}

absl::StatusOr<const ExecutorAudioData*> ExecutorInputs::GetAudioDataPtr()
    const {
  if (audio_data_.has_value()) {
    return &audio_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::audio_data_ is not set.");
}

absl::StatusOr<ExecutorAudioData*> ExecutorInputs::GetMutableAudioDataPtr() {
  if (audio_data_.has_value()) {
    return &audio_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::audio_data_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetTextTokenIdsPtr() const {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for TokenIds).");
  }
  return &(text_data_->GetTokenIds());
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableTextTokenIdsPtr() {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for "
        "TokenIds).");
  }
  return &(text_data_->GetMutableTokenIds());
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetVisionEmbeddingsPtr() const {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required for Vision "
        "Embeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> embeddings_ptr_status =
      vision_data_->GetEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(embeddings_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableVisionEmbeddingsPtr() {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required "
        "for Vision Embeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> embeddings_ptr_status =
      vision_data_->GetMutableEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(
        embeddings_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetVisionPerLayerEmbeddingsPtr() const {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required for Vision "
        "PerLayerEmbeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> per_layer_ptr_status =
      vision_data_->GetPerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(per_layer_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableVisionPerLayerEmbeddingsPtr() {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required "
        "for Vision PerLayerEmbeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> per_layer_ptr_status =
      vision_data_->GetMutablePerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(
        per_layer_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetAudioEmbeddingsPtr() const {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for Audio "
        "Embeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> embeddings_ptr_status =
      audio_data_->GetEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(embeddings_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableAudioEmbeddingsPtr() {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for "
        "Audio Embeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> embeddings_ptr_status =
      audio_data_->GetMutableEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(
        embeddings_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetAudioPerLayerEmbeddingsPtr() const {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for Audio "
        "PerLayerEmbeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> per_layer_ptr_status =
      audio_data_->GetPerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(per_layer_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableAudioPerLayerEmbeddingsPtr() {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for "
        "Audio PerLayerEmbeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> per_layer_ptr_status =
      audio_data_->GetMutablePerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(
        per_layer_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

void ExecutorInputs::SetTextData(ExecutorTextData&& text_data) {
  text_data_ = std::move(text_data);
}

void ExecutorInputs::SetVisionData(
    std::optional<ExecutorVisionData>&& vision_data) {
  vision_data_ = std::move(vision_data);
}

void ExecutorInputs::SetAudioData(
    std::optional<ExecutorAudioData>&& audio_data) {
  audio_data_ = std::move(audio_data);
}

std::ostream& operator<<(std::ostream& os, const ExecutorInputs& inputs) {
  os << "ExecutorInputs: {\n";

  os << kFieldIndent << "TextData: ";
  absl::StatusOr<const ExecutorTextData*> text_data_status =
      inputs.GetTextDataPtr();
  if (text_data_status.ok()) {
    os << *text_data_status.value();  // Relies on TextData's operator<<
  } else {
    os << "nullopt (" << text_data_status.status().message() << ")";
  }
  os << "\n";

  os << kFieldIndent << "VisionData: ";
  absl::StatusOr<const ExecutorVisionData*> vision_data_status =
      inputs.GetVisionDataPtr();
  if (vision_data_status.ok()) {
    os << *vision_data_status.value();  // Relies on VisionData's operator<<
  } else {
    os << "nullopt (" << vision_data_status.status().message() << ")";
  }
  os << "\n";

  os << kFieldIndent << "AudioData: ";
  absl::StatusOr<const ExecutorAudioData*> audio_data_status =
      inputs.GetAudioDataPtr();
  if (audio_data_status.ok()) {
    os << *audio_data_status.value();  // Relies on AudioData's operator<<
  } else {
    os << "nullopt (" << audio_data_status.status().message() << ")";
  }
  os << "\n"
     << "}";
  return os;
}

// --- ExecutorPrefillParams Implementation ---
ExecutorPrefillParams::ExecutorPrefillParams(int current_step,
                                             bool wait_for_completion,
                                             const std::atomic_bool* cancel)
    : current_step_(current_step),
      wait_for_completion_(wait_for_completion),
      cancel_(cancel) {}

int ExecutorPrefillParams::GetCurrentStep() const { return current_step_; }

void ExecutorPrefillParams::SetCurrentStep(int current_step) {
  current_step_ = current_step;
}

bool ExecutorPrefillParams::GetWaitForCompletion() const {
  return wait_for_completion_;
}

void ExecutorPrefillParams::SetWaitForCompletion(bool wait_for_completion) {
  wait_for_completion_ = wait_for_completion;
}

const std::atomic_bool* ExecutorPrefillParams::GetCancelFlag() const {
  return cancel_;
}

void ExecutorPrefillParams::SetCancelFlag(const std::atomic_bool* cancel) {
  cancel_ = cancel;
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorPrefillParams& params) {
  os << "ExecutorPrefillParams: {\n"
     << kFieldIndent << "CurrentStep: " << params.GetCurrentStep() << "\n"
     << kFieldIndent << "WaitForCompletion: " << std::boolalpha
     << params.GetWaitForCompletion() << "\n"
     << kFieldIndent << "CancelFlag: ";
  if (params.GetCancelFlag() != nullptr) {
    os << (params.GetCancelFlag()->load(std::memory_order_relaxed)
               ? "true (atomic)"
               : "false (atomic)");
  } else {
    os << "nullptr";
  }
  os << "\n"
     << "}";
  return os;
}

}  // namespace litert::lm
