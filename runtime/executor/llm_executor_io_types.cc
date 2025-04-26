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

#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/logging_tensor_buffer.h"

namespace litert::lm {

constexpr char kFieldIndent[] = "  ";

// Implementation for ExecutorTextData
std::ostream& operator<<(std::ostream& os, const ExecutorTextData& text_data) {
  os << "ExecutorTextData: {\n"
     << kFieldIndent << "TokenIds: " << text_data.GetTokenIds() << "\n"
     << "}";
  return os;
}

// Helper function to print an std::optional<TensorBuffer> field on a new
// indented line. This helper itself does not add a trailing newline.
static void PrintOptionalTensorBufferField(
    std::ostream& os, const std::string& field_name,
    const std::optional<::litert::TensorBuffer>& opt_buffer,
    const std::string& indent) {
  os << indent << field_name << ": ";
  if (opt_buffer.has_value()) {
    os << opt_buffer.value();
  } else {
    os << "nullopt";
  }
}

// Implementation for ExecutorVisionData
std::ostream& operator<<(std::ostream& os,
                         const ExecutorVisionData& vision_data) {
  os << "ExecutorVisionData: {\n";
  PrintOptionalTensorBufferField(os, "Embeddings", vision_data.GetEmbeddings(),
                                 kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferField(os, "PerLayerEmbeddings",
                                 vision_data.GetPerLayerEmbeddings(),
                                 kFieldIndent);
  os << "\n"
     << "}";
  return os;
}

// Implementation for ExecutorAudioData
std::ostream& operator<<(std::ostream& os,
                         const ExecutorAudioData& audio_data) {
  os << "ExecutorAudioData: {\n";
  PrintOptionalTensorBufferField(os, "Embeddings", audio_data.GetEmbeddings(),
                                 kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferField(os, "PerLayerEmbeddings",
                                 audio_data.GetPerLayerEmbeddings(),
                                 kFieldIndent);
  os << "\n"
     << "}";
  return os;
}

// Implementation for ExecutorInputs
std::ostream& operator<<(std::ostream& os, const ExecutorInputs& inputs) {
  os << "ExecutorInputs: {\n";

  os << kFieldIndent << "TextData: ";
  if (inputs.GetTextData().has_value()) {
    os << inputs.GetTextData().value();  // Relies on TextData's operator<<
  } else {
    os << "nullopt";
  }
  os << "\n";

  os << kFieldIndent << "VisionData: ";
  if (inputs.GetVisionData().has_value()) {
    os << inputs.GetVisionData().value();  // Relies on VisionData's operator<<
  } else {
    os << "nullopt";
  }
  os << "\n";

  os << kFieldIndent << "AudioData: ";
  if (inputs.GetAudioData().has_value()) {
    os << inputs.GetAudioData().value();  // Relies on AudioData's operator<<
  } else {
    os << "nullopt";
  }
  // No comma after the last field in this style
  os << "\n"
     << "}";
  return os;
}

// Implementation for ExecutorPrefillParams
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
