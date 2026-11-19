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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {

// Specification of an input or output tensor in a minimal test TFLite model.
struct TestTensorSpec {
  std::string name;
  std::vector<int32_t> dims;
  tflite::TensorType type;
};

// Builds a minimal executable TFLite FlatBuffer model with the requested
// signature input and output tensor specifications (all outputs emit zeros).
// Output tensors must use `tflite::TensorType_FLOAT32`, matching the
// unquantized `ADD` op synthesized for each output.
std::string BuildMinimalTfliteModelBuffer(
    absl::Span<const TestTensorSpec> inputs,
    absl::Span<const TestTensorSpec> outputs);

// Single `.litertlm` section payload used when building synthetic test bundles.
struct BundleSectionPayload {
  lm::schema::AnySectionDataType data_type =
      lm::schema::AnySectionDataType_NONE;
  std::string model_type;
  std::string backend_constraint;
  std::string prefer_activation_type;
  std::string bytes;
};

// Options for constructing a synthetic `.litertlm` Any2Any test bundle.
struct Any2AnyBundleOptions {
  // Optional embedded `Any2AnyMetadata` serialized as an
  // `Any2AnyMetadataProto` section when set.
  std::optional<lm::proto::Any2AnyMetadata> metadata;
  // Raw HuggingFace tokenizer JSON compressed into `HF_Tokenizer_Zlib`.
  std::string tokenizer_json;
  // Optional constraints attached to `tf_lite_text_embedder` and
  // `tf_lite_llm_backbone` sections.
  std::string backbone_backend_constraint;
  std::string backbone_prefer_activation_type;
  // Additional sections appended after the standard tokenizer,
  // `tf_lite_text_embedder`, and `tf_lite_llm_backbone` sections.
  std::vector<BundleSectionPayload> extra_sections;
};

// Writes a minimal `.litertlm` test bundle to `::testing::TempDir() / filename`
// containing the checksum fake embedder and LLM backbone models, compressed
// tokenizer JSON, optional embedded `Any2AnyMetadata`, and `extra_sections`.
absl::StatusOr<std::string> WriteAny2AnyTestBundle(
    absl::string_view filename, Any2AnyBundleOptions options);

// Fake upstream TTS stage allowing unit tests to enqueue `Any2AnyTtsOutput`
// items directly for downstream vocoder stage tests.
class FakeTtsStage : public SingleThreadedStageWithDeque<Any2AnyTtsOutput> {
 public:
  void Push(Any2AnyTtsOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

// Creates a mock `LiteRtRunner` configured with `Int32[1, input_tokens]` input
// and `Float32[1, output_samples]` output buffers.
//
// The default `Run()` behavior optionally appends each input token window to
// `recorded_windows` (if non-null) and writes deterministic PCM samples
// `code * 10 + sample_in_tok` using `samples_per_token` (or
// `output_samples / input_tokens` when `samples_per_token <= 0`).
std::unique_ptr<::testing::NiceMock<MockLiteRtRunner>> MakeMockVocoderRunner(
    int input_tokens, int output_samples,
    std::vector<std::vector<int32_t>>* recorded_windows = nullptr,
    int samples_per_token = 0);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_
