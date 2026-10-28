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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_

// All the loaded model resources the executor needs to hold to avoid the model
// being destroyed.
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
#include "support/tokenizer/huggingface_tokenizer.h"
#endif  // ENABLE_HUGGINGFACE_TOKENIZER
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
#include "support/tokenizer/sentencepiece_tokenizer.h"
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
using ::litert::support::HuggingFaceTokenizer;
#endif  // ENABLE_HUGGINGFACE_TOKENIZER
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
using ::litert::support::SentencePieceTokenizer;
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
using ::litert::support::Tokenizer;

// LITE_RUNTIME compatible helper functions converting capability-scoped
// Protobuf TfLiteModelType enums to canonical lowercase wire strings.
std::string TfLiteModelTypeToWireString(
    proto::LlmMetadata::TfLiteModelType model_type);
std::string TfLiteModelTypeToWireString(
    proto::EmbeddingMetadata::TfLiteModelType model_type);
std::string TfLiteModelTypeToWireString(
    proto::Text2SpeechMetadata::TfLiteModelType model_type);
std::string TfLiteModelTypeToWireString(
    proto::Speech2TextMetadata::TfLiteModelType model_type);
std::string TfLiteModelTypeToWireString(
    proto::Text2ImageMetadata::TfLiteModelType model_type);
std::string TfLiteModelTypeToWireString(
    proto::Any2AnyMetadata::TfLiteModelType model_type);

// Parses a wire string (e.g. "tf_lite_prefill_decode") or enum name (e.g.
// "TF_LITE_PREFILL_DECODE") into the LLM capability-scoped Protobuf enum. The
// parsing is case insensitive and LITE_RUNTIME compatible.
//
// args:
//   model_type_str: The wire string or enum name to parse.
// returns:
//   The matching proto::LlmMetadata::TfLiteModelType, or InvalidArgumentError
//   if the string does not name a known LLM model type.
absl::StatusOr<proto::LlmMetadata::TfLiteModelType> ParseLlmTfLiteModelType(
    absl::string_view model_type_str);

// True for the capability-scoped Protobuf TfLiteModelType enums, e.g.
// proto::LlmMetadata::TfLiteModelType or
// proto::EmbeddingMetadata::TfLiteModelType. Used to constrain the generic
// per-capability template wrappers that forward to by-name (wire string) APIs.
template <typename TfLiteModelTypeT>
constexpr bool kIsTfLiteModelType = std::is_enum_v<TfLiteModelTypeT>;

// True for the capability-scoped Protobuf TfLiteModelType enums other than the
// LLM one, e.g. proto::EmbeddingMetadata::TfLiteModelType,
// proto::Text2SpeechMetadata::TfLiteModelType,
// proto::Speech2TextMetadata::TfLiteModelType,
// proto::Text2ImageMetadata::TfLiteModelType or
// proto::Any2AnyMetadata::TfLiteModelType. Used to route these enums to the
// by-name (wire string) ModelResources overloads.
template <typename TfLiteModelTypeT>
constexpr bool kIsNonLlmTfLiteModelType =
    kIsTfLiteModelType<TfLiteModelTypeT> &&
    !std::is_same_v<TfLiteModelTypeT, proto::LlmMetadata::TfLiteModelType>;

// Describes the location of a contiguous region of bytes in a file.
struct FileRegion {
  size_t offset;
  size_t size;
};

// External buffer group name of a model's own TFLiteWeights section. The same
// name followed by kWeightGroupModelTypeSeparator and a model type wire string
// (e.g. "tflite_weights:tf_lite_embedder") references that model's
// TFLiteWeights section instead. This lets a model reuse a payload stored once
// for several models (such as a token embedding table that is also the tied
// output projection).
constexpr absl::string_view kModelWeightGroupName = "tflite_weights";
constexpr absl::string_view kWeightGroupModelTypeSeparator = ":";

// ModelResources is an interface that manages all the loaded model resources
// that need to be hold to avoid the model being destroyed. It provides a way
// to load the models in a lazy way.
// Basically, it will create the models when they are actually used. Before the
// Get*() functions are called, the models are not created yet. And once the
// models are created, they will be re-used for all the following calls.
//
// Every per-model accessor comes in three flavors:
// - a virtual overload taking proto::LlmMetadata::TfLiteModelType,
// - a virtual by-name overload taking the wire string (e.g.
//   "tf_lite_text_encoder"), whose default implementation parses the string
//   with ParseLlmTfLiteModelType() and dispatches to the LLM enum overload,
// - a template overload accepting any other capability-scoped Protobuf
//   TfLiteModelType enum (e.g. proto::EmbeddingMetadata::TfLiteModelType),
//   which forwards to the by-name overload.
// Subclasses overriding any member of these overload sets should add
// `using ModelResources::<Method>;` to keep the other overloads visible.
//
// It's not thread-safe.
class ModelResources {
 public:
  virtual ~ModelResources() = default;

  // Returns true if the ModelResources contains a TFLite model for the given
  // model type.
  virtual bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) {
    return false;
  }

  // Returns true if the ModelResources contains a TFLite model for the given
  // model type wire string.
  virtual bool HasTFLiteModel(absl::string_view model_type_str);

  // Returns true if the ModelResources contains a TFLite model for the given
  // capability-scoped TfLiteModelTypeT.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  bool HasTFLiteModel(TfLiteModelTypeT model_type) {
    return HasTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  // Returns the litert model. We will create the model if it is not created
  // yet. And the model is created from memory mapped file, so physical memory
  // is only allocated when the model is actually used.
  virtual absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return absl::UnimplementedError("GetTFLiteModel is not implemented.");
  }

  // Returns the litert model for the given model type wire string.
  virtual absl::StatusOr<const litert::Model*> GetTFLiteModel(
      absl::string_view model_type_str);

  // Returns the litert model for the given TfLiteModelTypeT, for supported
  // capability-scoped Engines. Here TfLiteModelTypeT is an enum type defined in
  // the capability-scoped metadata proto, e.g.
  // - proto::EmbeddingMetadata::TfLiteModelType,
  // - proto::Text2SpeechMetadata::TfLiteModelType,
  // - proto::Speech2TextMetadata::TfLiteModelType,
  // - proto::Text2ImageMetadata::TfLiteModelType,
  // - proto::Any2AnyMetadata::TfLiteModelType.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      TfLiteModelTypeT model_type) {
    return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  // Returns the TFLite model buffer. Note that the returned string_view is
  // valid only until the ModelResources is destroyed.
  // When there is no model for the given model type, it will return an error
  // status.
  // Prefer to use GetTFLiteModel() if possible, as this function will leave
  // the model lifecycle management to the caller.
  virtual absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return absl::UnimplementedError("GetTFLiteModelBuffer is not implemented.");
  }

  // Returns the TFLite model buffer for the given model type wire string.
  virtual absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str);

  // Returns the TFLite model buffer for the given capability-scoped
  // TfLiteModelTypeT (see GetTFLiteModel above).
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      TfLiteModelTypeT model_type) {
    return GetTFLiteModelBuffer(TfLiteModelTypeToWireString(model_type));
  }

  // Returns the reference to the ScopedFile. This is used for the getting the
  // external weights that should not be mmapped into the memory.
  virtual absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() {
    return absl::UnimplementedError("GetScopedFile is not implemented.");
  }

  // Returns the section start offset and end offset.
  virtual absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return absl::UnimplementedError(
        "GetWeightsSectionOffset is not implemented.");
  }

  // Returns the section start offset and end offset for the given model type
  // wire string.
  virtual absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str);

  // Returns the section start offset and end offset for the given
  // capability-scoped TfLiteModelTypeT.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      TfLiteModelTypeT model_type) {
    return GetWeightsSectionOffset(TfLiteModelTypeToWireString(model_type));
  }

  // Returns the region of the requested model type in the ModelResources.
  virtual absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion is not implemented.");
  }

  // Returns the region of the requested model type wire string in the
  // ModelResources.
  virtual absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str);

  // Returns the region of the requested capability-scoped TfLiteModelTypeT in
  // the ModelResources.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      TfLiteModelTypeT model_type) {
    return GetTFLiteModelSectionFileRegion(
        TfLiteModelTypeToWireString(model_type));
  }

  // Returns the TFLite model backend constraint. When there is no constraint
  // for the given model type, it will return an nullopt.
  virtual std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return std::nullopt;
  }

  // Returns the TFLite model backend constraint for the given model type wire
  // string, or nullopt if there is none.
  virtual std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str);

  // Returns the TFLite model backend constraint for the given
  // capability-scoped TfLiteModelTypeT, or nullopt if there is none.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      TfLiteModelTypeT model_type) {
    return GetTFLiteModelBackendConstraint(
        TfLiteModelTypeToWireString(model_type));
  }

  // Returns the TFLite model prefer activation type. When there is no
  // prefer activation type for the given model type, it will return an
  // nullopt.
  virtual std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) {
    return std::nullopt;
  }

  // Returns the TFLite model prefer activation type for the given model type
  // wire string, or nullopt if there is none.
  virtual std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str);

  // Returns the TFLite model prefer activation type for the given
  // capability-scoped TfLiteModelTypeT, or nullopt if there is none.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      TfLiteModelTypeT model_type) {
    return GetTFLiteModelPreferActivationType(
        TfLiteModelTypeToWireString(model_type));
  }

  // Builds a tokenizer instance from the model and returns it.
  virtual absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() = 0;

  // Builds a tokenizer instance for the specified model type and returns it.
  // Defaults to GetTokenizer() when model_type is TF_LITE_PREFILL_DECODE.
  virtual absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      proto::LlmMetadata::TfLiteModelType model_type);

  // Builds a tokenizer instance for the specified model type wire string and
  // returns it.
  virtual absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      absl::string_view model_type_str);

  // Builds a tokenizer instance for the specified capability-scoped
  // TfLiteModelTypeT and returns it.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      TfLiteModelTypeT model_type) {
    return GetTokenizer(TfLiteModelTypeToWireString(model_type));
  }

  // Returns a tokenizer for `model_type`, creating it on first use and
  // re-using it afterwards. Ownership stays with ModelResources; the returned
  // pointer is valid for the lifetime of this object.
  virtual absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      proto::LlmMetadata::TfLiteModelType model_type);

  // Returns a tokenizer for the model type wire string, creating it on first
  // use and re-using it afterwards.
  virtual absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      absl::string_view model_type_str);

  // Returns a tokenizer for the capability-scoped TfLiteModelTypeT, creating
  // it on first use and re-using it afterwards.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      TfLiteModelTypeT model_type) {
    return GetOrCreateTokenizer(TfLiteModelTypeToWireString(model_type));
  }

  // Returns the llm metadata.
  virtual absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() = 0;

  // Returns the executor metadata.
  virtual absl::StatusOr<const proto::ExecutorMetadata*>
  GetExecutorMetadata() {
    return absl::UnimplementedError("GetExecutorMetadata is not implemented.");
  }

  // Returns the embedding metadata.
  virtual absl::StatusOr<const proto::EmbeddingMetadata*>
  GetEmbeddingMetadata() {
    return absl::UnimplementedError("GetEmbeddingMetadata is not implemented.");
  }

  // Returns the Text2Speech metadata.
  virtual absl::StatusOr<const proto::Text2SpeechMetadata*>
  GetText2SpeechMetadata() {
    return absl::UnimplementedError(
        "GetText2SpeechMetadata is not implemented.");
  }

  // Returns the Speech2Text metadata.
  virtual absl::StatusOr<const proto::Speech2TextMetadata*>
  GetSpeech2TextMetadata() {
    return absl::UnimplementedError(
        "GetSpeech2TextMetadata is not implemented.");
  }

  // Returns the Text2Image metadata.
  virtual absl::StatusOr<const proto::Text2ImageMetadata*>
  GetText2ImageMetadata() {
    return absl::UnimplementedError(
        "GetText2ImageMetadata is not implemented.");
  }

  // Returns the Any2Any metadata.
  virtual absl::StatusOr<const proto::Any2AnyMetadata*> GetAny2AnyMetadata() {
    return absl::UnimplementedError("GetAny2AnyMetadata is not implemented.");
  }

  // Returns a zero-copy buffer for a GenericBinaryData section matching the
  // given name.
  virtual absl::StatusOr<absl::string_view> GetGenericBinaryDataBuffer(
      absl::string_view name) {
    return absl::UnimplementedError(
        "GetGenericBinaryDataBuffer is not implemented.");
  }

  // Returns the names of all named GenericBinaryData sections in the model
  // container.
  virtual std::vector<std::string> GetGenericBinaryDataNames() const;

  // Returns in-memory weights for a specific model type if available.
  // Used by the streaming weights API when running a model on CPU.
  virtual const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(proto::LlmMetadata::TfLiteModelType model_type) const;

  // Returns in-memory weights for a specific model type wire string if
  // available, or nullptr otherwise.
  virtual const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(absl::string_view model_type_str) const;

  // Returns in-memory weights for a specific capability-scoped
  // TfLiteModelTypeT if available, or nullptr otherwise.
  template <
      typename TfLiteModelTypeT,
      typename = std::enable_if_t<kIsNonLlmTfLiteModelType<TfLiteModelTypeT>>>
  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(TfLiteModelTypeT model_type) const {
    return GetWeightInMemoryMap(TfLiteModelTypeToWireString(model_type));
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
