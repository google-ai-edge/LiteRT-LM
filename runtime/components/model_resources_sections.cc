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

#include "runtime/components/model_resources_sections.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

namespace {

absl::StatusOr<litert::Model> CreateModelFromFileSection(ScopedFile& model_file,
                                                         uint64_t begin_offset,
                                                         uint64_t end_offset) {
  if (end_offset <= begin_offset) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid LiteRT-LM section range: [", begin_offset, ", ",
                     end_offset, ")"));
  }

  LITERT_ASSIGN_OR_RETURN(ScopedFile dup_file, model_file.Duplicate());
  LITERT_ASSIGN_OR_RETURN(int fd, dup_file.Release());
  litert::Expected<litert::Model> model =
      litert::Model::CreateFromFd(fd, begin_offset, end_offset - begin_offset);
#if defined(_WIN32)
  _close(fd);
#else
  close(fd);
#endif
  LITERT_ASSIGN_OR_RETURN(litert::Model loaded_model, model);
  return loaded_model;
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesSections::Create(
    Config config) {
  return absl::WrapUnique(new ModelResourcesSections(std::move(config)));
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesSections::GetOrOpenModelScopedFile(ModelType model_type) {
  if (auto it = model_scoped_files_.find(model_type);
      it != model_scoped_files_.end() && it->second != nullptr) {
    return std::ref(*it->second);
  }
  auto section_it = config_.tflite_models.find(model_type);
  if (section_it == config_.tflite_models.end()) {
    return absl::NotFoundError(
        absl::StrCat("TFLite model section not found for model type: ",
                     ModelTypeToString(model_type)));
  }
  if (section_it->second.scoped_file != nullptr) {
    model_scoped_files_[model_type] = section_it->second.scoped_file;
    return std::ref(*section_it->second.scoped_file);
  }
  if (!section_it->second.path.empty()) {
    ABSL_ASSIGN_OR_RETURN(ScopedFile opened,
                          ScopedFile::Open(section_it->second.path));
    auto shared = std::make_shared<ScopedFile>(std::move(opened));
    model_scoped_files_[model_type] = shared;
    return std::ref(*shared);
  }
  return absl::NotFoundError(
      absl::StrCat("No ScopedFile or file path available for model type: ",
                   ModelTypeToString(model_type)));
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesSections::GetOrOpenWeightsScopedFile(ModelType model_type) {
  if (auto it = weights_scoped_files_.find(model_type);
      it != weights_scoped_files_.end() && it->second != nullptr) {
    return std::ref(*it->second);
  }
  auto section_it = config_.tflite_weights.find(model_type);
  if (section_it == config_.tflite_weights.end()) {
    return absl::NotFoundError(
        absl::StrCat("TFLite weights section not found for model type: ",
                     ModelTypeToString(model_type)));
  }
  if (section_it->second.scoped_file != nullptr) {
    weights_scoped_files_[model_type] = section_it->second.scoped_file;
    return std::ref(*section_it->second.scoped_file);
  }
  if (!section_it->second.path.empty()) {
    ABSL_ASSIGN_OR_RETURN(ScopedFile opened,
                          ScopedFile::Open(section_it->second.path));
    auto shared = std::make_shared<ScopedFile>(std::move(opened));
    weights_scoped_files_[model_type] = shared;
    return std::ref(*shared);
  }
  return absl::NotFoundError(
      absl::StrCat("No ScopedFile or file path available for weights of type: ",
                   ModelTypeToString(model_type)));
}

absl::StatusOr<std::shared_ptr<MemoryMappedFile>>
ModelResourcesSections::GetOrMapModelFile(ModelType model_type) {
  if (auto it = mapped_model_files_.find(model_type);
      it != mapped_model_files_.end() && it->second != nullptr) {
    return it->second;
  }
  auto section_it = config_.tflite_models.find(model_type);
  if (section_it == config_.tflite_models.end()) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  if (section_it->second.mapped_file != nullptr) {
    mapped_model_files_[model_type] = section_it->second.mapped_file;
    return section_it->second.mapped_file;
  }
  ABSL_ASSIGN_OR_RETURN(std::reference_wrapper<ScopedFile> scoped_file_ref,
                        GetOrOpenModelScopedFile(model_type));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mapped_file,
                        MemoryMappedFile::Create(scoped_file_ref.get().file()));
  std::shared_ptr<MemoryMappedFile> shared_mmf = std::move(mapped_file);
  mapped_model_files_[model_type] = shared_mmf;
  return shared_mmf;
}

absl::StatusOr<const litert::Model*> ModelResourcesSections::GetTFLiteModel(
    ModelType model_type) {
  if (auto it = model_map_.find(model_type); it != model_map_.end()) {
    return it->second.get();
  }

  if (!config_.tflite_models.contains(model_type)) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }

  if (config_.enable_file_backed_model_loading) {
    absl::StatusOr<std::reference_wrapper<ScopedFile>> scoped_file =
        GetOrOpenModelScopedFile(model_type);
    if (scoped_file.ok()) {
      absl::StatusOr<size_t> file_size =
          ScopedFile::GetSize(scoped_file->get().file());
      if (file_size.ok() && *file_size > 0) {
        absl::StatusOr<litert::Model> model_from_section =
            CreateModelFromFileSection(scoped_file->get(), 0, *file_size);
        if (!model_from_section.ok()) {
          if (model_from_section.status().code() !=
              absl::StatusCode::kUnimplemented) {
            return model_from_section.status();
          }
          ABSL_VLOG(1) << "File-backed LiteRT model loading is unsupported; "
                          "falling back to buffer-backed loading.";
        } else {
          auto& model = model_map_[model_type];
          model = std::make_unique<litert::Model>(
              std::move(model_from_section).value());
          return model.get();
        }
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<MemoryMappedFile> mmf,
                        GetOrMapModelFile(model_type));
  if (mmf->length() == 0) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  litert::BufferRef<uint8_t> buffer_ref(
      reinterpret_cast<const uint8_t*>(mmf->data()), mmf->length());
  LITERT_ASSIGN_OR_RETURN(litert::Model model,
                          litert::Model::CreateFromBuffer(buffer_ref));
  model_map_[model_type] = std::make_unique<litert::Model>(std::move(model));
  return model_map_[model_type].get();
}

absl::StatusOr<absl::string_view> ModelResourcesSections::GetTFLiteModelBuffer(
    ModelType model_type) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<MemoryMappedFile> mmf,
                        GetOrMapModelFile(model_type));
  if (mmf->length() == 0) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  return absl::string_view(reinterpret_cast<const char*>(mmf->data()),
                           mmf->length());
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesSections::GetScopedFile() {
  return GetScopedFile(ModelType::kTfLitePrefillDecode);
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesSections::GetScopedFile(ModelType model_type) {
  if (config_.tflite_weights.contains(model_type)) {
    return GetOrOpenWeightsScopedFile(model_type);
  }
  return GetOrOpenModelScopedFile(model_type);
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesSections::GetWeightsSectionOffset(ModelType model_type) {
  ABSL_ASSIGN_OR_RETURN(std::reference_wrapper<ScopedFile> scoped_file_ref,
                        GetOrOpenWeightsScopedFile(model_type));
  ABSL_ASSIGN_OR_RETURN(size_t file_size,
                        ScopedFile::GetSize(scoped_file_ref.get().file()));
  return std::make_pair(size_t{0}, file_size);
}

absl::StatusOr<FileRegion>
ModelResourcesSections::GetTFLiteModelSectionFileRegion(ModelType model_type) {
  auto section_it = config_.tflite_models.find(model_type);
  if (section_it == config_.tflite_models.end()) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  if (section_it->second.mapped_file != nullptr) {
    return FileRegion{
        .offset = 0,
        .size = section_it->second.mapped_file->length(),
    };
  }
  ABSL_ASSIGN_OR_RETURN(std::reference_wrapper<ScopedFile> scoped_file_ref,
                        GetOrOpenModelScopedFile(model_type));
  ABSL_ASSIGN_OR_RETURN(size_t file_size,
                        ScopedFile::GetSize(scoped_file_ref.get().file()));
  return FileRegion{
      .offset = 0,
      .size = file_size,
  };
}

std::optional<std::string>
ModelResourcesSections::GetTFLiteModelBackendConstraint(ModelType model_type) {
  auto it = config_.tflite_models.find(model_type);
  if (it == config_.tflite_models.end()) {
    return std::nullopt;
  }
  return it->second.backend_constraint;
}

std::optional<std::string>
ModelResourcesSections::GetTFLiteModelPreferActivationType(
    ModelType model_type) {
  auto it = config_.tflite_models.find(model_type);
  if (it == config_.tflite_models.end()) {
    return std::nullopt;
  }
  return it->second.prefer_activation_type;
}

absl::StatusOr<std::unique_ptr<Tokenizer>>
ModelResourcesSections::GetTokenizer() {
  return GetTokenizer(ModelType::kTfLitePrefillDecode);
}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResourcesSections::GetTokenizer(
    ModelType model_type) {
#if !defined(ENABLE_SENTENCEPIECE_TOKENIZER) && \
    !defined(ENABLE_HUGGINGFACE_TOKENIZER)
  return absl::UnimplementedError(
      "Tokenizers cannot be used. Neither ENABLE_SENTENCEPIECE_TOKENIZER nor "
      "ENABLE_HUGGINGFACE_TOKENIZER are defined during build.");
#endif  // !ENABLE_SENTENCEPIECE_TOKENIZER && !ENABLE_HUGGINGFACE_TOKENIZER

  auto it = config_.tokenizers.find(model_type);
  if (it == config_.tokenizers.end()) {
    return absl::NotFoundError(
        absl::StrCat("No tokenizer found in the model for model type: ",
                     ModelTypeToString(model_type)));
  }

  std::unique_ptr<MemoryMappedFile> mapped_tokenizer;
  absl::string_view buffer_view;
  if (!it->second.buffer.empty()) {
    buffer_view = it->second.buffer;
  } else if (!it->second.path.empty()) {
    ABSL_ASSIGN_OR_RETURN(ScopedFile scoped_file,
                          ScopedFile::Open(it->second.path));
    ABSL_ASSIGN_OR_RETURN(mapped_tokenizer,
                          MemoryMappedFile::Create(scoped_file.file()));
    buffer_view = absl::string_view(
        reinterpret_cast<const char*>(mapped_tokenizer->data()),
        mapped_tokenizer->length());
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Tokenizer section has neither buffer nor path for type: ",
                     ModelTypeToString(model_type)));
  }

  if (it->second.format == TokenizerFormat::kSentencePiece) {
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
    return SentencePieceTokenizer::CreateFromBuffer(buffer_view);
#else
    return absl::UnimplementedError(
        "SentencePiece tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_SENTENCEPIECE_TOKENIZER=1.");
#endif
  }

  if (it->second.format == TokenizerFormat::kHuggingFaceJson) {
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
    return HuggingFaceTokenizer::CreateFromJson(std::string(buffer_view));
#else
    return absl::UnimplementedError(
        "HuggingFace tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_HUGGINGFACE_TOKENIZER=1.");
#endif
  }

  return absl::InvalidArgumentError("Unsupported tokenizer format.");
}

absl::StatusOr<const Tokenizer*> ModelResourcesSections::GetOrCreateTokenizer(
    ModelType model_type) {
  if (auto it = tokenizer_map_.find(model_type); it != tokenizer_map_.end()) {
    return it->second.get();
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                        GetTokenizer(model_type));
  return tokenizer_map_.emplace(model_type, std::move(tokenizer))
      .first->second.get();
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesSections::GetLlmMetadata() {
  if (!config_.llm_metadata.has_value()) {
    return absl::NotFoundError("LlmMetadata not found in the model.");
  }
  return &config_.llm_metadata.value();
}

absl::StatusOr<const proto::ExecutorMetadata*>
ModelResourcesSections::GetExecutorMetadata() {
  if (!config_.executor_metadata.has_value()) {
    return absl::NotFoundError("ExecutorMetadata not found in the model.");
  }
  return &config_.executor_metadata.value();
}

absl::StatusOr<const proto::EmbeddingMetadata*>
ModelResourcesSections::GetEmbeddingMetadata() {
  if (!config_.embedding_metadata.has_value()) {
    return absl::NotFoundError("EmbeddingMetadata not found in the model.");
  }
  return &config_.embedding_metadata.value();
}

}  // namespace litert::lm
