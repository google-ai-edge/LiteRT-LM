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

#include "runtime/util/lora_data.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/util/lora_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

constexpr absl::string_view kLoRARank = "lora_rank";
// Map the entire file when building the FlatBuffer model. Passing 0 to
// MemoryMappedFileWithAutoAlignment::Create maps to end-of-file. This is
// needed for inline TFLite LoRA files where tensor data is part of the
// FlatBuffer (e.g. litert-torch to_tflite() output).
constexpr int kMetadataMaxSize = 0;

absl::StatusOr<std::unique_ptr<tflite::FlatBufferModel>>
CreateFlatBufferModelFromBuffer(const void* buffer_addr, size_t buffer_size) {
  const bool obfuscated = !tflite::ModelBufferHasIdentifier(buffer_addr);
  if (obfuscated) {
    return absl::UnimplementedError(
        "Input is not valid flatbuffer model. Deobfuscation is not supported "
        "yet.");
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          reinterpret_cast<const char*>(buffer_addr), buffer_size);
  RET_CHECK(model) << "Error building tflite model.";
  return model;
}

// LoRA data based on FlatBufferModel.
class FlatBufferLoraData : public LoraData {
 public:
  ~FlatBufferLoraData() override = default;

  absl::StatusOr<int> GetLoRARank() override {
    const tflite::Metadata* metadata = GetMetadata(kLoRARank);
    if (metadata == nullptr) {
      return absl::NotFoundError("No LoRA metadata found.");
    }
    return static_cast<int>(metadata->buffer());
  }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    const tflite::Buffer* buffer = GetBuffer(name);
    if (buffer == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("No buffer found for tensor: ", name));
    }
    // Inline TFLite files (e.g. litert-torch to_tflite() output) store tensor
    // data in buffer->data() instead of at an external offset. Check for inline
    // data first.
    if (buffer->data() && buffer->data()->size() > 0) {
      return std::make_unique<BufferRef<uint8_t>>(
          buffer->data()->data(), buffer->data()->size());
    }
    return ReadData(buffer->offset(), buffer->size());
  }

  bool HasTensor(absl::string_view name) const override {
    return GetBuffer(name) != nullptr;
  }

  std::vector<std::string> GetAllTensorNames() const override {
    std::vector<std::string> tensor_names;
    const tflite::Model* tflite_model = GetFlatBufferModel()->GetModel();
    for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
      for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
        tensor_names.push_back(tfl_tensor->name()->c_str());
      }
    }
    return tensor_names;
  }

 protected:
  // Returns the FlatBufferModel object reference.
  // FlatBufferModel is owned by derived classes to be destroyed in correct
  // order, thus it is accessed by base class with a reference here.
  virtual const tflite::FlatBufferModel* GetFlatBufferModel() const = 0;

  // Reads data stored at the given offset and size.
  virtual absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) = 0;

 private:
  // Get metadata from the flatbuffer model.
  const tflite::Metadata* GetMetadata(absl::string_view name) {
    const tflite::Model* tflite_model = GetFlatBufferModel()->GetModel();
    if (tflite_model->metadata() == nullptr) {
      return nullptr;
    }

    for (const tflite::Metadata* metadata : *tflite_model->metadata()) {
      if (name == metadata->name()->c_str()) {
        return metadata;
      }
    }
    return nullptr;
  }

  const tflite::Buffer* GetBuffer(absl::string_view name) const {
    const tflite::Model* tflite_model = GetFlatBufferModel()->GetModel();
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers =
        *tflite_model->buffers();
    for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
      for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
        if (name != tfl_tensor->name()->c_str()) {
          continue;
        }
        if (tfl_tensor->buffer() >= buffers.size()) {
          continue;
        }
        return buffers.Get(tfl_tensor->buffer());
      }
    }
    return nullptr;
  }
};

// FlatBufferModel based LoRA data backed by a file.
class FileLoraData : public FlatBufferLoraData {
 public:
  // Constructor for FileLoraData.
  //
  // @param file A shared_ptr to the ScopedFile object representing the LoRA
  // data file.
  // @param region A unique_ptr to the MemoryMappedFileWithAutoAlignment object
  // representing the memory mapped region of the FlatBufferModel metadata.
  // @param model A unique_ptr to the FlatBufferModel object representing the
  // LoRA data metadata.
  explicit FileLoraData(
      std::shared_ptr<const ScopedFile> file,
      std::unique_ptr<MemoryMappedFileWithAutoAlignment> region,
      std::unique_ptr<tflite::FlatBufferModel> model, const std::string& key)
      : file_(std::move(file)),
        region_(std::move(region)),
        model_(std::move(model)),
        key_(key) {}

  ~FileLoraData() override = default;

 private:
  const tflite::FlatBufferModel* GetFlatBufferModel() const override {
    return model_.get();
  }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    ASSIGN_OR_RETURN(auto mapped_region,
                     MemoryMappedFileWithAutoAlignment::Create(
                         file_->file(), /*offset=*/offset,
                         /*size=*/size, key_));
    return std::make_unique<MmapBufferRef<uint8_t>>(std::move(mapped_region));
  }

 private:
  std::shared_ptr<const ScopedFile> file_;
  std::unique_ptr<MemoryMappedFileWithAutoAlignment> region_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  const std::string key_;
};

// FlatBufferModel based LoRA data backed by a BufferRef.
class BufferLoraData : public FlatBufferLoraData {
 public:
  // Constructor for BufferLoraData.
  //
  // @param data A BufferRef object representing the LoRA data.
  // @param model A unique_ptr to the FlatBufferModel object representing the
  // LoRA data.
  explicit BufferLoraData(BufferRef<uint8_t> data,
                          std::unique_ptr<tflite::FlatBufferModel> model)
      : data_(std::move(data)), model_(std::move(model)) {}

  ~BufferLoraData() override = default;

 private:
  const tflite::FlatBufferModel* GetFlatBufferModel() const override {
    return model_.get();
  }

  absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    return std::make_unique<BufferRef<uint8_t>>(
        data_.Data(), /*end_offset=*/offset + size, /*start_offset=*/offset);
  }

 private:
  BufferRef<uint8_t> data_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
};

}  // namespace

// static
absl::StatusOr<std::unique_ptr<LoraData>> LoraData::CreateFromFilePath(
    absl::string_view file_path) {
  ASSIGN_OR_RETURN(auto file, ScopedFile::Open(file_path));
  return CreateFromScopedFile(std::make_shared<ScopedFile>(std::move(file)));
}

// static
absl::StatusOr<std::unique_ptr<LoraData>> LoraData::CreateFromScopedFile(
    std::shared_ptr<const ScopedFile> file) {
  static std::atomic<uint32_t> next_key{0};
  const std::string key{absl::StrCat("FileLoraData_", next_key.fetch_add(1))};
  ASSIGN_OR_RETURN(auto mapped_file, MemoryMappedFileWithAutoAlignment::Create(
                                         file->file(), /*offset=*/0,
                                         /*size=*/kMetadataMaxSize, key));
  ASSIGN_OR_RETURN(auto model, CreateFlatBufferModelFromBuffer(
                                   mapped_file->data(), mapped_file->length()));
  RET_CHECK(model) << "Error building tflite model.";
  return std::make_unique<FileLoraData>(std::move(file), std::move(mapped_file),
                                        std::move(model), key);
}

// static
absl::StatusOr<std::unique_ptr<LoraData>> LoraData::CreateFromBuffer(
    BufferRef<uint8_t> buffer) {
  ASSIGN_OR_RETURN(auto model, CreateFlatBufferModelFromBuffer(buffer.Data(),
                                                               buffer.Size()));
  RET_CHECK(model) << "Error building tflite model.";
  return std::make_unique<BufferLoraData>(std::move(buffer), std::move(model));
}

}  // namespace litert::lm
