// Copyright 2024 The ODML Authors.
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

#include "runtime/executor/gpu_artisan/utils/model_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/metadata_utils.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace odml::infra::llm_utils {
namespace {

using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ScopedFile;

// The maximum size of the tflite::Model (excluding buffers).
constexpr uint64_t kTfliteBaseSize = 1024 * 1024;

class SpanHolder : public DataHolder<uint8_t> {
 public:
  explicit SpanHolder(absl::Span<uint8_t> data) : data_(data) {}

  absl::Span<uint8_t> GetData() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
};

class FreeingSpanHolder : public DataHolder<uint8_t> {
 public:
  explicit FreeingSpanHolder(absl::Span<uint8_t> data) : data_(data) {}
  ~FreeingSpanHolder() override { free(data_.data()); }

  absl::Span<uint8_t> GetData() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
};

// Base class for loading models from a tflite file.
class TfliteModelData : public ModelData {
 public:
  explicit TfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model)
      : model_(std::move(model)) {}
  ~TfliteModelData() override = default;

  std::optional<proto::LlmModelType> GetModelType() override {
    const tflite::Metadata* metadata = GetMetadata(kLlmModelTypeName);
    if (metadata == nullptr) {
      return std::nullopt;
    }
    return static_cast<proto::LlmModelType>(metadata->buffer());
  }

  std::optional<int> LoRARank() override {
    const tflite::Metadata* metadata = GetMetadata(kLoRARank);
    if (metadata == nullptr) return std::nullopt;
    return static_cast<int>(metadata->buffer());
  }

  const proto::LlmParameters& GetLlmParameters() override {
    return llm_parameters_;
  }

  absl::StatusOr<std::string> ReadMetadata(absl::string_view name) override {
    const tflite::Metadata* metadata = GetMetadata(name);
    if (metadata == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("Failed to get metadata: ", name));
    }
    const tflite::Buffer* backend_buffer =
        model_->GetModel()->buffers()->Get(metadata->buffer());
    ASSIGN_OR_RETURN(
        auto data, ReadData(backend_buffer->offset(), backend_buffer->size()));
    return std::string(reinterpret_cast<const char*>(data->GetData().data()),
                       data->GetData().size());
  }

  uint64_t GetMaxTensorSize() const override {
    uint64_t max_size = 0;
    const tflite::Model* tflite_model = model_->GetModel();
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers =
        *tflite_model->buffers();
    for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
      for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
        if (tfl_tensor->buffer() >= buffers.size()) {
          continue;
        }
        max_size =
            std::max(max_size, buffers.Get(tfl_tensor->buffer())->size());
      }
    }
    return max_size;
  }

  uint64_t GetTensorSize(absl::string_view name) const override {
    const tflite::Buffer* buffer = GetBuffer(name);
    if (buffer) {
      return buffer->size();
    }
    return 0;
  }

  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    const tflite::Buffer* buffer = GetBuffer(name);
    if (buffer) {
      return ReadData(buffer->offset(), buffer->size());
    }
    return nullptr;
  }

  absl::StatusOr<ModelWithData> ReadModel(absl::string_view name) override {
    ASSIGN_OR_RETURN(auto data, ReadTensor(name));
    if (!data) {
      return ModelWithData{};
    }
    auto model = tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(data->GetData().data()),
        data->GetData().size());
    return ModelWithData{
        .model = std::move(model),
        .data = std::move(data),
    };
  }

  absl::Status InitLlmParameters() {
    ASSIGN_OR_RETURN(std::string proto_str,
                     ReadMetadata("odml.infra.proto.LlmParameters"));
    RET_CHECK(llm_parameters_.ParseFromString(proto_str));
    return absl::OkStatus();
  }

 protected:
  virtual absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) = 0;

  std::shared_ptr<tflite::FlatBufferModel> model_;
  proto::LlmParameters llm_parameters_;

 private:
  const tflite::Buffer* GetBuffer(absl::string_view name) const {
    const tflite::Model* tflite_model = model_->GetModel();
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

  const tflite::Metadata* GetMetadata(absl::string_view name) {
    const tflite::Model* tflite_model = model_->GetModel();
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
};

// Loads from a tflite model which includes all buffers in the allocation.
class InMemoryTfliteModelData : public TfliteModelData {
 public:
  explicit InMemoryTfliteModelData(
      std::shared_ptr<tflite::FlatBufferModel> model)
      : TfliteModelData(std::move(model)) {}
  ~InMemoryTfliteModelData() override = default;

  void Clear() override {}

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    return std::make_unique<SpanHolder>(absl::MakeSpan(
        const_cast<uint8_t*>(
            static_cast<const uint8_t*>(model_->allocation()->base())) +
            offset,
        size));
  }
};

// Loads tflite data from a file as needed.
class FileTfliteModelData : public TfliteModelData {
 public:
  FileTfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model,
                      std::unique_ptr<DataHolder<const uint8_t>> model_data,
                      std::shared_ptr<const ScopedFile> file,
                      uint64_t base_offset = 0)
      : TfliteModelData(std::move(model)),
        file_(std::move(file)),
        model_data_(std::move(model_data)),
        base_offset_(base_offset) {}
  ~FileTfliteModelData() override = default;

  void SetMemoryMappedFileAccess(MemoryMappedFile::Access access) override {
    access_ = access;
  }

  void Clear() override {
    file_.reset();
    model_data_.reset();
  }

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    RET_CHECK(file_);
    return CreateMemoryMappedDataHolder<uint8_t>(
        file_->file(), base_offset_ + offset, size, key_, access_);
  }

 private:
  static uint32_t next_key_;
  const std::string key_{absl::StrCat("FileTfliteModelData_", next_key_++)};
  std::shared_ptr<const ScopedFile> file_;
  std::unique_ptr<DataHolder<const uint8_t>> model_data_;
  uint64_t base_offset_;
  MemoryMappedFile::Access access_ = MemoryMappedFile::Access::kCopy;
};

uint32_t FileTfliteModelData::next_key_ = 0;

// Loads tflite data from the provided function. This owns any data returned
// from the read data function.
class FunctionTfliteModelData : public TfliteModelData {
 public:
  FunctionTfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model,
                          ModelData::ReadDataFn fn)
      : TfliteModelData(std::move(model)), fn_(std::move(fn)) {}
  ~FunctionTfliteModelData() override { Clear(); }

  void Clear() override {
    free(const_cast<void*>(model_->allocation()->base()));
    fn_(0, 0, ReadMode::DISCARD_ALL);
  }

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    void* data = fn_(offset, size, ReadMode::DISCARD);
    RET_CHECK(data) << "Error fetching data.";
    return std::make_unique<FreeingSpanHolder>(
        absl::MakeSpan(static_cast<uint8_t*>(data), size));
  }

 private:
  ModelData::ReadDataFn fn_;
};

uint64_t AlignByN(uint64_t number, uint64_t n) {
  const uint64_t q = number / n;
  return (number % n == 0 ? q : q + 1) * n;
}

}  // namespace

OffsetAndSize GetAlignedOffsetAndSize(uint64_t base_offset,
                                      uint64_t base_size) {
  const size_t kAlignment = MemoryMappedFile::GetOffsetAlignment();
  uint64_t offset = (base_offset / kAlignment) * kAlignment;
  uint64_t size = AlignByN(base_offset - offset + base_size, kAlignment);
  return {.offset = offset, .size = size};
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    std::shared_ptr<tflite::FlatBufferModel> model) {
  auto model_data = std::make_shared<InMemoryTfliteModelData>(std::move(model));
  RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(ScopedFile file) {
  return Create(std::make_shared<ScopedFile>(std::move(file)));
}

absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    std::shared_ptr<const ScopedFile> file) {
  return Create(file, /*offset=*/0, /*size=*/kTfliteBaseSize);
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    std::shared_ptr<const ScopedFile> file, uint64_t offset, uint64_t size) {
  // Load the first chunk of the section as a tflite model, and load the rest
  // on-demand when needed.
  uint64_t base_size = std::min(size, kTfliteBaseSize);
  ASSIGN_OR_RETURN(auto data, CreateMemoryMappedDataHolder<const uint8_t>(
                                  file->file(), offset, base_size));
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(data->GetData().data()),
      data->GetData().size());
  RET_CHECK(model) << "Error building tflite model.";
  auto model_data = std::make_shared<FileTfliteModelData>(
      std::move(model), std::move(data),
      std::move(file), offset);
  RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(ReadDataFn fn) {
  // Load the first chunk of the file as a tflite model, and load the rest
  // on-demand when needed.
  void* data = fn(0, kTfliteBaseSize, ReadMode::KEEP);
  RET_CHECK(data) << "Error fetching data.";
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(data), kTfliteBaseSize);
  RET_CHECK(model) << "Error building tflite model.";
  auto model_data = std::make_shared<FunctionTfliteModelData>(std::move(model),
                                                              std::move(fn));
  RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    absl::string_view weight_path) {
  ASSIGN_OR_RETURN(auto tflite_file, ScopedFile::Open(weight_path));
  return ModelData::Create(std::move(tflite_file));
}

}  // namespace odml::infra::llm_utils
