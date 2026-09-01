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

#include "schema/capabilities/capabilities.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/status_macros.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm::schema::capabilities {
namespace {

// Maximum allowed size (10 MB) for serialized binary LlmMetadata protobuf
// sections. Protobuf metadata is typically a few KB; 10 MB provides ample
// headroom while guarding against OOM from corrupted offset values in
// malformed model files.
constexpr size_t kMaxProtoMetadataSize = 10 * 1024 * 1024;

// Maximum allowed size (10 MB) for reading NPU dispatch headers. The TFLite
// auxiliary model header containing DISPATCH_OP metadata is typically small.
constexpr size_t kMaxNpuSectionSize = 10 * 1024 * 1024;

// Maximum allowed size (2 GB) for vision sub-model FlatBuffers. In multimodal
// models, vision encoders typically range between 50 MB and 500 MB. 2 GB is the
// maximum buffer size addressable by the FlatBuffers specification (due to
// 32-bit signed offsets) and protects against unbounded memory allocations.
constexpr size_t kMaxVisionSectionSize = 2ULL * 1024 * 1024 * 1024;

// Formats a float for the CLI report. Uses general format '%g' to preserve
// precision without trailing zeros, but appends '.0' if the formatted string
// is parsed as a whole number (i.e. does not contain a decimal point or
// scientific notation) to explicitly signal that it is a float type.
std::string FormatFloatForReport(float val) {
  std::string s = absl::StrFormat("%g", val);
  if (!absl::StrContains(s, '.') && !absl::StrContains(s, 'e')) {
    absl::StrAppend(&s, ".0");
  }
  return s;
}

NpuBrand DetectNpuBrandFromTfliteBuffer(const char* data, size_t size) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data), size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return NpuBrand::kUnknown;
  }

  const tflite::Model* model = tflite::GetModel(data);
  if (model == nullptr) {
    return NpuBrand::kUnknown;
  }

  std::vector<int> dispatch_op_indices;
  if (model->operator_codes() != nullptr) {
    for (int i = 0; i < model->operator_codes()->size(); ++i) {
      const auto* op_code = model->operator_codes()->Get(i);
      if (op_code != nullptr && op_code->custom_code() != nullptr &&
          op_code->custom_code()->string_view() == "DISPATCH_OP") {
        dispatch_op_indices.push_back(i);
      }
    }
  }

  if (dispatch_op_indices.empty()) {
    return NpuBrand::kUnknown;
  }

  if (model->subgraphs() == nullptr) return NpuBrand::kUnknown;

  for (int s = 0; s < model->subgraphs()->size(); ++s) {
    const auto* subgraph = model->subgraphs()->Get(s);
    if (subgraph->operators() == nullptr) continue;

    for (int o = 0; o < subgraph->operators()->size(); ++o) {
      const auto* op = subgraph->operators()->Get(o);
      bool is_dispatch = false;
      for (int idx : dispatch_op_indices) {
        if (op->opcode_index() == idx) {
          is_dispatch = true;
          break;
        }
      }

      if (is_dispatch) {
        const auto* custom_options = op->custom_options();
        if (custom_options != nullptr && !custom_options->empty()) {
          auto root = flexbuffers::GetRoot(
              reinterpret_cast<const uint8_t*>(custom_options->data()),
              custom_options->size());
          if (root.IsMap()) {
            auto map = root.AsMap();
            auto name_val = map["name"];
            if (name_val.IsString()) {
              absl::string_view name(name_val.AsString().c_str(),
                                     name_val.AsString().length());
              if (absl::StrContains(name, "qnn")) {
                return NpuBrand::kQualcomm;
              } else if (absl::StrContains(name, "subgraph")) {
                return NpuBrand::kGoogleTensor;
              } else if (absl::StrContainsIgnoreCase(name, "partition")) {
                return NpuBrand::kMediaTek;
              }
            }
          }
        }
      }
    }
  }

  return NpuBrand::kUnknown;
}

bool IsMainLlmSection(absl::string_view model_type) {
  return model_type == "tf_lite_prefill_decode" ||
         model_type == "tf_lite_prefill_decode_hw" ||
         model_type == "tf_lite_prefill" ||
         model_type == "tf_lite_decode" ||
         model_type == "tf_lite_artisan_text_decoder";
}

// Determines hardware backend support (CPU, GPU, NPU) from backend constraint
// strings specified in the LiteRTLM container section headers (e.g. "cpu,gpu",
// "npu", "google_tensor_artisan").
//
// Backend Determination Logic:
// 1. If the constraint string is empty / omitted, the sub-model is assumed to
//    support both CPU and GPU execution by default.
// 2. If present, the constraint string is split by comma and tokens are
//    matched:
//    - "cpu" / "cpu_artisan" -> enables CPU backend.
//    - "gpu" / "gpu_artisan" -> enables GPU backend.
//    - "npu" / "google_tensor_artisan" -> enables NPU backend.
//    - "google_tensor_artisan" -> additionally flags the model as targeting
//      Google Tensor NPU.
void ParseBackendConstraint(absl::string_view constraint,
                            SupportedBackends& backends,
                            bool& is_artisan_tensor) {
  if (constraint.empty()) {
    // By default, models without explicit backend constraints support both
    // CPU & GPU.
    backends.cpu = true;
    backends.gpu = true;
    return;
  }
  for (auto b : absl::StrSplit(constraint, ',')) {
    b = absl::StripAsciiWhitespace(b);
    if (b == "cpu" || b == "cpu_artisan") {
      backends.cpu = true;
    } else if (b == "gpu" || b == "gpu_artisan") {
      backends.gpu = true;
    } else if (b == "npu" || b == "google_tensor_artisan") {
      backends.npu = true;
      if (b == "google_tensor_artisan") {
        is_artisan_tensor = true;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> ExtractVisionSignatureLengths(
    const char* data, size_t size) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data), size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return absl::InternalError(
        "Failed to verify vision model flatbuffer (corrupt model).");
  }

  const tflite::Model* model = tflite::GetModel(data);
  if (model == nullptr) {
    return absl::InternalError(
        "Failed to parse vision model flatbuffer (corrupt model).");
  }

  const auto* signature_defs = model->signature_defs();
  if (signature_defs == nullptr || signature_defs->empty()) {
    return std::vector<int>{};
  }

  std::vector<int> extracted_lengths;
  for (size_t sig_idx = 0; sig_idx < signature_defs->size(); ++sig_idx) {
    const auto* sig = signature_defs->Get(sig_idx);
    if (sig == nullptr) continue;

    const auto* outputs = sig->outputs();
    if (outputs == nullptr) continue;

    int output_tensor_index = -1;
    for (size_t out_idx = 0; out_idx < outputs->size(); ++out_idx) {
      const auto* output = outputs->Get(out_idx);
      if (output == nullptr) continue;
      if (outputs->size() == 1 ||
          (output->name() != nullptr &&
           output->name()->string_view() == "features")) {
        output_tensor_index = output->tensor_index();
        break;
      }
    }
    if (output_tensor_index == -1) {
      return absl::InternalError(
          "Failed to find output features tensor in vision signature.");
    }

    uint32_t subgraph_idx = sig->subgraph_index();
    const auto* subgraphs = model->subgraphs();
    if (subgraphs == nullptr || subgraph_idx >= subgraphs->size()) {
      return absl::InternalError("Invalid subgraph index in vision signature.");
    }

    const auto* subgraph = subgraphs->Get(subgraph_idx);
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
      return absl::InternalError("Corrupt subgraph in vision model.");
    }
    if (output_tensor_index >= subgraph->tensors()->size()) {
      return absl::InternalError(
          "Invalid output tensor index in vision subgraph.");
    }

    const auto* tensor = subgraph->tensors()->Get(output_tensor_index);
    if (tensor == nullptr || tensor->shape() == nullptr) {
      return absl::InternalError(
          "Missing output tensor or shape in vision model.");
    }

    const auto* shape = tensor->shape();
    if (shape->size() < 2) {
      return absl::InternalError(
          "Output features tensor has invalid rank (less than 2).");
    }

    int length = shape->Get(shape->size() - 2);
    extracted_lengths.push_back(length);
  }
  return extracted_lengths;
}

}  // namespace

absl::StatusOr<ModelCapabilities> InspectModel(std::istream& litertlm_stream) {
  litertlm_stream.seekg(0, std::ios::end);
  const std::streamoff total_stream_size = litertlm_stream.tellg();
  litertlm_stream.seekg(0, std::ios::beg);

  LitertlmHeader header;
  ABSL_RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_stream, &header));

  const LiteRTLMMetaData* metadata = header.metadata;
  RET_CHECK_NE(metadata, nullptr);

  ModelCapabilities info;

  // 1. Discover sections and identify modality models / speculative drafters.
  const litert::lm::schema::SectionMetadata* section_metadata_obj =
      metadata->section_metadata();
  RET_CHECK_NE(section_metadata_obj, nullptr);
  auto section_objects = section_metadata_obj->objects();
  RET_CHECK_NE(section_objects, nullptr);

  bool has_llm_metadata = false;
  uint64_t llm_metadata_begin = 0;
  uint64_t llm_metadata_end = 0;

  bool has_vision = false;
  bool has_audio = false;
  bool has_video = false;
  bool has_speculative_decoding = false;
  bool has_npu = false;
  uint64_t npu_section_begin = 0;
  uint64_t npu_section_end = 0;
  std::string text_backend_constraint;
  std::string vision_backend_constraint;
  std::string audio_backend_constraint;
  std::string video_backend_constraint;

  struct VisionSectionInfo {
    uint64_t begin;
    uint64_t end;
  };
  std::vector<VisionSectionInfo> vision_sections;

  for (size_t i = 0; i < section_objects->size(); ++i) {
    const auto* section = section_objects->Get(i);
    if (section == nullptr) continue;

    if (section->data_type() == AnySectionDataType_LlmMetadataProto) {
      // LLM Metadata Protobuf section found: record its stream offsets.
      has_llm_metadata = true;
      llm_metadata_begin = section->begin_offset();
      llm_metadata_end = section->end_offset();
    } else if (section->data_type() == AnySectionDataType_TFLiteModel) {
      // Scan TFLite model attributes to detect media and speculative features.
      bool is_vision = false;
      std::string model_type;
      std::string backend_constraint;
      if (const auto* items = section->items()) {
        for (size_t j = 0; j < items->size(); ++j) {
          const KeyValuePair* item = items->Get(j);
          if (item == nullptr || item->key() == nullptr) continue;
          if (item->key()->string_view() == "model_type") {
            const auto* value = item->value_as_StringValue();
            if (value && value->value()) {
              model_type = absl::AsciiStrToLower(value->value()->string_view());
            }
          } else if (item->key()->string_view() == "backend_constraint") {
            const auto* value = item->value_as_StringValue();
            if (value && value->value()) {
              backend_constraint = absl::AsciiStrToLower(
                  value->value()->string_view());
            }
          }
        }
      }
      if (!model_type.empty()) {
        if (model_type == "tf_lite_vision_adapter" ||
            model_type == "tf_lite_vision_encoder") {
          has_vision = true;
          is_vision = true;
        } else if (model_type == "tf_lite_audio_adapter" ||
                   model_type == "tf_lite_audio_encoder_hw") {
          has_audio = true;
        } else if (model_type == "tf_lite_video_adapter" ||
                   model_type == "tf_lite_video_encoder") {
          has_video = true;
        } else if (model_type == "tf_lite_mtp_drafter") {
          has_speculative_decoding = true;
        } else if (model_type == "tf_lite_aux") {
          has_npu = true;
          npu_section_begin = section->begin_offset();
          npu_section_end = section->end_offset();
        }

        if (IsMainLlmSection(model_type)) {
          text_backend_constraint = backend_constraint;
        } else if (model_type == "tf_lite_vision_adapter" ||
                   model_type == "tf_lite_vision_encoder") {
          vision_backend_constraint = backend_constraint;
        } else if (model_type == "tf_lite_audio_adapter" ||
                   model_type == "tf_lite_audio_encoder_hw") {
          audio_backend_constraint = backend_constraint;
        } else if (model_type == "tf_lite_video_adapter" ||
                   model_type == "tf_lite_video_encoder") {
          video_backend_constraint = backend_constraint;
        }
      }
      if (is_vision) {
        vision_sections.push_back(
            {section->begin_offset(), section->end_offset()});
      }
    }
  }

  // 2. Populate LlmInferenceCapability.
  LlmInferenceCapability llm_cap;
  llm_cap.input_modalities.text = true;
  llm_cap.input_modalities.vision = has_vision;
  llm_cap.input_modalities.audio = has_audio;
  llm_cap.input_modalities.video = has_video;

  llm_cap.output_modalities.text = true;
  llm_cap.supports_speculative_decoding = has_speculative_decoding;

  NpuBrand npu_brand = NpuBrand::kUnknown;
  bool is_artisan_tensor = false;

  // Resolve backend support for each active modality from section constraints.
  ParseBackendConstraint(text_backend_constraint,
                         llm_cap.text_supported_backends, is_artisan_tensor);
  if (has_vision) {
    ParseBackendConstraint(vision_backend_constraint,
                           llm_cap.vision_supported_backends,
                           is_artisan_tensor);
  }
  if (has_audio) {
    ParseBackendConstraint(audio_backend_constraint,
                           llm_cap.audio_supported_backends, is_artisan_tensor);
  }
  if (has_video) {
    ParseBackendConstraint(video_backend_constraint,
                           llm_cap.video_supported_backends, is_artisan_tensor);
  }

  // If a hardware-accelerated NPU sub-model section exists, enable NPU.
  if (has_npu) {
    llm_cap.text_supported_backends.npu = true;
  }

  // Detect specific NPU hardware brand if any modality has NPU enabled.
  bool has_any_npu = llm_cap.text_supported_backends.npu ||
                     llm_cap.vision_supported_backends.npu ||
                     llm_cap.audio_supported_backends.npu ||
                     llm_cap.video_supported_backends.npu;

  if (has_any_npu && npu_brand == NpuBrand::kUnknown) {
    if (is_artisan_tensor) {
      npu_brand = NpuBrand::kGoogleTensor;
    } else if (npu_section_end > npu_section_begin) {
      if (total_stream_size < 0 ||
          static_cast<uint64_t>(total_stream_size) >= npu_section_end) {
        size_t size = npu_section_end - npu_section_begin;
        if (size <= kMaxNpuSectionSize) {
          litertlm_stream.seekg(npu_section_begin);
          std::vector<char> npu_buffer(size);
          litertlm_stream.read(npu_buffer.data(), size);
          if (litertlm_stream) {
            npu_brand = DetectNpuBrandFromTfliteBuffer(npu_buffer.data(), size);
          }
        }
      }
    }
  }

  // Propagate detected NPU brand to all modalities that support NPU.
  if (llm_cap.text_supported_backends.npu) {
    llm_cap.text_supported_backends.npu_brand = npu_brand;
  }
  if (llm_cap.vision_supported_backends.npu) {
    llm_cap.vision_supported_backends.npu_brand = npu_brand;
  }
  if (llm_cap.audio_supported_backends.npu) {
    llm_cap.audio_supported_backends.npu_brand = npu_brand;
  }
  if (llm_cap.video_supported_backends.npu) {
    llm_cap.video_supported_backends.npu_brand = npu_brand;
  }

  // Overwrite from serialized binary LlmMetadata protobuf if present.
  if (has_llm_metadata && llm_metadata_end > llm_metadata_begin) {
    if (total_stream_size >= 0 &&
        static_cast<uint64_t>(total_stream_size) < llm_metadata_end) {
      return absl::InternalError("LLM metadata section exceeds stream bounds.");
    }
    size_t size = llm_metadata_end - llm_metadata_begin;
    if (size > kMaxProtoMetadataSize) {
      return absl::InternalError(
          "LLM metadata section size exceeds maximum allowed limit.");
    }
    litertlm_stream.seekg(llm_metadata_begin);
    auto buffer = std::make_unique<char[]>(size);
    litertlm_stream.read(buffer.get(), size);
    if (litertlm_stream) {
      proto::LlmMetadata proto_metadata;
      if (proto_metadata.ParseFromString(
              absl::string_view(buffer.get(), size))) {
        llm_cap.supports_thinking = proto_metadata.supports_thinking();
        llm_cap.min_runtime_version = proto_metadata.min_runtime_version();
        llm_cap.supports_function_calling =
            proto_metadata.supports_function_calling();
        if (proto_metadata.has_sampler_params()) {
          const auto& sp = proto_metadata.sampler_params();
          llm_cap.default_sampler_params.type =
              static_cast<SamplerType>(sp.type());
          llm_cap.default_sampler_params.k = sp.k();
          llm_cap.default_sampler_params.p = sp.p();
          llm_cap.default_sampler_params.temperature = sp.temperature();
        }
        if (proto_metadata.has_llm_model_type()) {
          const auto& model_type = proto_metadata.llm_model_type();
          int max_num_patches = 0;
          // Default to 3, which is the standard default for embedding gemma v2
          // and gemma4.
          int pooling_kernel_size = 3;
          if (model_type.has_gemma4()) {
            max_num_patches = model_type.gemma4().max_num_patches();
            if (model_type.gemma4().pooling_kernel_size() > 0) {
              pooling_kernel_size = model_type.gemma4().pooling_kernel_size();
            }
          } else if (model_type.has_generic_model()) {
            max_num_patches = model_type.generic_model().max_num_patches();
            pooling_kernel_size =
                model_type.generic_model().pooling_kernel_size() > 0
                    ? model_type.generic_model().pooling_kernel_size()
                    : 1;
          } else if (model_type.has_lfm2()) {
            max_num_patches = model_type.lfm2().max_num_patches();
            pooling_kernel_size = model_type.lfm2().pooling_kernel_size() > 0
                                      ? model_type.lfm2().pooling_kernel_size()
                                      : 2;
          }
          if (max_num_patches > 0) {
            llm_cap.max_vision_token_budget =
                max_num_patches / (pooling_kernel_size * pooling_kernel_size);
          }
        }
      }
    }
  }
  if (has_vision) {
    llm_cap.vision_signature_selection = std::vector<int>();

    for (const auto& vision_sec : vision_sections) {
      if (vision_sec.end <= vision_sec.begin) {
        return absl::InternalError("Invalid vision section offsets.");
      }
      if (total_stream_size >= 0 &&
          static_cast<uint64_t>(total_stream_size) < vision_sec.end) {
        return absl::InternalError("Vision section exceeds stream bounds.");
      }
      size_t size = vision_sec.end - vision_sec.begin;
      if (size > kMaxVisionSectionSize) {
        return absl::InternalError(
            "Vision model section exceeds maximum allowed limit.");
      }
      litertlm_stream.seekg(vision_sec.begin);
      auto buffer = std::make_unique<char[]>(size);
      litertlm_stream.read(buffer.get(), size);
      if (!litertlm_stream) {
        return absl::InternalError(
            "Failed to read vision model section from stream.");
      }

      ABSL_ASSIGN_OR_RETURN(
          auto lengths,
          ExtractVisionSignatureLengths(buffer.get(), size));
      llm_cap.vision_signature_selection->insert(
          llm_cap.vision_signature_selection->end(), lengths.begin(),
          lengths.end());
    }

    if (llm_cap.vision_signature_selection->empty()) {
      llm_cap.vision_signature_selection = std::nullopt;
    } else {
      auto& lengths = *llm_cap.vision_signature_selection;
      std::sort(lengths.begin(), lengths.end());
      lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    }
  } else {
    llm_cap.vision_signature_selection = std::nullopt;
  }

  info.llm_capability = llm_cap;

  return info;
}

absl::StatusOr<ModelCapabilities> InspectModel(
    absl::string_view litertlm_path) {
  std::ifstream input_file_stream(std::string(litertlm_path), std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  return InspectModel(input_file_stream);
}

std::ostream& operator<<(std::ostream& os,
                         const SupportedModalities& modalities) {
  if (modalities.text) os << "Text ";
  if (modalities.vision) os << "Vision ";
  if (modalities.audio) os << "Audio ";
  if (modalities.video) os << "Video ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const NpuBrand& brand) {
  switch (brand) {
    case NpuBrand::kQualcomm:
      os << "Qualcomm QNN";
      break;
    case NpuBrand::kGoogleTensor:
      os << "Google Tensor TPU";
      break;
    case NpuBrand::kMediaTek:
      os << "MediaTek Neuron";
      break;
    default:
      os << "Unknown";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const SupportedBackends& backends) {
  if (backends.cpu) os << "CPU ";
  if (backends.gpu) os << "GPU ";
  if (backends.npu) {
    os << "NPU (" << backends.npu_brand << ") ";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const LlmInferenceCapability& llm_cap) {
  auto sampler_type_str = [](SamplerType type) {
    switch (type) {
      case SamplerType::kTopK:
        return "TOP_K";
      case SamplerType::kTopP:
        return "TOP_P";
      case SamplerType::kGreedy:
        return "GREEDY";
      default:
        return "UNSPECIFIED";
    }
  };

  os << "[LLM Capabilities]\n"
     << "  Supports Function Call: "
     << (llm_cap.supports_function_calling ? "YES" : "NO") << "\n"
     << "  Supports Thinking:      "
     << (llm_cap.supports_thinking ? "YES" : "NO") << "\n"
     << "  Speculative Decoding:   "
     << (llm_cap.supports_speculative_decoding ? "YES" : "NO") << "\n"
     << "  Max Vision Token Budget: "
     << llm_cap.max_vision_token_budget << "\n"
     << "  Min Runtime Version:    "
     << (llm_cap.min_runtime_version.empty() ? "-1"
                                             : llm_cap.min_runtime_version)
     << "\n";

  if (llm_cap.vision_signature_selection.has_value()) {
    os << "  Vision Signature Selection: [";
    const auto& lengths = *llm_cap.vision_signature_selection;
    for (size_t i = 0; i < lengths.size(); ++i) {
      os << lengths[i];
      if (i + 1 < lengths.size()) os << ", ";
    }
    os << "]\n";
  } else {
    os << "  Vision Signature Selection: -1\n";
  }

  os << "  Sampler Type:           "
     << sampler_type_str(llm_cap.default_sampler_params.type) << "\n"
     << "  Sampler Temp:           "
     << FormatFloatForReport(llm_cap.default_sampler_params.temperature) << "\n"
     << "  Sampler Top K:          " << llm_cap.default_sampler_params.k << "\n"
     << "  Sampler Top P:          "
     << FormatFloatForReport(llm_cap.default_sampler_params.p) << "\n"
     << "  Input Modalities:       " << llm_cap.input_modalities << "\n";

  if (llm_cap.input_modalities.text) {
    os << "  Text Backends:          " << llm_cap.text_supported_backends
       << "\n";
  }
  if (llm_cap.input_modalities.vision) {
    os << "  Vision Backends:        " << llm_cap.vision_supported_backends
       << "\n";
  }
  if (llm_cap.input_modalities.audio) {
    os << "  Audio Backends:         " << llm_cap.audio_supported_backends
       << "\n";
  }
  if (llm_cap.input_modalities.video) {
    os << "  Video Backends:         " << llm_cap.video_supported_backends
       << "\n";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const ModelCapabilities& capabilities) {
  if (capabilities.llm_capability.has_value()) {
    os << *capabilities.llm_capability;
  } else {
    os << "[LLM Capabilities]\n  <none>\n";
  }
  return os;
}

}  // namespace litert::lm::schema::capabilities
