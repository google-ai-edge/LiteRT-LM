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

// Information extracted from NPU models and stamps.
struct NpuInfo {
  // Detected NPU hardware brand (e.g. Qualcomm, Google Tensor, MediaTek).
  NpuBrand brand = NpuBrand::kUnknown;
  // Specific NPU SoC / chipset identifier (e.g. "SM8850", "Tensor_G5").
  std::string soc_name;
};

// Specification of known NPU hardware brand matching rules and display names.
struct NpuBrandSpec {
  NpuBrand brand;
  absl::string_view display_name;
  // Keywords matched against LiteRtStamp manufacturer field.
  std::vector<absl::string_view> stamp_mfg_tokens;
  // Keywords matched against DISPATCH_OP operator names in flexbuffers.
  std::vector<absl::string_view> dispatch_op_tokens;
};

// Returns the authoritative registry of supported NPU brands and their matching
// tokens.
const std::vector<NpuBrandSpec>& GetNpuBrandRegistry() {
  static const auto* const kRegistry = new std::vector<NpuBrandSpec>{
      {NpuBrand::kQualcomm, "Qualcomm QNN", {"qualcomm", "qti"}, {"qnn"}},
      {NpuBrand::kGoogleTensor, "Google Tensor TPU", {"google"}, {"subgraph"}},
      {NpuBrand::kIntel,
       "Intel NPU",
       {"intel", "openvino", "intelopenvino"},
       {"openvino", "intel"}},
      {NpuBrand::kSamsung,
       "Samsung Exynos NPU",
       {"samsung", "exynos", "slsi"},
       {"samsung", "exynos", "slsi"}},
      {NpuBrand::kMediaTek,
       "MediaTek Neuron",
       {"mediatek", "mtk"},
       {"mtk", "neuron", "mediatek", "partition"}},
  };
  return *kRegistry;
}

// Returns true if the key represents an SoC name or model attribute.
bool IsSocNameKey(absl::string_view key) {
  static constexpr absl::string_view kSocKeys[] = {
      "soc_name", "soc_model", "target_soc", "soc", "chipset", "target_device",
  };
  for (absl::string_view candidate : kSocKeys) {
    if (absl::EqualsIgnoreCase(key, candidate)) return true;
  }
  return false;
}

// Parses a 250-byte LiteRtStamp payload buffer into NpuInfo (brand and SoC
// name). The stamp payload format consists of:
// - Bytes 0..124: Manufacturer / brand string (null-terminated if < 125 bytes).
// - Bytes 125..249: SoC name / chipset string (null-terminated if
//   < 125 bytes).
void ParseLiteRtStampPayload(absl::string_view stamp_payload, NpuInfo& info) {
  if (stamp_payload.size() < 250) {
    return;
  }
  absl::string_view mfg_raw = stamp_payload.substr(0, 125);
  size_t mfg_null = mfg_raw.find('\0');
  absl::string_view mfg = (mfg_null != absl::string_view::npos)
                              ? mfg_raw.substr(0, mfg_null)
                              : mfg_raw;

  absl::string_view model_raw = stamp_payload.substr(125, 125);
  size_t model_null = model_raw.find('\0');
  absl::string_view model_str = (model_null != absl::string_view::npos)
                                    ? model_raw.substr(0, model_null)
                                    : model_raw;

  if (info.brand == NpuBrand::kUnknown) {
    for (const auto& spec : GetNpuBrandRegistry()) {
      for (absl::string_view token : spec.stamp_mfg_tokens) {
        if (absl::StrContainsIgnoreCase(mfg, token)) {
          info.brand = spec.brand;
          break;
        }
      }
      if (info.brand != NpuBrand::kUnknown) break;
    }
  }
  if (!model_str.empty() && info.soc_name.empty() &&
      info.brand != NpuBrand::kUnknown) {
    info.soc_name = std::string(model_str);
  }
}

// Inspects a TFLite model buffer to detect target NPU brand and SoC model.
// Checks:
// 1. LiteRtStamp metadata buffer embedded in the TFLite model.
// 2. DISPATCH_OP custom options (flexbuffers) for runtime acceleration info.
// 3. Fallback scan for LiteRtStamp string and 250-byte stamp in the buffer.
NpuInfo DetectNpuInfoFromTfliteBuffer(absl::string_view tflite_buffer) {
  NpuInfo info;
  if (tflite_buffer.size() < kMaxVisionSectionSize) {
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t*>(tflite_buffer.data()),
        tflite_buffer.size());
    if (tflite::VerifyModelBuffer(verifier)) {
      const tflite::Model* model = tflite::GetModel(tflite_buffer.data());
      if (model != nullptr) {
        // 1. Inspect LiteRtStamp in TFLite metadata if present.
        if (model->metadata() != nullptr && model->buffers() != nullptr) {
          for (size_t i = 0; i < model->metadata()->size(); ++i) {
            const auto* meta = model->metadata()->Get(i);
            if (meta != nullptr && meta->name() != nullptr &&
                meta->name()->string_view() == "LiteRtStamp") {
              uint32_t buf_idx = meta->buffer();
              if (buf_idx < model->buffers()->size()) {
                const auto* buf = model->buffers()->Get(buf_idx);
                if (buf != nullptr && buf->data() != nullptr) {
                  const auto* data_vec = buf->data();
                  if (data_vec->size() >= 250) {
                    ParseLiteRtStampPayload(
                        absl::string_view(
                            reinterpret_cast<const char*>(data_vec->data()),
                            data_vec->size()),
                        info);
                  }
                }
              }
            }
          }
        }

        // 2. Inspect DISPATCH_OP operators in subgraphs.
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

        if (!dispatch_op_indices.empty() && model->subgraphs() != nullptr) {
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
                    if (name_val.IsString() &&
                        info.brand == NpuBrand::kUnknown) {
                      absl::string_view name(name_val.AsString().c_str(),
                                             name_val.AsString().length());
                      for (const auto& spec : GetNpuBrandRegistry()) {
                        for (absl::string_view token :
                             spec.dispatch_op_tokens) {
                          if (absl::StrContainsIgnoreCase(name, token)) {
                            info.brand = spec.brand;
                            break;
                          }
                        }
                        if (info.brand != NpuBrand::kUnknown) break;
                      }
                    }
                    auto soc_val = map["soc_name"];
                    if (!soc_val.IsString()) {
                      soc_val = map["soc_model"];
                    }
                    if (soc_val.IsString() && info.soc_name.empty()) {
                      info.soc_name = std::string(soc_val.AsString().c_str(),
                                                  soc_val.AsString().length());
                    }
                    auto target_soc_val = map["target_soc"];
                    if (target_soc_val.IsString() && info.soc_name.empty()) {
                      info.soc_name =
                          std::string(target_soc_val.AsString().c_str(),
                                      target_soc_val.AsString().length());
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 3. Fallback scan for LiteRtStamp string and 250-byte stamp in the buffer.
  if (info.brand == NpuBrand::kUnknown || info.soc_name.empty()) {
    size_t stamp_pos = tflite_buffer.find("LiteRtStamp");
    if (stamp_pos != absl::string_view::npos) {
      size_t search_start = stamp_pos > 4096 ? stamp_pos - 4096 : 0;
      size_t search_end = std::min(tflite_buffer.size(), stamp_pos + 4096);
      absl::string_view window =
          tflite_buffer.substr(search_start, search_end - search_start);
      // Flatbuffer byte vector for 250 bytes has 32-bit little-endian length
      // 250: 0xfa 0x00 0x00 0x00
      static constexpr char kLenPrefix[4] = {'\xfa', '\x00', '\x00', '\x00'};
      size_t prefix_pos = window.find(absl::string_view(kLenPrefix, 4));
      if (prefix_pos != absl::string_view::npos &&
          prefix_pos + 4 + 250 <= window.size()) {
        ParseLiteRtStampPayload(window.substr(prefix_pos + 4, 250), info);
      }
    }
  }

  return info;
}

// Returns true if the given model type corresponds to a primary LLM text
// compute section (e.g. prefill, decode, artisan text decoder).
bool IsMainLlmSection(absl::string_view model_type) {
  return model_type == "tf_lite_prefill_decode" ||
         model_type == "tf_lite_prefill_decode_hw" ||
         model_type == "tf_lite_prefill" ||
         model_type == "tf_lite_decode" ||
         model_type == "tf_lite_artisan_text_decoder";
}

// Determines hardware backend support (CPU, GPU, NPU) and priority ordering
// from backend constraint strings specified in the LiteRTLM container section
// headers (e.g. "cpu,gpu", "gpu,cpu", "npu", "google_tensor_artisan").
//
// Backend Determination Logic:
// 1. If the constraint string is empty / omitted, the sub-model is assumed to
//    support both CPU and GPU execution by default, with CPU as the default
//    backend.
// 2. If present, the constraint string is split by comma and tokens are
//    matched in order:
//    - "cpu" / "cpu_artisan" -> enables CPU backend.
//    - "gpu" / "gpu_artisan" -> enables GPU backend.
//    - "npu" / "google_tensor_artisan" -> enables NPU backend.
//    - "google_tensor_artisan" -> additionally flags the model as targeting
//      Google Tensor NPU.
//    The first valid backend token specified defines the default_backend.
void ParseBackendConstraint(absl::string_view constraint,
                            SupportedBackends& backends,
                            bool& is_artisan_tensor) {
  if (constraint.empty()) {
    // By default, models without explicit backend constraints support both
    // CPU & GPU. Default backend is CPU.
    backends.cpu = true;
    backends.gpu = true;
    backends.default_backend = BackendType::kCpu;
    backends.preferred_backends = {BackendType::kCpu, BackendType::kGpu};
    return;
  }
  for (auto b : absl::StrSplit(constraint, ',')) {
    b = absl::StripAsciiWhitespace(b);
    if (b == "cpu" || b == "cpu_artisan") {
      backends.cpu = true;
      if (std::find(backends.preferred_backends.begin(),
                    backends.preferred_backends.end(),
                    BackendType::kCpu) == backends.preferred_backends.end()) {
        backends.preferred_backends.push_back(BackendType::kCpu);
      }
    } else if (b == "gpu" || b == "gpu_artisan") {
      backends.gpu = true;
      if (std::find(backends.preferred_backends.begin(),
                    backends.preferred_backends.end(),
                    BackendType::kGpu) == backends.preferred_backends.end()) {
        backends.preferred_backends.push_back(BackendType::kGpu);
      }
    } else if (b == "npu" || b == "google_tensor_artisan") {
      backends.npu = true;
      if (std::find(backends.preferred_backends.begin(),
                    backends.preferred_backends.end(),
                    BackendType::kNpu) == backends.preferred_backends.end()) {
        backends.preferred_backends.push_back(BackendType::kNpu);
      }
      if (b == "google_tensor_artisan") {
        is_artisan_tensor = true;
      }
    }
  }
  if (!backends.preferred_backends.empty()) {
    backends.default_backend = backends.preferred_backends.front();
  }
}

// Extracts discrete vision token signature lengths from a vision model TFLite
// flatbuffer by inspecting output tensor dimensions across signature defs.
absl::StatusOr<std::vector<int>> ExtractVisionSignatureLengths(
    absl::string_view vision_buffer) {
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(vision_buffer.data()),
      vision_buffer.size());
  if (!tflite::VerifyModelBuffer(verifier)) {
    return absl::InternalError(
        "Failed to verify vision model flatbuffer (corrupt model).");
  }

  const tflite::Model* model = tflite::GetModel(vision_buffer.data());
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

// Check if the number is a magic number.
// The number is a magic number if it is prime and greater than 10.
bool IsMagicNumber(int64_t number) {
  if (number < 11) {
    return false;
  }
  if (number % 2 == 0) {
    return false;
  }
  for (int64_t i = 3; i * i <= number; i += 2) {
    if (number % i == 0) {
      return false;
    }
  }
  return true;
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

  bool found_main_tflite = false;
  uint64_t main_tflite_begin = 0;
  uint64_t main_tflite_end = 0;

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

  std::string text_soc_name;
  std::string vision_soc_name;
  std::string audio_soc_name;
  std::string video_soc_name;
  std::string global_soc_name;

  if (const auto* sys_meta = metadata->system_metadata()) {
    if (const auto* entries = sys_meta->entries()) {
      for (size_t j = 0; j < entries->size(); ++j) {
        const KeyValuePair* item = entries->Get(j);
        if (item == nullptr || item->key() == nullptr) continue;
        absl::string_view key = item->key()->string_view();
        if (key == "soc_name" || key == "soc_model" || key == "target_soc" ||
            key == "soc") {
          const auto* value = item->value_as_StringValue();
          if (value && value->value() && global_soc_name.empty()) {
            global_soc_name = std::string(value->value()->string_view());
          }
        }
      }
    }
  }

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
      std::string section_soc_name;
      // In this context, "is_adapter" refers to any auxiliary model (like
      // vision/audio adapters, encoders, or speculative drafters) that is not
      // the main compute pipeline. We skip these when searching for the main
      // TFLite model.
      bool is_adapter = false;
      if (const auto* items = section->items()) {
        for (size_t j = 0; j < items->size(); ++j) {
          const KeyValuePair* item = items->Get(j);
          if (item == nullptr || item->key() == nullptr) continue;
          absl::string_view key = item->key()->string_view();
          if (key == "model_type") {
            const auto* value = item->value_as_StringValue();

            if (value && value->value()) {
              model_type = absl::AsciiStrToLower(value->value()->string_view());
            }
          } else if (key == "backend_constraint") {
            const auto* value = item->value_as_StringValue();
            if (value && value->value()) {
              backend_constraint = absl::AsciiStrToLower(
                  value->value()->string_view());
            }
          } else if (IsSocNameKey(key)) {
            const auto* value = item->value_as_StringValue();
            if (value && value->value()) {
              section_soc_name = std::string(value->value()->string_view());
              if (global_soc_name.empty()) {
                global_soc_name = section_soc_name;
              }
            }
          }
        }
      }

      // Classify the sub-model section by its model_type to identify supported
      // input modalities (vision, audio, video), speculative decoding drafters,
      // and hardware-accelerated NPU sub-models.
      if (!model_type.empty()) {
        if (model_type == "tf_lite_vision_adapter" ||
            model_type == "tf_lite_vision_encoder") {
          has_vision = true;
          is_vision = true;
          is_adapter = true;
        } else if (model_type == "tf_lite_audio_adapter" ||
                   model_type == "tf_lite_audio_encoder_hw") {
          has_audio = true;
          is_adapter = true;
        } else if (model_type == "tf_lite_video_adapter" ||
                   model_type == "tf_lite_video_encoder") {
          has_video = true;
          is_adapter = true;
        } else if (model_type == "tf_lite_mtp_drafter") {
          has_speculative_decoding = true;
          is_adapter = true;
        } else if (model_type == "tf_lite_aux") {
          has_npu = true;
          npu_section_begin = section->begin_offset();
          npu_section_end = section->end_offset();
        }

        // Associate backend constraints and SoC models with their respective
        // modality.
        if (IsMainLlmSection(model_type)) {
          text_backend_constraint = backend_constraint;
          if (!section_soc_name.empty()) text_soc_name = section_soc_name;
        } else if (model_type == "tf_lite_vision_adapter" ||
                   model_type == "tf_lite_vision_encoder") {
          vision_backend_constraint = backend_constraint;
          if (!section_soc_name.empty()) vision_soc_name = section_soc_name;
        } else if (model_type == "tf_lite_audio_adapter" ||
                   model_type == "tf_lite_audio_encoder_hw") {
          audio_backend_constraint = backend_constraint;
          if (!section_soc_name.empty()) audio_soc_name = section_soc_name;
        } else if (model_type == "tf_lite_video_adapter" ||
                   model_type == "tf_lite_video_encoder") {
          video_backend_constraint = backend_constraint;
          if (!section_soc_name.empty()) video_soc_name = section_soc_name;
        }
      }
      if (is_vision) {
        vision_sections.push_back(
            {section->begin_offset(), section->end_offset()});
      }

      // Identify the main TFLite graph section for fallback NPU/SoC inspection.
      if (IsMainLlmSection(model_type)) {
        main_tflite_begin = section->begin_offset();
        main_tflite_end = section->end_offset();
        found_main_tflite = true;
      } else if (!is_adapter && !found_main_tflite) {
        main_tflite_begin = section->begin_offset();
        main_tflite_end = section->end_offset();
        found_main_tflite = true;
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
    if (std::find(llm_cap.text_supported_backends.preferred_backends.begin(),
                  llm_cap.text_supported_backends.preferred_backends.end(),
                  BackendType::kNpu) ==
        llm_cap.text_supported_backends.preferred_backends.end()) {
      llm_cap.text_supported_backends.preferred_backends.push_back(
          BackendType::kNpu);
    }
    if (llm_cap.text_supported_backends.default_backend ==
        BackendType::kUnspecified) {
      llm_cap.text_supported_backends.default_backend = BackendType::kNpu;
    }
  }

  // Detect specific NPU hardware brand and SoC if any modality has NPU enabled.
  bool has_any_npu = llm_cap.text_supported_backends.npu ||
                     llm_cap.vision_supported_backends.npu ||
                     llm_cap.audio_supported_backends.npu ||
                     llm_cap.video_supported_backends.npu;

  auto inspect_section_for_npu = [&](uint64_t begin, uint64_t end) {
    if (end <= begin) return;
    if (total_stream_size >= 0 &&
        static_cast<uint64_t>(total_stream_size) < begin) {
      return;
    }
    size_t size = end - begin;
    size_t read_size = std::min(size, kMaxNpuSectionSize);
    litertlm_stream.seekg(begin);
    std::vector<char> buffer(read_size);
    litertlm_stream.read(buffer.data(), read_size);
    if (litertlm_stream || litertlm_stream.gcount() > 0) {
      size_t actual_read = litertlm_stream.gcount();
      NpuInfo npu_info = DetectNpuInfoFromTfliteBuffer(
          absl::string_view(buffer.data(), actual_read));
      if (npu_brand == NpuBrand::kUnknown) {
        npu_brand = npu_info.brand;
      }
      if (global_soc_name.empty() && !npu_info.soc_name.empty()) {
        global_soc_name = npu_info.soc_name;
      }
    }
  };

  if (has_any_npu) {
    if (is_artisan_tensor && npu_brand == NpuBrand::kUnknown) {
      npu_brand = NpuBrand::kGoogleTensor;
    }
    if (npu_section_end > npu_section_begin) {
      inspect_section_for_npu(npu_section_begin, npu_section_end);
    }
    if (found_main_tflite &&
        (npu_brand == NpuBrand::kUnknown || global_soc_name.empty())) {
      inspect_section_for_npu(main_tflite_begin, main_tflite_end);
    }
  }

  auto resolve_soc_name = [&](const std::string& modality_soc) -> std::string {
    return !modality_soc.empty() ? modality_soc : global_soc_name;
  };

  // Propagate detected NPU brand and SoC name to all modalities that support
  // NPU.
  auto update_npu_modality = [&](SupportedBackends& sb,
                                 const std::string& modality_soc) {
    if (sb.npu) {
      sb.npu_brand = npu_brand;
      sb.soc_name = resolve_soc_name(modality_soc);
      if (std::find(sb.preferred_backends.begin(), sb.preferred_backends.end(),
                    BackendType::kNpu) == sb.preferred_backends.end()) {
        sb.preferred_backends.push_back(BackendType::kNpu);
      }
    }
  };
  update_npu_modality(llm_cap.text_supported_backends, text_soc_name);
  update_npu_modality(llm_cap.vision_supported_backends, vision_soc_name);
  update_npu_modality(llm_cap.audio_supported_backends, audio_soc_name);
  update_npu_modality(llm_cap.video_supported_backends, video_soc_name);

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
        llm_cap.max_context_tokens = proto_metadata.max_num_tokens();
        llm_cap.is_dynamic_context = IsMagicNumber(llm_cap.max_context_tokens);
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
          ExtractVisionSignatureLengths(absl::string_view(buffer.get(), size)));
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

  // 3. Fallback: inspect main TFLite model graph for context size if not
  // already populated from LlmMetadata protobuf (e.g. for legacy or raw
  // TFLite models). For large models (e.g. 7B+ models > 4GB), LlmMetadata
  // protobuf is the primary source of truth, avoiding multi-gigabyte memory
  // allocations.
  if (llm_cap.max_context_tokens == 0 && found_main_tflite &&
      main_tflite_end > main_tflite_begin) {
    if (total_stream_size < 0 ||
        static_cast<uint64_t>(total_stream_size) >= main_tflite_end) {
      size_t size = main_tflite_end - main_tflite_begin;
      if (size <= kMaxVisionSectionSize) {
        litertlm_stream.seekg(main_tflite_begin);
        auto buffer = std::make_unique<char[]>(size);
        litertlm_stream.read(buffer.get(), size);
        if (litertlm_stream) {
          flatbuffers::Verifier verifier(
              reinterpret_cast<const uint8_t*>(buffer.get()), size);
          if (tflite::VerifyModelBuffer(verifier)) {
            const tflite::Model* tflite_model = tflite::GetModel(buffer.get());
            if (tflite_model != nullptr &&
                tflite_model->signature_defs() != nullptr) {
              for (const auto* sig : *tflite_model->signature_defs()) {
                if (sig == nullptr || sig->signature_key() == nullptr) continue;
                absl::string_view sig_key = sig->signature_key()->string_view();
                if (absl::StartsWith(sig_key, "prefill")) {
                  if (sig->inputs() != nullptr) {
                    for (const auto* input : *sig->inputs()) {
                      if (input == nullptr || input->name() == nullptr) {
                        continue;
                      }
                      absl::string_view input_name =
                          input->name()->string_view();
                      if (absl::StrContains(input_name, "mask")) {
                        uint32_t tensor_idx = input->tensor_index();
                        uint32_t subgraph_idx = sig->subgraph_index();
                        if (tflite_model->subgraphs() != nullptr &&
                            subgraph_idx < tflite_model->subgraphs()->size()) {
                          const auto* subgraph =
                              tflite_model->subgraphs()->Get(subgraph_idx);
                          if (subgraph != nullptr &&
                              subgraph->tensors() != nullptr &&
                              tensor_idx < subgraph->tensors()->size()) {
                            const auto* tensor =
                                subgraph->tensors()->Get(tensor_idx);
                            if (tensor != nullptr &&
                                tensor->shape() != nullptr) {
                              int rank = tensor->shape()->size();
                              if (rank > 0) {
                                int64_t dim = tensor->shape()->Get(rank - 1);
                                llm_cap.max_context_tokens = dim;
                                llm_cap.is_dynamic_context = IsMagicNumber(dim);
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  info.llm_capability = llm_cap;

  return info;
}

// Inspects a LiteRT-LM model file located at the specified file path and
// returns its extracted capabilities and configuration parameters.
absl::StatusOr<ModelCapabilities> InspectModel(
    absl::string_view litertlm_path) {
  std::ifstream input_file_stream(std::string(litertlm_path), std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  return InspectModel(input_file_stream);
}

// Formats the supported input/output modalities into a space-separated string
// (e.g. "Text Vision ").
std::ostream& operator<<(std::ostream& os,
                         const SupportedModalities& modalities) {
  if (modalities.text) os << "Text ";
  if (modalities.vision) os << "Vision ";
  if (modalities.audio) os << "Audio ";
  if (modalities.video) os << "Video ";
  return os;
}

// Formats the detected NPU brand into a human-readable name (e.g.
// "Qualcomm QNN", "Google Tensor TPU", "MediaTek Neuron", "Intel NPU",
// "Samsung Exynos NPU", "Unknown").
std::ostream& operator<<(std::ostream& os, const NpuBrand& brand) {
  for (const auto& spec : GetNpuBrandRegistry()) {
    if (spec.brand == brand) {
      return os << spec.display_name;
    }
  }
  return os << "Unknown";
}

// Formats the hardware backend type into a human-readable string ("CPU", "GPU",
// "NPU", "UNSPECIFIED").
std::ostream& operator<<(std::ostream& os, const BackendType& backend) {
  switch (backend) {
    case BackendType::kCpu:
      os << "CPU";
      break;
    case BackendType::kGpu:
      os << "GPU";
      break;
    case BackendType::kNpu:
      os << "NPU";
      break;
    default:
      os << "UNSPECIFIED";
      break;
  }
  return os;
}

// Formats the supported backends, detected NPU hardware / SoC name, and
// default backend into a human-readable string in priority order (e.g.
// "NPU (Qualcomm QNN SM8850) GPU CPU (Default: NPU)").
std::ostream& operator<<(std::ostream& os,
                         const SupportedBackends& backends) {
  if (!backends.preferred_backends.empty()) {
    for (BackendType b : backends.preferred_backends) {
      if (b == BackendType::kCpu) {
        os << "CPU ";
      } else if (b == BackendType::kGpu) {
        os << "GPU ";
      } else if (b == BackendType::kNpu) {
        os << "NPU (" << backends.npu_brand;
        if (!backends.soc_name.empty()) {
          os << " " << backends.soc_name;
        }
        os << ") ";
      }
    }
    os << "(Default: " << backends.preferred_backends.front() << ")";
  } else {
    if (backends.cpu) os << "CPU ";
    if (backends.gpu) os << "GPU ";
    if (backends.npu) {
      os << "NPU (" << backends.npu_brand;
      if (!backends.soc_name.empty()) {
        os << " " << backends.soc_name;
      }
      os << ") ";
    }
    if (backends.default_backend != BackendType::kUnspecified) {
      os << "(Default: " << backends.default_backend << ")";
    }
  }
  return os;
}

// Formats the full LLM inference capabilities (modalities, backends, token
// limits, dynamic context, sampler params, etc.) into a structured report.
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
     << "  Max Context Tokens:     " << llm_cap.max_context_tokens << "\n"
     << "  Is Dynamic Context:     "
     << (llm_cap.is_dynamic_context ? "YES" : "NO") << "\n"
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

// Formats the top-level ModelCapabilities object into the output stream.
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
