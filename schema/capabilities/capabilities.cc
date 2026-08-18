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

#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/status_macros.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm::schema::capabilities {

absl::StatusOr<ModelMetadataInfo> InspectModel(std::istream& litertlm_stream) {
  litertlm_stream.seekg(0);
  LitertlmHeader header;
  ABSL_RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_stream, &header));

  const LiteRTLMMetaData* metadata = header.metadata;
  RET_CHECK_NE(metadata, nullptr);

  ModelMetadataInfo info;

  // 1. Extract System / File-level metadata.
  if (const auto* system_metadata = metadata->system_metadata()) {
    if (const auto* entries = system_metadata->entries()) {
      for (size_t i = 0; i < entries->size(); ++i) {
        const KeyValuePair* item = entries->Get(i);
        if (item == nullptr || item->key() == nullptr) continue;
        absl::string_view key = item->key()->string_view();
        const auto* val = item->value_as_StringValue();
        if (val == nullptr || val->value() == nullptr) continue;
        std::string value_str = std::string(val->value()->string_view());

        if (key == "model_class") {
          info.model_class = value_str;
        } else if (key == "tf_hub_model_id") {
          info.tf_hub_model_id = value_str;
        } else if (key == "min_litertlm_version") {
          info.min_litertlm_version = value_str;
        }
      }
    }
  }

  // 2. Discover sections and capabilities.
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
  bool has_speculative_decoding = false;

  for (size_t i = 0; i < section_objects->size(); ++i) {
    const auto* section = section_objects->Get(i);
    if (section == nullptr) continue;

    if (section->data_type() == AnySectionDataType_LlmMetadataProto) {
      has_llm_metadata = true;
      llm_metadata_begin = section->begin_offset();
      llm_metadata_end = section->end_offset();
    } else if (section->data_type() == AnySectionDataType_TFLiteModel) {
      if (const auto* items = section->items()) {
        for (size_t j = 0; j < items->size(); ++j) {
          const KeyValuePair* item = items->Get(j);
          if (item == nullptr || item->key() == nullptr) continue;
          if (item->key()->string_view() == "model_type") {
            const auto* value = item->value_as_StringValue();
            if (value == nullptr || value->value() == nullptr) continue;
            absl::string_view model_type = value->value()->string_view();
            if (model_type == "tf_lite_vision_adapter" ||
                model_type == "tf_lite_vision_encoder") {
              has_vision = true;
            } else if (model_type == "tf_lite_audio_adapter" ||
                       model_type == "tf_lite_audio_encoder_hw") {
              has_audio = true;
            } else if (model_type == "tf_lite_mtp_drafter") {
              has_speculative_decoding = true;
            }
          }
        }
      }
    }
  }

  // 3. Populate LlmInferenceCapability if LLM metadata is present.
  if (has_llm_metadata) {
    LlmInferenceCapability llm_cap;
    llm_cap.input_modalities.push_back(Modality::kText);
    if (has_vision) {
      llm_cap.input_modalities.push_back(Modality::kVision);
    }
    if (has_audio) {
      llm_cap.input_modalities.push_back(Modality::kAudio);
    }
    llm_cap.output_modalities.push_back(Modality::kText);
    llm_cap.supports_speculative_decoding = has_speculative_decoding;

    if (llm_metadata_end > llm_metadata_begin) {
      size_t size = llm_metadata_end - llm_metadata_begin;
      litertlm_stream.seekg(llm_metadata_begin);
      std::unique_ptr<char[]> buffer(new char[size]);
      litertlm_stream.read(buffer.get(), size);
      if (litertlm_stream) {
        proto::LlmMetadata proto_metadata;
        if (proto_metadata.ParseFromArray(buffer.get(), size)) {
          llm_cap.max_context_length = proto_metadata.max_num_tokens();
          if (proto_metadata.has_sampler_params()) {
            llm_cap.default_sampler_params = proto_metadata.sampler_params();
          }
          for (const auto& arg : proto_metadata.jinja_prompt_template_args()) {
            if (arg == "tools") {
              llm_cap.supports_function_calling = true;
            }
            if (arg == "thought") {
              llm_cap.supports_thinking = true;
            }
          }
          for (const auto& channel : proto_metadata.channels()) {
            if (channel.is_reasoning_channel()) {
              llm_cap.supports_thinking = true;
            }
          }
          for (const auto& backend : proto_metadata.supported_backends()) {
            llm_cap.supported_backends.push_back(backend);
          }
          for (int32_t res : proto_metadata.supported_vision_resolutions()) {
            llm_cap.supported_vision_resolutions.push_back(res);
          }
        }
      }
    }
    info.llm_capability = llm_cap;
  }

  return info;
}

absl::StatusOr<ModelMetadataInfo> InspectModel(
    const std::string& litertlm_path) {
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  return InspectModel(input_file_stream);
}

}  // namespace litert::lm::schema::capabilities
