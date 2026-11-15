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

#ifdef __cplusplus
#error "This file must be compiled as C, not C++."
#endif

#include "c/c_abi_enum_test_helper.h"

#include <stddef.h>

#include "c/conversation.h"
#include "c/embedding_engine.h"
#include "c/engine.h"
#include "c/error_reporter.h"
#include "c/model_info.h"

LiteRtLmStatusCode c_abi_test_set_min_log_level(int raw_level) {
  return litert_lm_set_min_log_level((LiteRtLmLogSeverity)raw_level);
}

LiteRtLmStatusCode c_abi_test_input_data_create(
    int raw_type, const void* data, size_t size,
    LiteRtLmInputData** out_input_data) {
  return litert_lm_input_data_create((LiteRtLmInputDataType)raw_type, data,
                                     size, out_input_data);
}

LiteRtLmStatusCode c_abi_test_sampler_params_create(
    int raw_type, LiteRtLmSamplerParams** out_params) {
  return litert_lm_sampler_params_create((LiteRtLmSamplerType)raw_type,
                                         out_params);
}

LiteRtLmStatusCode c_abi_test_speculative_decoding_config_set_mode(
    LiteRtLmSpeculativeDecodingConfig* config, int raw_mode) {
  return litert_lm_speculative_decoding_config_set_mode(
      config, (LiteRtLmSpeculativeDecodingMode)raw_mode);
}

LiteRtLmStatusCode c_abi_test_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings, int raw_type) {
  return litert_lm_engine_settings_set_activation_data_type(
      settings, (LiteRtLmActivationDataType)raw_type);
}

LiteRtLmStatusCode c_abi_test_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args, int raw_constraint_type,
    const char* constraint_string) {
  return litert_lm_conversation_optional_args_set_constraint(
      optional_args, (LiteRtLmConstraintType)raw_constraint_type,
      constraint_string);
}

LiteRtLmStatusCode c_abi_test_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config, int raw_provider_type) {
  return litert_lm_conversation_config_set_constraint_provider(
      config, (LiteRtLmConstraintProviderType)raw_provider_type);
}

LiteRtLmStatusCode
c_abi_test_embedding_engine_settings_set_activation_data_type(
    LiteRtLmEmbeddingEngineSettings* settings, int raw_type) {
  return litert_lm_embedding_engine_settings_set_activation_data_type(
      settings, (LiteRtLmActivationDataType)raw_type);
}

LiteRtLmStatusCode c_abi_test_embedding_options_set_input_overflow_strategy(
    LiteRtLmEmbeddingOptions* options, int raw_strategy) {
  return litert_lm_embedding_options_set_input_overflow_strategy(
      options, (LiteRtLmInputOverflowStrategy)raw_strategy);
}

LiteRtLmStatusCode c_abi_test_loaded_file_supports_input_modality(
    const LiteRtLmLoadedFile* loaded_file, int raw_modality,
    bool* out_supported) {
  return litert_lm_loaded_file_supports_input_modality(
      loaded_file, (LiteRtLmModality)raw_modality, out_supported);
}

LiteRtLmStatusCode c_abi_test_loaded_file_modality_supported_backends(
    const LiteRtLmLoadedFile* loaded_file, int raw_modality,
    LiteRtLmBackendType* backends, size_t max_size, size_t* out_count) {
  return litert_lm_loaded_file_modality_supported_backends(
      loaded_file, (LiteRtLmModality)raw_modality, backends, max_size,
      out_count);
}

LiteRtLmStatusCode c_abi_test_loaded_file_modality_npu_brand(
    const LiteRtLmLoadedFile* loaded_file, int raw_modality,
    LiteRtLmNpuBrand* out_npu_brand) {
  return litert_lm_loaded_file_modality_npu_brand(
      loaded_file, (LiteRtLmModality)raw_modality, out_npu_brand);
}

LiteRtLmStatusCode c_abi_test_loaded_file_modality_soc_name(
    const LiteRtLmLoadedFile* loaded_file, int raw_modality,
    const char** out_soc_name) {
  return litert_lm_loaded_file_modality_soc_name(
      loaded_file, (LiteRtLmModality)raw_modality, out_soc_name);
}
