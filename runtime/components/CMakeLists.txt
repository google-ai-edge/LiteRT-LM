# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(embedding_lookup)
add_subdirectory(constrained_decoding)
add_subdirectory(preprocessor)
add_subdirectory(tool_use)

# ==============================================================================
# Model Resources Interface
# ==============================================================================
add_litertlm_library(runtime_components_model_resources INTERFACE)
add_library(LiteRTLM::Runtime::Components::ModelResourcesInterface
  ALIAS runtime_components_model_resources)

target_include_directories(runtime_components_model_resources
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_model_resources
  INTERFACE
    LiteRTLM::Runtime::Components::TokenizerInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Sampler Interface
# ==============================================================================
add_litertlm_library(runtime_components_sampler INTERFACE)
add_library(LiteRTLM::Runtime::Components::SamplerInterface
  ALIAS runtime_components_sampler)

target_include_directories(runtime_components_sampler
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_sampler
  INTERFACE
    LITERTLM_DEPS
)

# ==============================================================================
# Tokenizer Interface
# ==============================================================================
add_litertlm_library(runtime_components_tokenizer INTERFACE)
add_library(LiteRTLM::Runtime::Components::TokenizerInterface
  ALIAS runtime_components_tokenizer)

target_include_directories(runtime_components_tokenizer
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tokenizer
  INTERFACE
    runtime_util_convert_tensor_buffer
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA
# ==============================================================================
add_litertlm_library(runtime_components_lora STATIC
  lora.cc
)
add_library(LiteRTLM::Runtime::Components::Lora
  ALIAS runtime_components_lora)

target_include_directories(runtime_components_lora
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/litert/src/litert_external-build/include
)

target_link_libraries(runtime_components_lora
  PUBLIC
    runtime_util_litert_status_util
    runtime_util_lora_data
    runtime_util_lora_util
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA Manager
# ==============================================================================
add_litertlm_library(runtime_components_lora_manager STATIC
  lora_manager.cc
)
add_library(LiteRTLM::Runtime::Components::LoraManager
  ALIAS runtime_components_lora_manager)

target_include_directories(runtime_components_lora_manager
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_lora_manager
  PUBLIC
    LiteRTLM::Runtime::Components::Lora
    LiteRTLM::Runtime::Util::LoraData
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LITERTLM_DEPS
)

# ==============================================================================
# Model Resources LiteRT LM
# ==============================================================================
add_litertlm_library(runtime_components_model_resources_litert_lm STATIC
  model_resources_litert_lm.cc
)
add_library(LiteRTLM::Runtime::Components::ModelResourcesLitertLm
  ALIAS runtime_components_model_resources_litert_lm)

target_include_directories(runtime_components_model_resources_litert_lm
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_model_resources_litert_lm
  PUBLIC
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Util::LitertLmLoader
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Schema::Core

    LITERTLM_DEPS
)

# ==============================================================================
# Model Resources Streaming
# ==============================================================================
add_litertlm_library(runtime_components_model_resources_streaming STATIC
  model_resources_streaming.cc
)
add_library(LiteRTLM::Runtime::Components::ModelResourcesStreaming
  ALIAS runtime_components_model_resources_streaming)

target_include_directories(runtime_components_model_resources_streaming
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_model_resources_streaming
  PUBLIC
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Util::LitertLmLoader
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Schema::Core

    LITERTLM_DEPS
)


# ==============================================================================
# Model Resources Task
# ==============================================================================
add_litertlm_library(runtime_components_model_resources_task STATIC
  model_resources_task.cc
)
add_library(LiteRTLM::Runtime::Components::ModelResourcesTask
  ALIAS runtime_components_model_resources_task)

target_include_directories(runtime_components_model_resources_task
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_model_resources_task
  PUBLIC
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Util::ModelAssetBundleResources
    LiteRTLM::Runtime::Util::MetadataUtil
    LiteRTLM::Runtime::Components::TokenizerInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Prompt Template
# ==============================================================================
add_litertlm_library(runtime_components_prompt_template STATIC
  prompt_template.cc
)
add_library(LiteRTLM::Runtime::Components::PromptTemplate
  ALIAS runtime_components_prompt_template)

target_include_directories(runtime_components_prompt_template
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/json/include
)

target_link_libraries(runtime_components_prompt_template
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Sampler Factory
# ==============================================================================
add_litertlm_library(runtime_components_sampler_factory STATIC
  sampler_factory.cc
)
add_library(LiteRTLM::Runtime::Components::SamplerFactory
  ALIAS runtime_components_sampler_factory)

target_include_directories(runtime_components_sampler_factory
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_sampler_factory
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::TopPCpuSampler
    LITERTLM_DEPS
)

# ==============================================================================
# Sampling CPU Util
# ==============================================================================
add_litertlm_library(runtime_components_sampling_cpu_util STATIC
  sampling_cpu_util.cc
)
add_library(LiteRTLM::Runtime::Components::SamplingCpuUtil
  ALIAS runtime_components_sampling_cpu_util)

target_include_directories(runtime_components_sampling_cpu_util
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_sampling_cpu_util
   PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Scoring CPU Util
# ==============================================================================
add_litertlm_library(runtime_components_scoring_cpu_util STATIC
  scoring_cpu_util.cc
)
add_library(LiteRTLM::Runtime::Components::ScoringCpuUtil
  ALIAS runtime_components_scoring_cpu_util)

target_include_directories(runtime_components_scoring_cpu_util
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_scoring_cpu_util
  PUBLIC
    LiteRTLM::Runtime::Components::SamplingCpuUtil
    LITERTLM_DEPS
)

# # ==============================================================================
# # SentencePiece Tokenizer
# # ==============================================================================
# add_litertlm_library(runtime_components_sentencepiece_tokenizer STATIC
#   sentencepiece_tokenizer.cc
# )
# add_library(LiteRTLM::Runtime::Components::SentencePieceTokenizer
#   ALIAS runtime_components_sentencepiece_tokenizer)

# target_compile_definitions(runtime_components_sentencepiece_tokenizer
#   PUBLIC ENABLE_SENTENCEPIECE_TOKENIZER
# )

# target_include_directories(runtime_components_sentencepiece_tokenizer
#   PRIVATE
#     ${LITERTLM_GENERATED_SRC_DIR}
#     ${LITERTLM_INCLUDE_PATHS}
# )

# target_link_libraries(runtime_components_sentencepiece_tokenizer
#   PUBLIC
#     LiteRTLM::Runtime::Components::TokenizerInterface
#   PRIVATE
#     LITERTLM_DEPS
# )

# ==============================================================================
# Stop Token Detector
# ==============================================================================
add_litertlm_library(runtime_components_stop_token_detector STATIC
  stop_token_detector.cc
)
add_library(LiteRTLM::Runtime::Components::StopTokenDetector
  ALIAS runtime_components_stop_token_detector)

target_include_directories(runtime_components_stop_token_detector
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_stop_token_detector
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Token ID Util
# ==============================================================================
add_litertlm_library(runtime_components_token_id_util STATIC
  token_id_util.cc
)
add_library(LiteRTLM::Runtime::Components::TokenIdUtil
  ALIAS runtime_components_token_id_util)

target_include_directories(runtime_components_token_id_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_token_id_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Top P CPU Sampler
# ==============================================================================
add_litertlm_library(runtime_components_top_p_cpu_sampler STATIC
  top_p_cpu_sampler.cc
)
add_library(LiteRTLM::Runtime::Components::TopPCpuSampler
  ALIAS runtime_components_top_p_cpu_sampler)

target_include_directories(runtime_components_top_p_cpu_sampler
  PUBLIC
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_top_p_cpu_sampler
  PUBLIC
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::SamplingCpuUtil
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::TensorBufferUtil
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_components_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components ALIAS runtime_components_libs)

target_link_libraries(runtime_components_libs INTERFACE
  LiteRTLM::Runtime::Components::ConstrainedDecoding
  LiteRTLM::Runtime::Components::EmbeddingLookup
  LiteRTLM::Runtime::Components::Preprocessor
  LiteRTLM::Runtime::Components::ToolUse
  LiteRTLM::Runtime::Components::ModelResourcesInterface
  LiteRTLM::Runtime::Components::SamplerInterface
  LiteRTLM::Runtime::Components::TokenizerInterface
  LiteRTLM::Runtime::Components::Lora
  LiteRTLM::Runtime::Components::LoraManager
  LiteRTLM::Runtime::Components::ModelResourcesInterface
  LiteRTLM::Runtime::Components::ModelResourcesLitertLm
  LiteRTLM::Runtime::Components::ModelResourcesStreaming
  LiteRTLM::Runtime::Components::ModelResourcesTask
  LiteRTLM::Runtime::Components::PromptTemplate
  LiteRTLM::Runtime::Components::SamplerFactory
  LiteRTLM::Runtime::Components::SamplingCpuUtil
  LiteRTLM::Runtime::Components::ScoringCpuUtil
  LiteRTLM::Runtime::Components::StopTokenDetector
  LiteRTLM::Runtime::Components::TokenIdUtil
  LiteRTLM::Runtime::Components::TopPCpuSampler
)