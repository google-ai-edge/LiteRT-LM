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

# ==============================================================================
# Audio Executor Interface
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor INTERFACE)
add_library(LiteRTLM::Runtime::Executor::AudioExecutor ALIAS runtime_executor_audio_executor)

target_include_directories(runtime_executor_audio_executor
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Audio Executor Base
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_base INTERFACE)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorBase ALIAS runtime_executor_audio_executor_base)

target_include_directories(runtime_executor_audio_executor_base
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_base
  INTERFACE
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Audio Executor Settings
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_settings STATIC
  audio_executor_settings.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorSettings ALIAS runtime_executor_audio_executor_settings)

target_include_directories(runtime_executor_audio_executor_settings
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# Audio Executor Utils
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_utils STATIC
  audio_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorUtils ALIAS runtime_executor_audio_executor_utils)

target_include_directories(runtime_executor_audio_executor_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    runtime_util_litert_status_util
)
    

# ==============================================================================
# Audio LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_audio_litert_compiled_model_executor STATIC
  audio_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioLiteRTCompiledModel
  ALIAS runtime_executor_audio_litert_compiled_model_executor)

target_include_directories(runtime_executor_audio_litert_compiled_model_executor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::AudioExecutor
    LiteRTLM::Runtime::Executor::AudioExecutorSettings
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Components::ModelResources::Interface
    runtime_util_litert_status_util
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    LiteRTLM::Runtime::Util::LoggingTensorBuffer

    LITERTLM_DEPS
)

# ==============================================================================
# Executor Settings Base
# ==============================================================================
add_litertlm_library(runtime_executor_executor_settings_base STATIC
  executor_settings_base.cc
)
add_library(LiteRTLM::Runtime::Executor::ExecutorSettingsBase ALIAS runtime_executor_executor_settings_base)

target_include_directories(runtime_executor_executor_settings_base
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_executor_settings_base
  PUBLIC
    runtime_util_scoped_file
    runtime_util_litert_status_util
    runtime_util_memory_mapped_file

    LITERTLM_DEPS
)

# ==============================================================================
# Fake LLM Executor
# ==============================================================================
add_litertlm_library(runtime_executor_fake_llm_executor STATIC
  fake_llm_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMFakeExecutor ALIAS runtime_executor_fake_llm_executor)

target_include_directories(runtime_executor_fake_llm_executor
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_fake_llm_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLM::Interface
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT Compiled Model Executor Utils
# ==============================================================================
add_litertlm_library(runtime_executor_litert_compiled_model_executor_utils STATIC
  litert_compiled_model_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils ALIAS runtime_executor_litert_compiled_model_executor_utils)

target_include_directories(runtime_executor_litert_compiled_model_executor_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_litert_compiled_model_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Components::ModelResources::LiteRTLM
    LiteRTLM::Runtime::Components::ModelResources::Task
    LiteRTLM::Runtime::Components::Tokenizer::SentencePiece
    runtime_util_file_format_util
    runtime_util_litert_lm_loader
    runtime_util_litert_status_util
    runtime_util_memory_mapped_file
    runtime_util_model_asset_bundle_resources
    runtime_util_scoped_file
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor Interface
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor INTERFACE)
add_library(LiteRTLM::Runtime::Executor::LLM::Interface ALIAS runtime_executor_llm_executor)

target_include_directories(runtime_executor_llm_executor
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor
  INTERFACE
    LiteRTLM::Runtime::Executor::LLMExecutorBase
    runtime_util_logging_tensor_buffer
)

# ==============================================================================
# LLM Executor Base
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_base INTERFACE)
add_library(LiteRTLM::Runtime::Executor::LLMExecutorBase ALIAS runtime_executor_llm_executor_base)

target_include_directories(runtime_executor_llm_executor_base
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_base
  INTERFACE
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor IO Types
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_io_types STATIC
  llm_executor_io_types.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMExecutorIoTypes ALIAS runtime_executor_llm_executor_io_types)

target_include_directories(runtime_executor_llm_executor_io_types
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_io_types
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    runtime_util_logging_tensor_buffer
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor Processed Tokens
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_processed_tokens STATIC
  llm_executor_processed_tokens.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMExecutorProcessedTokens ALIAS runtime_executor_llm_executor_processed_tokens)

target_include_directories(runtime_executor_llm_executor_processed_tokens
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_processed_tokens
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor Settings
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_settings STATIC
  llm_executor_settings.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMExecutorSettings ALIAS runtime_executor_llm_executor_settings)

target_include_directories(runtime_executor_llm_executor_settings
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    runtime_util_litert_status_util
    runtime_util_logging
    LITERTLM_DEPS
)
# ==============================================================================
# LLM Executor Settings Utils
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_settings_utils STATIC
  llm_executor_settings_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMExecutorSettingsUtils ALIAS runtime_executor_llm_executor_settings_utils)

target_include_directories(runtime_executor_llm_executor_settings_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_settings_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    runtime_util_litert_status_util
    runtime_util_logging
    LITERTLM_DEPS
)
# ==============================================================================
# Common Utils
# ==============================================================================
add_litertlm_library(runtime_executor_common_utils STATIC
  common_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::CommonUtils ALIAS runtime_executor_common_utils)

target_include_directories(runtime_executor_common_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_common_utils
  PUBLIC
    runtime_util_litert_status_util
    LITERTLM_DEPS
)



# ==============================================================================
# LLM LiteRT Compiled Model Cache Utils
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_cache_utils STATIC
  llm_litert_compiled_model_cache_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelCacheUtils ALIAS runtime_executor_llm_litert_compiled_model_cache_utils)

target_include_directories(runtime_executor_llm_litert_compiled_model_cache_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_cache_utils
  PUBLIC
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT Compiled Model Executor Factory
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_executor_factory STATIC
  llm_litert_compiled_model_executor_factory.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelExecutorFactory ALIAS runtime_executor_llm_litert_compiled_model_executor_factory)

target_include_directories(runtime_executor_llm_litert_compiled_model_executor_factory
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_executor_factory
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LiteRTLM::Runtime::Executor::LLM::CompiledModelExecutor
    LiteRTLM::Runtime::Executor::LLM::NpuCompiledModel
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LLM::Interface
    LiteRTLM::Runtime::Executor::LLMExecutorProcessedTokens
    LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelCacheUtils
    LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Components::ModelResources::LiteRTLM
    LiteRTLM::Runtime::Components::ModelResources::Task
    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::Sampler::Factory
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    runtime_util_convert_tensor_buffer
    runtime_util_file_util
    runtime_util_litert_status_util
    runtime_util_lora_util
    runtime_util_scoped_file
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_executor STATIC
  llm_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::LLM::CompiledModelExecutor ALIAS runtime_executor_llm_litert_compiled_model_executor)

target_include_directories(runtime_executor_llm_litert_compiled_model_executor
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LiteRTLM::Runtime::Executor::LLMExecutorSettingsUtils
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LLM::Interface
    LiteRTLM::Runtime::Executor::LLMExecutorProcessedTokens
    LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelCacheUtils
    LiteRTLM::Runtime::Executor::LLMLiteRTMtpDrafter
    LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Components::ModelResources::LiteRTLM
    LiteRTLM::Runtime::Components::ModelResources::Task
    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::Sampler::Factory
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    runtime_util_convert_tensor_buffer
    runtime_util_file_util
    runtime_util_litert_status_util
    runtime_util_lora_util
    runtime_util_scoped_file
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_npu_compiled_model_executor STATIC
  llm_litert_npu_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::LLM::NpuCompiledModel ALIAS runtime_executor_llm_litert_npu_compiled_model_executor)

target_include_directories(runtime_executor_llm_litert_npu_compiled_model_executor
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_npu_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LLM::Interface
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Components::ModelResources::Task
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Executor::LLMExecutorProcessedTokens
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# Magic Number Configs Helper
# ==============================================================================
add_litertlm_library(runtime_executor_magic_number_configs_helper STATIC
  magic_number_configs_helper.cc
)
add_library(LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper ALIAS runtime_executor_magic_number_configs_helper)

target_include_directories(runtime_executor_magic_number_configs_helper
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_magic_number_configs_helper
  PUBLIC
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LiteRTLM::Runtime::Components::ModelResources::Interface
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# Vision Executor Interface
# ==============================================================================
add_litertlm_library(runtime_executor_vision_executor INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Vision::Interface ALIAS runtime_executor_vision_executor)

target_include_directories(runtime_executor_vision_executor
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_executor
  INTERFACE
    LiteRTLM::Runtime::Executor::Vision::Base
)

# ==============================================================================
# Vision Executor Base
# ==============================================================================
add_litertlm_library(runtime_executor_vision_executor_base INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Vision::Base ALIAS runtime_executor_vision_executor_base)

target_include_directories(runtime_executor_vision_executor_base
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_executor_base
  INTERFACE
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Vision Executor Settings
# ==============================================================================
add_litertlm_library(runtime_executor_vision_executor_settings STATIC
  vision_executor_settings.cc
)
add_library(LiteRTLM::Runtime::Executor::Vision::Settings ALIAS runtime_executor_vision_executor_settings)

target_include_directories(runtime_executor_vision_executor_settings
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LITERTLM_DEPS
)


# ==============================================================================
# Vision Executor Utils
# ==============================================================================

add_litertlm_library(runtime_executor_vision_executor_utils STATIC
  vision_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::Vision::Utils ALIAS runtime_executor_vision_executor_utils)

target_include_directories(runtime_executor_vision_executor_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LITERTLM_DEPS
)

# ==============================================================================
# Vision LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_vision_litert_compiled_model_executor STATIC
  vision_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::Vision::CompiledModel ALIAS runtime_executor_vision_litert_compiled_model_executor)

target_include_directories(runtime_executor_vision_litert_compiled_model_executor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
    LiteRTLM::Runtime::Executor::Vision::Settings
    runtime_util_convert_tensor_buffer
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    runtime_util_litert_status_util
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Executor::Vision::Interface

    LITERTLM_DEPS
)

# ==============================================================================
# Default Static GPU Accelerator
# ==============================================================================
add_litertlm_library(runtime_executor_default_static_gpu_accelerator INTERFACE)
# Note: Empty target for CPU builds, but required for linking consistency.

# ==============================================================================
# LLM LiteRT MTP Drafter
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_mtp_drafter STATIC
  llm_litert_mtp_drafter.cc
)
add_library(LiteRTLM::Runtime::Executor::LLMLiteRTMtpDrafter ALIAS runtime_executor_llm_litert_mtp_drafter)

target_include_directories(runtime_executor_llm_litert_mtp_drafter
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_mtp_drafter
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LLMExecutorSettings
    LiteRTLM::Runtime::Components::ModelResources::Interface
    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::Sampler::Factory
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor ALIAS runtime_executor_libs)

target_link_libraries(runtime_executor_libs INTERFACE
  LiteRTLM::Runtime::Executor::AudioExecutor
  LiteRTLM::Runtime::Executor::AudioExecutorSettings
  LiteRTLM::Runtime::Executor::AudioExecutorUtils
  LiteRTLM::Runtime::Executor::AudioLiteRTCompiledModel
  LiteRTLM::Runtime::Executor::ExecutorSettingsBase
  LiteRTLM::Runtime::Executor::LLMFakeExecutor
  LiteRTLM::Runtime::Executor::LiteRTCompiledModelExecutorUtils
  LiteRTLM::Runtime::Executor::LLMExecutorIoTypes
  LiteRTLM::Runtime::Executor::LLMExecutorProcessedTokens
  LiteRTLM::Runtime::Executor::LLMExecutorSettings
  LiteRTLM::Runtime::Executor::LLMExecutorSettingsUtils
  LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelExecutorFactory
  LiteRTLM::Runtime::Executor::LLMLiteRTCompiledModelCacheUtils
  LiteRTLM::Runtime::Executor::LLM::CompiledModelExecutor
  LiteRTLM::Runtime::Executor::LLMLiteRTMtpDrafter
  LiteRTLM::Runtime::Executor::LLM::NpuCompiledModel
  LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
  LiteRTLM::Runtime::Executor::Vision::Settings
  LiteRTLM::Runtime::Executor::Vision::CompiledModel
  runtime_executor_default_static_gpu_accelerator
  LiteRTLM::Runtime::Executor::AudioExecutorUtils
)