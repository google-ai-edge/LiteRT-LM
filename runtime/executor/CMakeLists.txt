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

add_subdirectory(litert)
add_subdirectory(npu)
add_subdirectory(audio)
add_subdirectory(vision)
add_subdirectory(embedding)


# ==============================================================================
# Default Static GPU Accelerator
# ==============================================================================
add_litertlm_library(runtime_executor_default_static_gpu_accelerator INTERFACE)
add_library(LiteRTLM::Runtime::Executor::DefaultStaticGpuAcceleratorInterface
  ALIAS runtime_executor_default_static_gpu_accelerator)

# ==============================================================================
# LLM Processed Context Interface
# ==============================================================================
add_litertlm_library(runtime_executor_llm_processed_context INTERFACE)
add_library(LiteRTLM::Runtime::Executor::LlmProcessedContextInterface 
  ALIAS runtime_executor_llm_processed_context)

target_include_directories(runtime_executor_llm_processed_context 
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)


# ==============================================================================
# Audio LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_audio_litert_compiled_model_executor STATIC
  audio_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioLitertCompiledModelExecutor
  ALIAS runtime_executor_audio_litert_compiled_model_executor)

target_include_directories(runtime_executor_audio_litert_compiled_model_executor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::AudioExecutorInterface
    LiteRTLM::Runtime::Executor::AudioExecutorSettings
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils

    LITERTLM_DEPS
)

# ==============================================================================
# Executor Settings Base
# ==============================================================================
add_litertlm_library(runtime_executor_executor_settings_base STATIC
  executor_settings_base.cc
)
add_library(LiteRTLM::Runtime::Executor::ExecutorSettingsBase
ALIAS runtime_executor_executor_settings_base)

target_include_directories(runtime_executor_executor_settings_base
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_executor_settings_base
  PUBLIC
    LiteRTLM::Runtime::Util::ScopedFile
    # LiteRTLM::Runtime::Util::MemoryMappedFile

    LITERTLM_DEPS
)

# ==============================================================================
# Fake LLM Executor
# ==============================================================================
add_litertlm_library(runtime_executor_fake_llm_executor STATIC
  fake_llm_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::FakeLlmExecutor ALIAS runtime_executor_fake_llm_executor)

target_include_directories(runtime_executor_fake_llm_executor
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_fake_llm_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Util::ConvertTensorBuffer

    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT Compiled Model Executor Utils
# ==============================================================================
add_litertlm_library(runtime_executor_litert_compiled_model_executor_utils STATIC
  litert_compiled_model_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils ALIAS runtime_executor_litert_compiled_model_executor_utils)

target_include_directories(runtime_executor_litert_compiled_model_executor_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_litert_compiled_model_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::ModelResourcesLitertLm
    LiteRTLM::Runtime::Components::ModelResourcesTask
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Components::EmbeddingLookup::Text
    LiteRTLM::Generated::Protobuf
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::FileFormatUtil
    LiteRTLM::Runtime::Util::LitertLmLoader
    LiteRTLM::Runtime::Util::ModelAssetBundleResources
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Runtime::Util::TensorBufferUtil
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor Interface
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor INTERFACE)
add_library(LiteRTLM::Runtime::Executor::LlmExecutorInterface
  ALIAS runtime_executor_llm_executor)

target_include_directories(runtime_executor_llm_executor
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor
  INTERFACE
    LiteRTLM::Runtime::Executor::LlmExecutorBaseInterface
    LiteRTLM::Runtime::Util::LoggingTensorBuffer
)

# ==============================================================================
# LLM Executor Base Interface
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_base INTERFACE)
add_library(LiteRTLM::Runtime::Executor::LlmExecutorBaseInterface
  ALIAS runtime_executor_llm_executor_base)

target_include_directories(runtime_executor_llm_executor_base
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_base
  INTERFACE
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor IO Types
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_io_types STATIC
  llm_executor_io_types.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
  ALIAS runtime_executor_llm_executor_io_types)

target_include_directories(runtime_executor_llm_executor_io_types
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_io_types
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    LiteRTLM::Runtime::Util::LoggingTensorBuffer
    LITERTLM_DEPS
)

# ==============================================================================
# LLM Executor Processed Tokens
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_processed_tokens STATIC
  llm_executor_processed_tokens.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
  ALIAS runtime_executor_llm_executor_processed_tokens)

target_include_directories(runtime_executor_llm_executor_processed_tokens
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
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
add_library(LiteRTLM::Runtime::Executor::LlmExecutorSettings
  ALIAS runtime_executor_llm_executor_settings)

target_include_directories(runtime_executor_llm_executor_settings
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Util::Logging
    LITERTLM_DEPS
)
# ==============================================================================
# LLM Executor Settings Utils
# ==============================================================================
add_litertlm_library(runtime_executor_llm_executor_settings_utils STATIC
  llm_executor_settings_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmExecutorSettingsUtils
  ALIAS runtime_executor_llm_executor_settings_utils)

target_include_directories(runtime_executor_llm_executor_settings_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_executor_settings_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Util::FileUtil
    LiteRTLM::Runtime::Util::Logging
    LITERTLM_DEPS
)
# ==============================================================================
# Common Utils
# ==============================================================================
add_litertlm_library(runtime_executor_common_utils STATIC
  common_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::CommonUtils
  ALIAS runtime_executor_common_utils)

target_include_directories(runtime_executor_common_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_common_utils
  PUBLIC
    LITERTLM_DEPS
)



# ==============================================================================
# LLM LiteRT Compiled Model Cache Utils
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_cache_utils STATIC
  llm_litert_compiled_model_cache_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmLitertCompiledModelCacheUtils ALIAS runtime_executor_llm_litert_compiled_model_cache_utils)

target_include_directories(runtime_executor_llm_litert_compiled_model_cache_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_cache_utils
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT Compiled Model Executor Factory
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_executor_factory STATIC
  llm_litert_compiled_model_executor_factory.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutorFactory 
ALIAS runtime_executor_llm_litert_compiled_model_executor_factory)

target_include_directories(runtime_executor_llm_litert_compiled_model_executor_factory
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_executor_factory
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutor
    LiteRTLM::Runtime::Executor::LlmNpuCompiledModel
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
    LiteRTLM::Runtime::Executor::LlmLitertCompiledModelCacheUtils
    LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::ModelResourcesLitertLm
    LiteRTLM::Runtime::Components::ModelResourcesTask
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::SamplerFactory
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::FileUtil
    LiteRTLM::Runtime::Util::LoraUtil
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_compiled_model_executor STATIC
  llm_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutor
  ALIAS runtime_executor_llm_litert_compiled_model_executor)

target_include_directories(runtime_executor_llm_litert_compiled_model_executor
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LlmExecutorSettingsUtils
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
    LiteRTLM::Runtime::Executor::LlmLitertCompiledModelCacheUtils
    LiteRTLM::Runtime::Executor::LlmLitertMtpDrafter
    LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::ModelResourcesLitertLm
    LiteRTLM::Runtime::Components::ModelResourcesTask
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::SamplerFactory
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::FileUtil
    LiteRTLM::Runtime::Util::LoraUtil
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_npu_compiled_model_executor STATIC
  llm_litert_npu_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmNpuCompiledModel ALIAS runtime_executor_llm_litert_npu_compiled_model_executor)

target_include_directories(runtime_executor_llm_litert_npu_compiled_model_executor
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_npu_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::Npu::Embedder
    LiteRTLM::Runtime::Executor::Npu::KVCache
    LiteRTLM::Runtime::Executor::Npu::Mask
    LiteRTLM::Runtime::Executor::Npu::Rope
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::ModelResourcesTask
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
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
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_magic_number_configs_helper
  PUBLIC
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    runtime_util_litert_status_util

    LITERTLM_DEPS
)


# ==============================================================================
# Vision LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_vision_litert_compiled_model_executor STATIC
  vision_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::VisionLitertCompiledModelExecutor
ALIAS runtime_executor_vision_litert_compiled_model_executor)

target_include_directories(runtime_executor_vision_litert_compiled_model_executor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_vision_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::VisionExecutorSettings
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Executor::VisionExecutorInterface
    LITERTLM_DEPS
)



# ==============================================================================
# LLM LiteRT MTP Drafter
# ==============================================================================
add_litertlm_library(runtime_executor_llm_litert_mtp_drafter STATIC
  llm_litert_mtp_drafter.cc
)
add_library(LiteRTLM::Runtime::Executor::LlmLitertMtpDrafter ALIAS runtime_executor_llm_litert_mtp_drafter)

target_include_directories(runtime_executor_llm_litert_mtp_drafter
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_llm_litert_mtp_drafter
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::SamplerFactory
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LITERTLM_DEPS
)



# ==============================================================================
# Embedding LiteRT Compiled Model Executor
# ==============================================================================
add_litertlm_library(runtime_executor_embedding_litert_compiled_model_executor STATIC
  embedding_litert_compiled_model_executor.cc
)
add_library(LiteRTLM::Runtime::Executor::EmbeddingLitertCompiledModelExecutor ALIAS
  runtime_executor_embedding_litert_compiled_model_executor)

target_include_directories(runtime_executor_embedding_litert_compiled_model_executor
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_embedding_litert_compiled_model_executor
  PUBLIC
    LiteRTLM::Runtime::Executor::EmbeddingExecutorSettings
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# Model Signature Utils
# ==============================================================================
add_litertlm_library(runtime_executor_model_signature_utils STATIC
  model_signature_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::ModelSignatureUtils ALIAS
  runtime_executor_model_signature_utils)

target_include_directories(runtime_executor_model_signature_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_model_signature_utils
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor ALIAS runtime_executor_libs)

target_link_libraries(runtime_executor_libs INTERFACE
  LiteRTLM::Runtime::Executor::Audio
  LiteRTLM::Runtime::Executor::Vision
  LiteRTLM::Runtime::Executor::LlmExecutorInterface
  LiteRTLM::Runtime::Executor::AudioLitertCompiledModelExecutor
  LiteRTLM::Runtime::Executor::CommonUtils
  LiteRTLM::Runtime::Executor::ExecutorSettingsBase
  LiteRTLM::Runtime::Executor::FakeLlmExecutor
  LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
  LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
  LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
  LiteRTLM::Runtime::Executor::LlmExecutorSettings
  LiteRTLM::Runtime::Executor::LlmExecutorSettingsUtils
  LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutorFactory
  LiteRTLM::Runtime::Executor::LlmLitertCompiledModelCacheUtils
  LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutor
  LiteRTLM::Runtime::Executor::LlmLitertMtpDrafter
  LiteRTLM::Runtime::Executor::Npu
  LiteRTLM::Runtime::Executor::LlmNpuCompiledModel
  LiteRTLM::Runtime::Executor::LlmProcessedContextInterface
  LiteRTLM::Runtime::Executor::MagicNumberConfigsHelper
  LiteRTLM::Runtime::Executor::VisionLitertCompiledModelExecutor
  LiteRTLM::Runtime::Executor::DefaultStaticGpuAcceleratorInterface
  LiteRTLM::Runtime::Executor::Litert
  LiteRTLM::Runtime::Executor::Embedding
  LiteRTLM::Runtime::Executor::EmbeddingLitertCompiledModelExecutor
  LiteRTLM::Runtime::Executor::ModelSignatureUtils
)