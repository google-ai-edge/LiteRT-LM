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
# Convert Tensor Buffer
# ==============================================================================
add_litertlm_library(runtime_util_convert_tensor_buffer INTERFACE)
add_library(LiteRTLM::Runtime::Util::ConvertTensorBuffer ALIAS runtime_util_convert_tensor_buffer)

target_include_directories(runtime_util_convert_tensor_buffer
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_convert_tensor_buffer
  INTERFACE
    LITERTLM_DEPS
)

# ==============================================================================
# Data Stream
# ==============================================================================
add_litertlm_library(runtime_util_data_stream STATIC
  data_stream.cc
)
add_library(LiteRTLM::Runtime::Util::DataStream ALIAS runtime_util_data_stream)

target_include_directories(runtime_util_data_stream
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_data_stream
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::TensorBufferUtil
    runtime_executor_llm_executor_io_types
    LITERTLM_DEPS
)

# ==============================================================================
# Tensor Buffer Util
# ==============================================================================
add_litertlm_library(runtime_util_tensor_buffer_util STATIC
  tensor_buffer_util.cc
)
add_library(LiteRTLM::Runtime::Util::TensorBufferUtil ALIAS runtime_util_tensor_buffer_util)

target_include_directories(runtime_util_tensor_buffer_util
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_tensor_buffer_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Scoped File (Platform Specific)
# ==============================================================================
set(LITERT_CC_INTERNAL_DIR "${LITERT_SRC_DIR}/cc/internal/" CACHE PATH "")
add_litertlm_library(runtime_util_scoped_file INTERFACE)
add_library(LiteRTLM::Runtime::Util::ScopedFile ALIAS runtime_util_scoped_file)

target_include_directories(runtime_util_scoped_file
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# File Util
# ==============================================================================
add_litertlm_library(runtime_util_file_util STATIC
  file_util.cc
)
add_library(LiteRTLM::Runtime::Util::FileUtil ALIAS runtime_util_file_util)

target_include_directories(runtime_util_file_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_file_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# File Data Stream
# ==============================================================================
add_litertlm_library(runtime_util_file_data_stream STATIC
  data_stream.cc
)
add_library(LiteRTLM::Runtime::Util::FileDataStream ALIAS runtime_util_file_data_stream)

target_include_directories(runtime_util_file_data_stream
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_file_data_stream
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::TensorBufferUtil
    runtime_executor_llm_executor_io_types
    LITERTLM_DEPS
)

# ==============================================================================
# File Format Util
# ==============================================================================
add_litertlm_library(runtime_util_file_format_util STATIC
  file_format_util.cc
)
add_library(LiteRTLM::Runtime::Util::FileFormatUtil ALIAS runtime_util_file_format_util)

target_include_directories(runtime_util_file_format_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_file_format_util
  PUBLIC
    runtime_executor_executor_settings_base
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# Executor Data Util
# ==============================================================================
add_litertlm_library(runtime_util_executor_data_util STATIC
  executor_data_util.cc
)
add_library(LiteRTLM::Runtime::Util::ExecutorDataUtil ALIAS runtime_util_executor_data_util)

target_include_directories(runtime_util_executor_data_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_executor_data_util
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::TensorBufferUtil
    runtime_executor_llm_executor_io_types
    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT LM Loader
# ==============================================================================
add_litertlm_library(runtime_util_litert_lm_loader STATIC
  litert_lm_loader.cc
)
add_library(LiteRTLM::Runtime::Util::LitertLmLoader ALIAS runtime_util_litert_lm_loader)

target_include_directories(runtime_util_litert_lm_loader
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_litert_lm_loader
  PUBLIC
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Schema::Core
    LITERTLM_DEPS
    LiteRTLM::Runtime::Components::ModelResourcesInterface # Components
)

# ==============================================================================
# LiteRT LM Streaming Loader
# ==============================================================================
add_litertlm_library(runtime_util_litert_lm_streaming_loader STATIC
  litert_lm_streaming_loader.cc
)
add_library(LiteRTLM::Runtime::Util::LitertLmStreamingLoader ALIAS runtime_util_litert_lm_streaming_loader)

target_include_directories(runtime_util_litert_lm_streaming_loader
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_litert_lm_streaming_loader
  PUBLIC
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Schema::Core
    LITERTLM_DEPS
    LiteRTLM::Runtime::Components::ModelResourcesInterface
)

# ==============================================================================
# LiteRT Util
# ==============================================================================
add_litertlm_library(runtime_util_litert_util STATIC
  litert_util.cc
)
add_library(LiteRTLM::Runtime::Util::LitertUtil ALIAS runtime_util_litert_util)

target_include_directories(runtime_util_litert_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_litert_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Logging
# ==============================================================================
add_litertlm_library(runtime_util_logging STATIC
  logging.cc
)
add_library(LiteRTLM::Runtime::Util::Logging ALIAS runtime_util_logging)

target_include_directories(runtime_util_logging
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_logging
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Log Tensor Buffer
# ==============================================================================
add_litertlm_library(runtime_util_log_tensor_buffer STATIC
  log_tensor_buffer.cc
)
add_library(LiteRTLM::Runtime::Util::LogTensorBuffer ALIAS runtime_util_log_tensor_buffer)

target_include_directories(runtime_util_log_tensor_buffer
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_log_tensor_buffer
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Logging Tensor Buffer
# ==============================================================================
add_litertlm_library(runtime_util_logging_tensor_buffer STATIC
  logging_tensor_buffer.cc
)
add_library(LiteRTLM::Runtime::Util::LoggingTensorBuffer ALIAS runtime_util_logging_tensor_buffer)

target_include_directories(runtime_util_logging_tensor_buffer
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_logging_tensor_buffer
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA Util
# ==============================================================================
add_litertlm_library(runtime_util_lora_util STATIC
  lora_util.cc
)
add_library(LiteRTLM::Runtime::Util::LoraUtil ALIAS runtime_util_lora_util)

target_include_directories(runtime_util_lora_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_lora_util
  PUBLIC
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA Data
# ==============================================================================
add_litertlm_library(runtime_util_lora_data STATIC
  lora_data.cc
)
add_library(LiteRTLM::Runtime::Util::LoraData ALIAS runtime_util_lora_data)

target_include_directories(runtime_util_lora_data
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_lora_data
  PUBLIC
    LiteRTLM::Runtime::Util::LoraUtil
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# Metadata Util
# ==============================================================================
add_litertlm_library(runtime_util_metadata_util STATIC
  metadata_util.cc
)
add_library(LiteRTLM::Runtime::Util::MetadataUtil ALIAS runtime_util_metadata_util)

target_include_directories(runtime_util_metadata_util
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_metadata_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Zip Readonly Mem File
# ==============================================================================
add_litertlm_library(runtime_util_zip_readonly_mem_file STATIC
  zip_readonly_mem_file.cc
)
add_library(LiteRTLM::Runtime::Util::ZipReadonlyMemFile ALIAS runtime_util_zip_readonly_mem_file)

target_include_directories(runtime_util_zip_readonly_mem_file
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/minizip
)

target_link_libraries(runtime_util_zip_readonly_mem_file
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Zip Utils
# ==============================================================================
add_litertlm_library(runtime_util_zip_utils STATIC
  zip_utils.cc
)
add_library(LiteRTLM::Runtime::Util::ZipUtils ALIAS runtime_util_zip_utils)

target_include_directories(runtime_util_zip_utils
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}
)

target_link_libraries(runtime_util_zip_utils
  PUBLIC
    LiteRTLM::Runtime::Util::ZipReadonlyMemFile
    LITERTLM_DEPS
)

# ==============================================================================
# Model Asset Bundle Resources
# ==============================================================================
add_litertlm_library(runtime_util_model_asset_bundle_resources STATIC
  model_asset_bundle_resources.cc
)
add_library(LiteRTLM::Runtime::Util::ModelAssetBundleResources
  ALIAS runtime_util_model_asset_bundle_resources)

target_include_directories(runtime_util_model_asset_bundle_resources
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_model_asset_bundle_resources
  PUBLIC
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Runtime::Util::ZipUtils
    LITERTLM_DEPS
)

# ==============================================================================
#  Model Type Utils
# ==============================================================================
add_litertlm_library(runtime_util_model_type_utils STATIC
  model_type_utils.cc
)
add_library(LiteRTLM::Runtime::Util::ModelTypeUtils ALIAS runtime_util_model_type_utils)

target_include_directories(runtime_util_model_type_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_model_type_utils
  PUBLIC
    LiteRTLM::Runtime::Components::TokenizerInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Metrics Util
# ==============================================================================
add_litertlm_library(runtime_util_metrics_util STATIC
  metrics_util.cc
)
add_library(LiteRTLM::Runtime::Util::MetricsUtil
  ALIAS runtime_util_metrics_util)

target_include_directories(runtime_util_metrics_util
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_util_metrics_util
  PUBLIC
    LiteRTLM::Runtime::Engine::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_util_libs INTERFACE)
add_library(LiteRTLM::Runtime::Util ALIAS runtime_util_libs)

target_link_libraries(runtime_util_libs INTERFACE
  LiteRTLM::Runtime::Util::ConvertTensorBuffer
  LiteRTLM::Runtime::Util::DataStream
  LiteRTLM::Runtime::Util::ExecutorDataUtil
  LiteRTLM::Runtime::Util::FileDataStream
  LiteRTLM::Runtime::Util::FileFormatUtil
  LiteRTLM::Runtime::Util::FileUtil
  LiteRTLM::Runtime::Util::LitertLmLoader
  LiteRTLM::Runtime::Util::LitertLmStreamingLoader 
  LiteRTLM::Runtime::Util::LitertUtil
  LiteRTLM::Runtime::Util::Logging
  LiteRTLM::Runtime::Util::LogTensorBuffer 
  LiteRTLM::Runtime::Util::LoggingTensorBuffer
  LiteRTLM::Runtime::Util::LoraData
  LiteRTLM::Runtime::Util::LoraUtil
  LiteRTLM::Runtime::Util::MetadataUtil
  LiteRTLM::Runtime::Util::MetricsUtil
  LiteRTLM::Runtime::Util::ModelAssetBundleResources
  LiteRTLM::Runtime::Util::ModelTypeUtils
  LiteRTLM::Runtime::Util::ScopedFile
  LiteRTLM::Runtime::Util::TensorBufferUtil
  LiteRTLM::Runtime::Util::ZipReadonlyMemFile
  LiteRTLM::Runtime::Util::ZipUtils
)