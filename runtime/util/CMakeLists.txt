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
# 0. Local Proto: External File
# ==============================================================================
# set(EXTERNAL_FILE_PROTO "external_file.proto")
# set(EXTERNAL_FILE_PROTO_SRCS "${CMAKE_BINARY_DIR}/runtime/util/external_file.pb.cc")
# set(EXTERNAL_FILE_PROTO_HDRS "${CMAKE_BINARY_DIR}/runtime/util/external_file.pb.h")

# add_custom_command(
#   OUTPUT ${EXTERNAL_FILE_PROTO_SRCS} ${EXTERNAL_FILE_PROTO_HDRS}
#   COMMAND ${PROTO_COMPILER}
#   ARGS --cpp_out=${CMAKE_BINARY_DIR} -I${PROJECT_SOURCE_DIR} ${PKG_ROOT}/${EXTERNAL_FILE_PROTO}
#   DEPENDS ${PKG_ROOT}/${EXTERNAL_FILE_PROTO}
#   COMMENT "Generating C++ code for external_file.proto"
#   VERBATIM
# )

# add_litertlm_library(runtime_util_external_file_cc_proto STATIC
#   ${EXTERNAL_FILE_PROTO_SRCS}
#   ${EXTERNAL_FILE_PROTO_HDRS}
# )
# target_include_directories(runtime_util_external_file_cc_proto
#   PUBLIC
#     ${LITERTLM_INCLUDE_PATHS}
#     ${CMAKE_BINARY_DIR}
#   )
# target_link_libraries(runtime_util_external_file_cc_proto PUBLIC LITERTLM_DEPS)

# ==============================================================================
# Convert Tensor Buffer
# ==============================================================================
add_litertlm_library(runtime_util_convert_tensor_buffer INTERFACE)
add_library(LiteRTLM::Runtime::Util::ConvertTensorBuffer ALIAS runtime_util_convert_tensor_buffer)

target_include_directories(runtime_util_convert_tensor_buffer
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_convert_tensor_buffer
  INTERFACE
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
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_tensor_buffer_util
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT Status Util
# ==============================================================================
add_litertlm_library(runtime_util_litert_status_util INTERFACE)
add_library(LiteRTLM::Runtime::Util::LiteRtStatusUtil ALIAS runtime_util_litert_status_util)

target_include_directories(runtime_util_litert_status_util
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_litert_status_util
  INTERFACE
    LITERTLM_DEPS
)

# ==============================================================================
# Scoped File (Platform Specific)
# ==============================================================================
set(LITERT_CC_INTERNAL_DIR "${LITERT_SRC_DIR}/cc/internal/" CACHE PATH "")
set(SCOPED_FILE_SRCS "${LITERT_CC_INTERNAL_DIR}/scoped_file.cc")
if(WIN32)
  list(APPEND SCOPED_FILE_SRCS "${LITERT_CC_INTERNAL_DIR}/scoped_file_win.cc")
else()
  list(APPEND SCOPED_FILE_SRCS "${LITERT_CC_INTERNAL_DIR}/scoped_file_posix.cc")
endif()

add_litertlm_library(runtime_util_scoped_file STATIC
  ${SCOPED_FILE_SRCS}
)
add_library(LiteRTLM::Runtime::Util::ScopedFile ALIAS runtime_util_scoped_file)

target_include_directories(runtime_util_scoped_file
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_scoped_file
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LITERTLM_DEPS
)

# ==============================================================================
# Memory Mapped File (Platform Specific)
# ==============================================================================
if(WIN32)
  set(MMAP_SRC memory_mapped_file_win.cc)
else()
  set(MMAP_SRC memory_mapped_file_posix.cc)
endif()

add_litertlm_library(runtime_util_memory_mapped_file STATIC
  ${MMAP_SRC}
)
add_library(LiteRTLM::Runtime::Util::MemoryMappedFile ALIAS runtime_util_memory_mapped_file)

if(WIN32)
  target_compile_definitions(runtime_util_memory_mapped_file PRIVATE _WIN32_WINNT=_WIN32_WINNT_WIN10)
endif()

target_include_directories(runtime_util_memory_mapped_file
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_memory_mapped_file
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
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
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_file_util
  PUBLIC
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
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_file_format_util
  PUBLIC
    runtime_executor_executor_settings_base
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::MemoryMappedFile
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
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_executor_data_util
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::TensorBufferUtil
    runtime_executor_llm_executor_io_types # [Restored explicit link]
    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT LM Loader
# ==============================================================================
add_litertlm_library(runtime_util_litert_lm_loader STATIC
  litert_lm_loader.cc
)
add_library(LiteRTLM::Runtime::Util::LMLoader ALIAS runtime_util_litert_lm_loader)

target_include_directories(runtime_util_litert_lm_loader
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_litert_lm_loader
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::MemoryMappedFile
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Schema::Core
    LITERTLM_DEPS
    LiteRTLM::Runtime::Components::ModelResources::Interface # Components
)

# ==============================================================================
# Logging
# ==============================================================================
add_litertlm_library(runtime_util_logging INTERFACE)
add_library(LiteRTLM::Runtime::Util::Logging ALIAS runtime_util_logging)

target_include_directories(runtime_util_logging
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_logging
  INTERFACE
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
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_logging_tensor_buffer
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
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_log_tensor_buffer
  PUBLIC
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA Util
# ==============================================================================
add_litertlm_library(runtime_util_lora_util STATIC
  lora_util.cc
)
add_library(LiteRTLM::Runtime::Util::LoRAUtil ALIAS runtime_util_lora_util)

target_include_directories(runtime_util_lora_util
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_lora_util
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::MemoryMappedFile
    LiteRTLM::Runtime::Util::ScopedFile
    LITERTLM_DEPS
)

# ==============================================================================
# LoRA Data
# ==============================================================================
add_litertlm_library(runtime_util_lora_data STATIC
  lora_data.cc
)
add_library(LiteRTLM::Runtime::Util::LoRAData ALIAS runtime_util_lora_data)

target_include_directories(runtime_util_lora_data
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_lora_data
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::LoRAUtil
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
    ${GENERATED_SRC_DIR}
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
    ${GENERATED_SRC_DIR}
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
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}
)

target_link_libraries(runtime_util_zip_utils
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::ZipReadonlyMemFile
    LITERTLM_DEPS
)

# ==============================================================================
# Model Asset Bundle Resources
# ==============================================================================
add_litertlm_library(runtime_util_model_asset_bundle_resources STATIC
  model_asset_bundle_resources.cc
)
add_library(LiteRTLM::Runtime::Util::ModelAssetBundleResources ALIAS runtime_util_model_asset_bundle_resources)

target_include_directories(runtime_util_model_asset_bundle_resources
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_model_asset_bundle_resources
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::MemoryMappedFile
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Runtime::Util::ZipUtils
    LITERTLM_DEPS
)

# ==============================================================================
# Model Type Utils
# ==============================================================================
add_litertlm_library(runtime_util_model_type_utils STATIC
  model_type_utils.cc
)
add_library(LiteRTLM::Runtime::Util::ModelTypeUtils ALIAS runtime_util_model_type_utils)

target_include_directories(runtime_util_model_type_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_model_type_utils
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_util_libs INTERFACE)
add_library(LiteRTLM::Runtime::Util ALIAS runtime_util_libs)

target_link_libraries(runtime_util_libs INTERFACE
  LiteRTLM::Runtime::Util::ConvertTensorBuffer
  LiteRTLM::Runtime::Util::ExecutorDataUtil
  LiteRTLM::Runtime::Util::FileFormatUtil
  LiteRTLM::Runtime::Util::FileUtil
  LiteRTLM::Runtime::Util::LMLoader
  LiteRTLM::Runtime::Util::LoggingTensorBuffer
  LiteRTLM::Runtime::Util::LogTensorBuffer
  LiteRTLM::Runtime::Util::LoRAData
  LiteRTLM::Runtime::Util::LoRAUtil
  LiteRTLM::Runtime::Util::MemoryMappedFile
  LiteRTLM::Runtime::Util::MetadataUtil
  LiteRTLM::Runtime::Util::ModelAssetBundleResources
  LiteRTLM::Runtime::Util::ModelTypeUtils
  LiteRTLM::Runtime::Util::ScopedFile
  LiteRTLM::Runtime::Util::TensorBufferUtil
  LiteRTLM::Runtime::Util::ZipReadonlyMemFile
  LiteRTLM::Runtime::Util::ZipUtils
)