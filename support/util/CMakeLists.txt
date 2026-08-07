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

# ==============================================================================
# Memory Mapped File
# ==============================================================================

set(LITERT_MEMORY_MAPPED_FILE_SRC "")
if(WIN32)
  list(APPEND LITERT_MEMORY_MAPPED_FILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/memory_mapped_file_win.cc")
else()
  list(APPEND LITERT_MEMORY_MAPPED_FILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/memory_mapped_file_posix.cc")
endif()

add_litertlm_library(support_util_memory_mapped_file STATIC
  ${LITERT_MEMORY_MAPPED_FILE_SRC}
)
add_library(LiteRTLM::Support::Util::MemoryMappedFile ALIAS support_util_memory_mapped_file)

target_include_directories(support_util_memory_mapped_file
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_util_memory_mapped_file
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# IO Types
# ==============================================================================
add_litertlm_library(support_util_io_types STATIC
  io_types.cc
)
add_library(LiteRTLM::Support::Util::IoTypes ALIAS support_util_io_types)

target_include_directories(support_util_io_types
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_util_io_types
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Alloc Defs
# ==============================================================================
add_litertlm_library(support_util_alloc_defs STATIC
  alloc_defs.cc
)
add_library(LiteRTLM::Support::Util::AllocDefs ALIAS support_util_alloc_defs)

target_include_directories(support_util_alloc_defs
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_util_alloc_defs
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(support_util_libs INTERFACE)
add_library(LiteRTLM::Support::Util
  ALIAS support_util_libs)

target_link_libraries(support_util_libs INTERFACE
  LiteRTLM::Support::Util::MemoryMappedFile
  LiteRTLM::Support::Util::IoTypes
  LiteRTLM::Support::Util::AllocDefs
)
