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
# 1. Header Schema (Consuming the pre-generated FlatBuffer)
# ==============================================================================
add_litertlm_library(schema_core_litertlm_header_schema INTERFACE)
add_library(LiteRTLM::Schema::Core::HeaderSchema ALIAS schema_core_litertlm_header_schema)

target_include_directories(schema_core_litertlm_header_schema
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_header_schema
  INTERFACE
    LITERTLM_DEPS
)

# ==============================================================================
# 2. Header
# ==============================================================================
add_litertlm_library(schema_core_litertlm_header INTERFACE)
add_library(LiteRTLM::Schema::Core::Header ALIAS schema_core_litertlm_header)

target_include_directories(schema_core_litertlm_header
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_header
  INTERFACE
    LiteRTLM::Schema::Core::HeaderSchema
    LITERTLM_DEPS
)

# ==============================================================================
# 3. Utils
# ==============================================================================
add_litertlm_library(schema_core_litertlm_utils STATIC
  litertlm_utils.cc
)
add_library(LiteRTLM::Schema::Core::Utils ALIAS schema_core_litertlm_utils)

target_include_directories(schema_core_litertlm_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_utils
  PUBLIC
    LiteRTLM::Schema::Core::Header
    LiteRTLM::Schema::Core::HeaderSchema
    LITERTLM_DEPS
)

# ==============================================================================
# 4. Read
# ==============================================================================
add_litertlm_library(schema_core_litertlm_read STATIC
  litertlm_read.cc
)
add_library(LiteRTLM::Schema::Core::Read ALIAS schema_core_litertlm_read)

target_include_directories(schema_core_litertlm_read
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${TENSORFLOW_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_read
  PUBLIC
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    LiteRTLM::Runtime::Util::MemoryMappedFile
    LiteRTLM::Runtime::Util::ScopedFile
    LiteRTLM::Schema::Core::Header
    LiteRTLM::Schema::Core::HeaderSchema
    LiteRTLM::Schema::Core::Utils
    LITERTLM_DEPS
)

# ==============================================================================
# 5. Section
# ==============================================================================
add_litertlm_library(schema_core_litertlm_section INTERFACE)
add_library(LiteRTLM::Schema::Core::Section ALIAS schema_core_litertlm_section)

target_include_directories(schema_core_litertlm_section
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_section
  INTERFACE
    LiteRTLM::Schema::Core::HeaderSchema
    LiteRTLM::Runtime::Util::LiteRtStatusUtil
    zlib_lib
    LITERTLM_DEPS
)

# ==============================================================================
# 6. Print
# ==============================================================================
add_litertlm_library(schema_core_litertlm_print STATIC
  litertlm_print.cc
)
add_library(LiteRTLM::Schema::Core::Print ALIAS schema_core_litertlm_print)

target_include_directories(schema_core_litertlm_print
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_core_litertlm_print
  PUBLIC
    LiteRTLM::Schema::Core::HeaderSchema
    LiteRTLM::Schema::Core::Read
    LiteRTLM::Schema::Core::Utils
    LITERTLM_DEPS
)

# ==============================================================================
# 7. Folder Facade
# ==============================================================================
add_library(schema_core_libs INTERFACE)
add_library(LiteRTLM::Schema::Core ALIAS schema_core_libs)

target_link_libraries(schema_core_libs INTERFACE
  LiteRTLM::Schema::Core::Header
  LiteRTLM::Schema::Core::HeaderSchema
  LiteRTLM::Schema::Core::Print
  LiteRTLM::Schema::Core::Read
  LiteRTLM::Schema::Core::Section
  LiteRTLM::Schema::Core::Utils
)