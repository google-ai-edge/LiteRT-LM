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

set(LITERTLM_CMAKE_PACKAGES_DIR "${LITERTLM_CMAKE_DIR}/packages" 
    CACHE PATH "LiteRT-LM: Directory for build packages")
set(LITERTLM_PATCHES_DIR "${LITERTLM_CMAKE_DIR}/patches"
    CACHE PATH "LiteRT-LM: Directory for build patches")
set(LITERTLM_SCRIPTS_DIR "${LITERTLM_CMAKE_DIR}/scripts"
    CACHE PATH "LiteRT-LM: Directory for internal build and automation scripts")

set(LITERTLM_GENERATED_SRC_DIR "${CMAKE_BINARY_DIR}/generated/src" 
    CACHE PATH "Root for generated code")
set(LITERTLM_STAGING_DIR "${CMAKE_BINARY_DIR}/staging/lib" 
    CACHE PATH "")
set(LITERTLM_THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party"
    CACHE PATH "")

set(LITERTLM_EXTERNAL_PROJECT_BIN_DIR "${CMAKE_BINARY_DIR}/external" 
    CACHE PATH "LiteRT-LM: Root directory for external projects")
# set(LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR "${LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR}"
#     CACHE PATH "LiteRT-LM: Root directory for prebuild external projects")

set(LITERTLM_INCLUDE_PATHS
  "${LITERTLM_GENERATED_SRC_DIR}"
  "${CMAKE_BINARY_DIR}"
  CACHE PATH "")

list(APPEND CMAKE_MODULE_PATH   "${LITERTLM_MODULES_DIR}")
set(LITERTLM_GENERATORS_DIR "${LITERTLM_MODULES_DIR}/generators" CACHE INTERNAL "")



set_property(GLOBAL PROPERTY LITERTLM_ARCHIVE_REGISTRY "")
set_property(GLOBAL PROPERTY LITERTLM_TARGET_REGISTRY "")



# --- Rust
set(Rust_RUSTUP_INSTALL_MISSING_TARGET ON CACHE BOOL "Auto-install rust targets")
set(LITERTLM_RUST_TARGET "" CACHE STRING "Rust target triple (e.g., aarch64-linux-android). Leave empty for host.")
set(LITERTLM_RUST_FILES
    "${LITERTLM_PROJECT_ROOT}/runtime/components/rust/minijinja_template.rs"
    "${LITERTLM_PROJECT_ROOT}/runtime/components/tool_use/rust/parsers.rs"
    CACHE INTERNAL "Rust files to include in the cxx bridge")
set(LITERTLM_CARGO_TOML "${LITERTLM_PROJECT_ROOT}/Cargo.toml" CACHE PATH "Path to LiteRT-LM's Cargo.toml")
if(DEFINED LITERTLM_RUST_LINKER_OVERRIDE AND DEFINED LITERTLM_RUST_CARGO_ENV_VAR)
    set(ENV{${LITERTLM_RUST_CARGO_ENV_VAR}} "${LITERTLM_RUST_LINKER_OVERRIDE}")
    message(STATUS "[LiteRTLM] Routing Rust Cargo: ${LITERTLM_RUST_LINKER_OVERRIDE}")
endif()
if(DEFINED LITERTLM_RUST_LINKER_OVERRIDE AND DEFINED LITERTLM_CORROSION_LINKER_VAR)
    set(${LITERTLM_CORROSION_LINKER_VAR} "${LITERTLM_RUST_LINKER_OVERRIDE}" CACHE FILEPATH "Override Rust Linker")
    message(STATUS "[LiteRTLM] Routing Rust Corrosion Linker: ${LITERTLM_RUST_LINKER_OVERRIDE}")
endif()

set(LITERTLM_PROTO_FILES
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/engine.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/sampler_params.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/token.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/executor/proto/constrained_decoding_options.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/util/external_file.proto"
  CACHE INTERNAL
  "Protobuf files that will be processed with protoc")

set(LITERTLM_FLATBUFFERS_FILES
    "${LITERTLM_PROJECT_ROOT}/schema/core/litertlm_header_schema.fbs"
    CACHE INTERNAL
    "FlatBuffers files that will be processed with flatc")


# --- Directory Generation
file(MAKE_DIRECTORY "${LITERTLM_STAGING_DIR}")
file(MAKE_DIRECTORY "${LITERTLM_GENERATED_SRC_DIR}")
file(MAKE_DIRECTORY "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}")

# --- Build
option(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL
    "LiteRT definition to disable GPU in build config"
    TRUE)
option(LITERT_BUILD_CONFIG_DISABLE_NPU_VAL
    "LiteRT definition to disable NPU in build config"
    TRUE)

option(LITERTLM_BUILD_FETCH_CONTENT 
    "Enable downloading and building FetchContent dependencies" 
    ON)
option(LITERTLM_BUILD_EXTERNAL_PROJECTS 
    "Enable building ExternalProject dependencies"
    ON)  
option(LITERTLM_BUILD_TARGETS
    "Enable building the primary LiteRT-LM targets and executables"
    ON)

# --- Compiler Flags
add_compile_options("-fpermissive")


# --- Optional
option(_unverified_targets "Builds executables that are still in progress" OFF)

option(ENABLE_CCACHE "Enables the use of ccache" OFF)
if(${ENABLE_CCACHE} STREQUAL "ON")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      message(STATUS "Using CCache to speed up builds")
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
      message(WARNING "CCache not found. Install it (`sudo apt install ccache`) for faster builds!")
  endif()
endif()

