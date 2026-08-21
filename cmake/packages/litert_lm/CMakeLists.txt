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

if(CMAKE_VERSION VERSION_LESS "3.25")
    message(FATAL_ERROR
        "CMake 3.25+ is required because we use the 'block()' command for variable isolation.\n"
        "Your current version is: ${CMAKE_VERSION}.\n"
        "Please upgrade CMake to proceed."
    )
endif()
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(LiteRT-LM LANGUAGES C CXX)

# --- CMake Settings ---
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.20 CACHE STRING "Required CMake policy version" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive" CACHE STRING "Required CXX flags" FORCE)

if(NOT LITERTLM_PROJECT_ROOT)
    message(WARNING "LITERTLM_PROJECT_ROOT is not defined. Setting LITERTLM_PROJECT_ROOT to ${CMAKE_CURRENT_SOURCE_DIR}/../../..}")
    set(LITERTLM_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "LiteRT-LM: Absolute path to the repository root")
endif()
set(CMAKE_SOURCE_DIR ${LITERTLM_PROJECT_ROOT} CACHE PATH 
    "Reset to repo root" FORCE)
message(DEBUG "[LiteRTLM] CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# --- Setup ---
set(LITERTLM_ORCHESTRATION_PHASE
    ${LITERTLM_ORCHESTRATION_PHASE} CACHE INTERNAL
    "LiteRT-LM: Phase of the super build" FORCE)
message(DEBUG "[LiteRTLM] LITERTLM_ORCHESTRATION_PHASE: ${LITERTLM_ORCHESTRATION_PHASE}")
set(LITERTLM_CMAKE_DIR ${LITERTLM_PROJECT_ROOT}/cmake
    CACHE PATH "LiteRT-LM: Directory containing cmake scripts")
set(LITERTLM_MODULES_DIR ${LITERTLM_CMAKE_DIR}/modules
    CACHE PATH "LiteRT-LM: Modules path, appended to CMAKE_MODULE_PATH")
message(DEBUG "[LiteRTLM] LITERTLM_MODULES_DIR: ${LITERTLM_MODULES_DIR}")
set(LITERTLM_CMAKE_PACKAGES_DIR
    ${LITERTLM_CMAKE_DIR}/packages CACHE PATH
    "LiteRT-LM: Directory for build packages")
message(DEBUG "[LiteRTLM] LITERTLM_CMAKE_PACKAGES_DIR: ${LITERTLM_CMAKE_PACKAGES_DIR}")

set(LITERTLM_PATCHES_DIR "${LITERTLM_CMAKE_DIR}/patches"
    CACHE PATH "LiteRT-LM: Directory for build patches")
set(LITERTLM_SCRIPTS_DIR "${LITERTLM_CMAKE_DIR}/scripts"
    CACHE PATH "LiteRT-LM: Directory for internal build and automation scripts")
set(LITERTLM_GENERATED_SRC_DIR "${CMAKE_BINARY_DIR}" 
    CACHE PATH "Root for generated code")
set(LITERTLM_STAGING_DIR "${CMAKE_BINARY_DIR}/../tmp/lib" 
    CACHE PATH "")
set(LITERTLM_THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party"
    CACHE PATH "")
set(LITERTLM_EXTERNAL_PROJECT_BIN_DIR "${CMAKE_BINARY_DIR}/external" 
    CACHE PATH "LiteRT-LM: Root directory for external projects")
set(LITERTLM_IMPORTED_SRC_DIR "${LITERTLM_CMAKE_DIR}/import/src" CACHE PATH "")
set(LITERTLM_IMPORTED_LITERT_SRC_DIR "${LITERTLM_IMPORTED_SRC_DIR}/litert" )
set(LITERTLM_INCLUDE_PATHS
  "${LITERTLM_GENERATED_SRC_DIR}"
  "${CMAKE_BINARY_DIR}"
  CACHE PATH "")

list(APPEND CMAKE_MODULE_PATH   "${LITERTLM_MODULES_DIR}")
set(LITERTLM_GENERATORS_DIR "${LITERTLM_MODULES_DIR}/generators" CACHE PATH "")

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
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/embedding_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/embedding_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/engine.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/litert_lm_metrics.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/sampler_params.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/token.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/executor/proto/constrained_decoding_options.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/executor_metadata.proto"
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

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_MODULES_DIR}/macros.cmake")
include("${LITERTLM_CMAKE_PACKAGES_DIR}/packages.cmake")

if("${LITERTLM_ORCHESTRATION_PHASE}" STREQUAL "litert_lm")
    set(LITERTLM_HOST_PROTOC "${LITERTLM_PREBUILD_PROTOC}" CACHE PATH "")
    set(LITERTLM_HOST_PROTOC_BIN_DIR "${LITERTLM_PREBUILD_PROTOC_BIN_DIR}" CACHE PATH "")
    set(LITERTLM_HOST_FLATC "${LITERTLM_PREBUILD_FLATC}" CACHE PATH "")
    set(LITERTLM_HOST_FLATC_BIN_DIR "${LITERTLM_PREBUILD_FLATC_BIN_DIR}" CACHE PATH "")
endif()

# --- Targeted Prebuild Execution
# Iterate through the required tools build scripts
if(DEFINED LITERTLM_PREBUILD_PHASE)
string(REPLACE " " ";" LITERTLM_PREBUILD_TARGETS "${LITERTLM_PREBUILD_TARGETS}")
    block()
        set(LITERTLM_EXTERNAL_PROJECT_BIN_DIR ${LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR})
        message(STATUS "[LiteRTLM] Prebuild Phase: Provisioning targeted tools: ${LITERTLM_PREBUILD_TARGETS}")        
        foreach(_target ${LITERTLM_PREBUILD_TARGETS})
            load_package(${_target})
        endforeach()
    endblock()
endif()

# --- Build ---
if(LITERTLM_BUILD_FETCH_CONTENT)
    message(STATUS "[LiteRTLM] Building FetchContent dependencies...")
    include("${LITERTLM_MODULES_DIR}/fetch_content.cmake")
else()
    message(STATUS "[LiteRTLM] Skipping FetchContent dependencies...")
endif()

# ExternalProjects Deps
if(LITERTLM_BUILD_EXTERNAL_PROJECTS)
    message(STATUS "[LiteRTLM] Building ExternalProject dependencies...")
    include("${LITERTLM_MODULES_DIR}/external_project.cmake")
else()
    message(STATUS "[LiteRTLM] Skipping ExternalProject dependencies...")
endif()

# LiteRT-LM
if("${LITERTLM_ORCHESTRATION_PHASE}" STREQUAL "litert_lm")
    message(STATUS "[LiteRTLM] Building LiteRT-LM...")

    include("${LITERTLM_MODULES_DIR}/collect_dependencies.cmake")
    include("${LITERTLM_MODULES_DIR}/generators.cmake")

    generate_src_files(LITERTLM_CORE_FILES ${ALL_SOURCE_FILES} ${ALL_HEADER_FILES})
    generate_src_files(LITERTLM_FLATBUFFERS_FILES ${ALL_SCHEMA_FILES})
    generate_src_files(LITERTLM_RUST_FILES ${ALL_RUST_FILES})

    add_custom_target(generator_complete
        COMMENT "--- ALL GENERATORS COMPLETE: RELEASING COMPILER GATE ---")
    add_dependencies(generator_complete
        LiteRTLM::Generated::Protobuf
        litertlm_cxx_bridge)

    if(NOT EXISTS "${LITERTLM_GENERATED_SRC_DIR}/runtime/components/constrained_decoding/gemma_model_constraint_provider.cc")
      file(COPY "${LITERTLM_PATCHES_DIR}/stubs/gemma_model_constraint_provider.cc"
        DESTINATION "${LITERTLM_GENERATED_SRC_DIR}/runtime/components/constrained_decoding")
    endif()

    set(LITERTLM_TENSORFLOW_MISSING_SRCS
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/allocation.cc"
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/mmap_allocation.cc"
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/mmap_allocation_disabled.cc"
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc"
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/core/model_builder_base.cc")
    set_source_files_properties(${LITERTLM_TENSORFLOW_MISSING_SRCS} PROPERTIES GENERATED TRUE)

    add_compile_definitions(
        absl_nullable=
        absl_nonnull=
        ENABLE_HUGGINGFACE_TOKENIZER
        ENABLE_SENTENCEPIECE_TOKENIZER)

    add_subdirectory("${LITERTLM_PROJECT_ROOT}/c" "${LITERTLM_GENERATED_SRC_DIR}/c")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/js" "${LITERTLM_GENERATED_SRC_DIR}/js")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/schema" "${LITERTLM_GENERATED_SRC_DIR}/schema")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/runtime" "${LITERTLM_GENERATED_SRC_DIR}/runtime")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/support" "${LITERTLM_GENERATED_SRC_DIR}/support")

    include("${LITERTLM_PACKAGE_DIR}/litert_lm_aggregate.cmake")
    generate_litertlm_aggregate()

    add_executable(litert_lm_main
        "${LITERTLM_GENERATED_SRC_DIR}/runtime/engine/litert_lm_main.cc")

    if(TARGET litertlm_anchor)
        add_dependencies(litert_lm_main litertlm_anchor)
    endif()

    target_include_directories(litert_lm_main PUBLIC
        ${LITERTLM_INCLUDE_PATHS})

    target_link_options(litert_lm_main PRIVATE
    "LINKER:--export-dynamic-symbol=LiteRt*")

    # --- Toolchain-Specific Linker Flags ---
    set(_LITERTLM_LINK_MULTIDEF "")
    set(_LITERTLM_LINK_GROUP_START "")
    set(_LITERTLM_LINK_GROUP_END "")
    set(_LITERTLM_LINK_WHOLE_START "")
    set(_LITERTLM_LINK_WHOLE_END "")
    set(_LITERTLM_SYSLIBS "")

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        if(APPLE)
            # macOS Linker (Mach-O)
            set(_LITERTLM_LINK_MULTIDEF "-Wl,-multiply_defined,suppress")
            set(_LITERTLM_LINK_WHOLE_START "-Wl,-all_load")
            set(_LITERTLM_LINK_WHOLE_END "")
            set(_LITERTLM_SYSLIBS "-lz" "-lpthread" "-ldl") # No librt on macOS
        elseif(ANDROID)
            # Android / Bionic (NO standalone rt or pthread)
            set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
            set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
            set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
            set(_LITERTLM_SYSLIBS "-lz -ldl -llog")
        else()
            # Linux Linker (ELF)
            set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
            set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
            set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
            set(_LITERTLM_LINK_WHOLE_START "-Wl,--whole-archive")
            set(_LITERTLM_LINK_WHOLE_END "-Wl,--no-whole-archive")
            set(_LITERTLM_SYSLIBS "-lz" "-lrt" "-lpthread" "-ldl")
        endif()
    elseif(MSVC)
        # Windows Linker
        set(_LITERTLM_LINK_MULTIDEF "/FORCE:MULTIPLE")
        set(_LITERTLM_LINK_WHOLE_START "/WHOLEARCHIVE")
        set(_LITERTLM_LINK_WHOLE_END "")
        set(_LITERTLM_SYSLIBS "") # MSVC defaults usually cover this
    endif()

    separate_arguments(_LITERTLM_ODML_EXPANDED_LIST NATIVE_COMMAND "${_LITERTLM_ODML_PAYLOAD}")
    separate_arguments(_LITERTLM_CORE_EXPANDED_LIST NATIVE_COMMAND "${_LITERTLM_CORE_PAYLOAD}")

    target_link_libraries(litert_lm_main
        PUBLIC
            ${_LITERTLM_LINK_MULTIDEF}
            ${_LITERTLM_LINK_GROUP_START}

            ${_LITERTLM_LINK_WHOLE_START}

            ${_LITERTLM_ODML_EXPANDED_LIST}
            ${_LITERTLM_CORE_EXPANDED_LIST}

            ${_LITERTLM_LINK_WHOLE_END}

            libpng_lib
            kissfft_lib
            miniaudio_lib
            minizip_lib
            minja_lib
            zlib_lib
            antlr_lib
            LiteRTLM::nlohmann_json::nlohmann_json
            opencl_headers_lib

            ${_LITERTLM_SYSLIBS}
            ${_LITERTLM_LINK_GROUP_END}
    )
else()
    message(STATUS "[LiteRTLM] Skipping target layer build (LITERTLM_BUILD_TARGETS=OFF)")
endif()