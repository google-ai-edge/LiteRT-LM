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

# cl.exe and clang-cl. Tested before any GNU/Clang check so that clang-cl,
# which reports CMAKE_CXX_COMPILER_ID "Clang", never takes the GNU branches.
include("${CMAKE_CURRENT_LIST_DIR}/../../modules/litertlm_platform.cmake")
litertlm_is_msvc_frontend(LITERTLM_MSVC_FRONTEND)

# --- CMake Settings ---
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.20 CACHE STRING "Required CMake policy version" FORCE)
if(NOT LITERTLM_MSVC_FRONTEND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive" CACHE STRING "Required CXX flags" FORCE)
endif()

if(CMAKE_CROSSCOMPILING)
  set(_litertlm_cross_compile "ON")
else()
  set(_litertlm_cross_compile "OFF")
endif()

set(LITERTLM_CROSSCOMPILING "${_litertlm_cross_compile}" CACHE INTERNAL "LiteRTLM cross compile flag")

set(ENV{VIRTUAL_ENV} "${LITERTLM_VENV_DIR}")

if(LITERTLM_CROSSCOMPILING OR WIN32)
  # Windows: litert_lm_main does not embed Python, and the import library of
  # the host interpreter may not match the target architecture.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON3_LIB "")
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  set(PYTHON3_LIB Python3::Python)
endif()
include_directories(SYSTEM "${LITERTLM_VENV_DIR}/include")

message(STATUS "Using isolated Python: ${Python3_EXECUTABLE}")
if(CMAKE_HOST_WIN32)
set(LITERTLM_ENV_WRAPPER
"VIRTUAL_ENV=${LITERTLM_VENV_DIR}"
"--modify" "PATH=path_list_prepend:${LITERTLM_VENV_DIR}/Scripts"
CACHE INTERNAL "LiteRT-LM: Environment variables for uv virtual environment"
)
else()
set(LITERTLM_ENV_WRAPPER
"VIRTUAL_ENV=${LITERTLM_VENV_DIR}"
"PATH=${LITERTLM_VENV_DIR}/bin:$ENV{PATH}"
CACHE INTERNAL "LiteRT-LM: Environment variables for uv virtual environment"
)
endif()
set(LITERTLM_UV_ARGS
"-DPython3_ROOT_DIR=${LITERTLM_VENV_DIR}"
"-DPython3_FIND_VIRTUALENV=FIRST")

if(NOT LITERTLM_PROJECT_ROOT)
    message(WARNING "LITERTLM_PROJECT_ROOT is not defined. Setting LITERTLM_PROJECT_ROOT to ${CMAKE_CURRENT_SOURCE_DIR}/../../..}")
    set(LITERTLM_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "LiteRT-LM: Absolute path to the repository root")
endif()
set(CMAKE_SOURCE_DIR ${LITERTLM_PROJECT_ROOT} CACHE PATH
    "Reset to repo root" FORCE)
message(DEBUG "[LiteRTLM] CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# --- Setup ---
set(LITERTLM_ENV_WRAPPER "${LITERTLM_ENV_WRAPPER}" CACHE INTERNAL
    "LiteRT-LM: Environment variables for UV virtual environment")
set(LITERTLM_UV_ARGS "${LITERTLM_UV_ARGS}" CACHE INTERNAL
    "LiteRT-LM: Arguments for UV virtual environment")
set(LITERTLM_VENV_DIR "${LITERTLM_VENV_DIR}" CACHE PATH
    "LiteRT-LM: Path to the uv virtual environment")
set(LITERTLM_BUILD_COMMAND_ARGS "${LITERTLM_BUILD_COMMAND_ARGS}" CACHE INTERNAL
    "LiteRT-LM: Arguments for the build command")
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
if(NOT LITERTLM_CROSSCOMPILING AND DEFINED LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR)
    set(LITERTLM_EXTERNAL_PROJECT_BIN_DIR "${LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR}"
        CACHE PATH "LiteRT-LM: Root directory for external projects" FORCE)
else()
    set(LITERTLM_EXTERNAL_PROJECT_BIN_DIR "${CMAKE_BINARY_DIR}/external"
        CACHE PATH "LiteRT-LM: Root directory for external projects")
endif()
set(LITERTLM_IMPORTED_SRC_DIR "${LITERTLM_CMAKE_DIR}/import/src" CACHE PATH "")
set(LITERTLM_IMPORTED_LITERT_SRC_DIR "${LITERTLM_IMPORTED_SRC_DIR}/litert" )
set(LITERTLM_INCLUDE_PATHS
  "${LITERTLM_GENERATED_SRC_DIR}"
  "${CMAKE_BINARY_DIR}"
  CACHE PATH "")

set(LITERTLM_TARGET_MAP_PATH
    "${LITERTLM_CMAKE_PACKAGES_DIR}/litert_lm/${LITERTLM_ORCHESTRATION_PHASE}_target_map.cmake"
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

# --- Windows (MSVC ABI: cl.exe or clang-cl)
if(WIN32 AND LITERTLM_MSVC_FRONTEND)
  # The target maps are resolved before any dependency is built; keep every
  # entry here and let the link step drop the archives that were not built.
  set(LITERTLM_TARGET_MAP_PRUNE_MISSING OFF)

  # generate_target_map.cmake rewrites the checked-in *_target_map.cmake files
  # in the source tree. On Windows that would replace them with MSVC archive
  # names; kvp_parse_map() translates the checked-in maps instead.
  set(LITERTLM_GENERATE_TARGET_MAP OFF)

  # The compiler comes from the developer shell. Fail now, with the reason,
  # instead of with LNK1112 (machine type mismatch) at the very end.
  if(LITERTLM_WINDOWS_TARGET_ARCH AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
    string(TOUPPER "${LITERTLM_WINDOWS_TARGET_ARCH}" _litertlm_expected_arch)
    string(TOUPPER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" _litertlm_actual_arch)
    if(NOT _litertlm_expected_arch STREQUAL _litertlm_actual_arch)
      message(FATAL_ERROR
        "[LiteRTLM] ${CMAKE_CXX_COMPILER} targets ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID} "
        "but the build targets ${LITERTLM_WINDOWS_TARGET_ARCH}. Run the build from a "
        "Visual Studio developer shell for ${LITERTLM_WINDOWS_TARGET_ARCH}.")
    endif()
  endif()

  # Do not leave the Rust target to Corrosion's detection: pin it to the
  # architecture the C++ compiler targets (or to LITERTLM_RUST_TARGET), so a
  # cross-compiling developer shell cannot end up with Rust archives for the
  # build machine.
  litertlm_windows_rust_target(_litertlm_rust_target)
  if(_litertlm_rust_target)
    set(LITERTLM_RUST_TARGET "${_litertlm_rust_target}" CACHE STRING
        "Rust target triple (e.g., aarch64-linux-android). Leave empty for host." FORCE)
    set(Rust_CARGO_TARGET "${_litertlm_rust_target}" CACHE STRING
        "Rust target triple used by Corrosion" FORCE)
    message(STATUS "[LiteRTLM] Rust target: ${_litertlm_rust_target}")
  endif()

  # Forwarded to every ExternalProject so that all of them agree on the build
  # type, the C runtime and the warning/charset flags. Projects configured
  # through an explicit CONFIGURE_COMMAND also need the generator: without -G
  # CMake falls back to Visual Studio.
  set(LITERTLM_EP_PLATFORM_ARGS
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
      "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
      "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
  )
  set(LITERTLM_EP_GENERATOR_ARGS "-G" "${CMAKE_GENERATOR}")
  if(CMAKE_MAKE_PROGRAM)
    list(APPEND LITERTLM_EP_GENERATOR_ARGS "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
  endif()
endif()

set(LITERTLM_PROTO_FILES
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/any2any_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/any2any_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/any2any_session_config.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/embedding_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/embedding_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/engine.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/image_gen_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/litert_lm_metrics.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/llm_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/sampler_params.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/speech2text_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/speech2text_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/speech2text_session_config.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/text2image_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/text2image_session_config.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/text2speech_metadata.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/text2speech_model_type.proto"
  "${LITERTLM_PROJECT_ROOT}/runtime/proto/text2speech_session_config.proto"
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

# --- Feature Flags ---
option(LITERT_LM_ENABLE_SENTENCEPIECE_TOKENIZER
    "Enable SentencePiece tokenizer support"
    ON)
option(LITERT_LM_ENABLE_HUGGINGFACE_TOKENIZER
    "Enable HuggingFace BPE tokenizer support"
    ON)
option(LITERT_LM_ENABLE_MINIJINJA_TEMPLATE
    "Enable Minijinja prompt template engine"
    ON)
option(LITERT_LM_ENABLE_CONSTRAINED_DECODING
    "Enable constrained decoding support"
    ON)
option(LITERT_LM_ENABLE_LLGUIDANCE
    "Enable LlGuidance constrained decoding backend"
    ON)
option(LITERT_LM_ENABLE_FST_CONSTRAINTS
    "Enable FST constraint provider"
    ON)
option(LITERT_LM_ENABLE_TOOL_USE_PARSER
    "Enable tool-use expression parsers"
    ON)
option(LITERT_LM_ENABLE_GEMMA_PROCESSORS
    "Enable Gemma family model data processors"
    ON)
option(LITERT_LM_ENABLE_QWEN_PROCESSOR
    "Enable Qwen family model data processor"
    ON)
option(LITERT_LM_ENABLE_FASTVLM_PROCESSOR
    "Enable FastVLM model data processor"
    ON)
option(LITERT_LM_ENABLE_LFM2_PROCESSOR
    "Enable LFM2 model data processor"
    ON)
option(LITERT_LM_ENABLE_MINICPM_PROCESSOR
    "Enable MiniCPM family model data processors"
    ON)

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
if(NOT LITERTLM_MSVC_FRONTEND)
add_compile_options("-fpermissive")
endif()

# --- Optional
option(_unverified_targets "Builds executables that are still in progress" OFF)

# --- Centralized ExternalProject & FetchContent Optimizations ---
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)

include(ExternalProject)

file(READ "${CMAKE_ROOT}/Modules/ExternalProject.cmake" _ep_content)
string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_ROOT}/Modules" _ep_content "${_ep_content}")
string(REPLACE "\${CMAKE_CURRENT_FUNCTION_LIST_DIR}" "${CMAKE_ROOT}/Modules" _ep_content "${_ep_content}")
string(REPLACE "function(ExternalProject_Add " "function(_ExternalProject_Add " _ep_content "${_ep_content}")
cmake_language(EVAL CODE "${_ep_content}")

function(ExternalProject_Add name)
    cmake_parse_arguments(PARSE_ARGV 1 _EP "" "GIT_REPOSITORY;GIT_TAG" "")

    set(_call_args "[===[${name}]===]")
    set(_skip_section FALSE)
    set(_is_commit_sha FALSE)
    if(_EP_GIT_REPOSITORY AND _EP_GIT_TAG)
        string(LENGTH "${_EP_GIT_TAG}" _tag_len)
        if(_tag_len EQUAL 40 AND _EP_GIT_TAG MATCHES "^[0-9a-fA-F]+$")
            set(_is_commit_sha TRUE)
        endif()
    endif()

    set(_ep_keywords
        DEPENDS PREFIX LIST_SEPARATOR TMP_DIR STAMP_DIR DOWNLOAD_DIR
        SOURCE_DIR SOURCE_SUBDIR BINARY_DIR INSTALL_DIR
        DOWNLOAD_COMMAND URL URL_HASH URL_MD5 DOWNLOAD_NAME DOWNLOAD_NO_EXTRACT
        GIT_REPOSITORY GIT_TAG GIT_REMOTE_NAME GIT_SUBMODULES GIT_SUBMODULES_RECURSE
        GIT_SHALLOW GIT_PROGRESS GIT_CONFIG GIT_REMOTE_UPDATE_STRATEGY
        UPDATE_COMMAND UPDATE_DISCONNECTED
        PATCH_COMMAND
        CONFIGURE_COMMAND CMAKE_COMMAND CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM
        CMAKE_GENERATOR_TOOLSET CMAKE_GENERATOR_INSTANCE CMAKE_ARGS CMAKE_CACHE_ARGS
        CMAKE_CACHE_DEFAULT_ARGS
        BUILD_COMMAND BUILD_IN_SOURCE BUILD_ALWAYS BUILD_BYPRODUCTS
        INSTALL_COMMAND
        TEST_COMMAND TEST_BEFORE_INSTALL TEST_AFTER_INSTALL TEST_EXCLUDE_FROM_MAIN
        LOG_DOWNLOAD LOG_UPDATE LOG_PATCH LOG_CONFIGURE LOG_BUILD LOG_INSTALL LOG_TEST
        LOG_MERGED_STDOUTERR LOG_OUTPUT_ON_FAILURE
        USES_TERMINAL_DOWNLOAD USES_TERMINAL_UPDATE USES_TERMINAL_PATCH
        USES_TERMINAL_CONFIGURE USES_TERMINAL_BUILD USES_TERMINAL_INSTALL USES_TERMINAL_TEST
        STEP_TARGETS INDEPENDENT_STEP_TARGETS
    )

    math(EXPR _last_idx "${ARGC} - 1")
    foreach(_i RANGE 1 ${_last_idx})
        set(_arg "${ARGV${_i}}")
        if(_arg IN_LIST _ep_keywords)
            if(_arg STREQUAL "UPDATE_COMMAND" OR _arg STREQUAL "GIT_SHALLOW")
                set(_skip_section TRUE)
            elseif(_is_commit_sha AND (_arg STREQUAL "GIT_REPOSITORY" OR _arg STREQUAL "GIT_TAG"))
                set(_skip_section TRUE)
            else()
                set(_skip_section FALSE)
                string(APPEND _call_args " [===[${_arg}]===]")
            endif()
        elseif(NOT _skip_section)
            string(APPEND _call_args " [===[${_arg}]===]")
        endif()
    endforeach()

    if(_EP_GIT_REPOSITORY)
        if(_is_commit_sha)
            string(APPEND _call_args
                " DOWNLOAD_COMMAND [===[${CMAKE_COMMAND}]===]"
                " [===[-DSRC_DIR=<SOURCE_DIR>]===]"
                " [===[-DREPO_URL=${_EP_GIT_REPOSITORY}]===]"
                " [===[-DTAG=${_EP_GIT_TAG}]===]"
                " -P [===[${LITERTLM_SCRIPTS_DIR}/shallow_git_fetch.cmake]===]"
                " UPDATE_COMMAND [===[]===]"
            )
        else()
            string(APPEND _call_args
                " GIT_SHALLOW TRUE"
                " GIT_CONFIG [===[filter.lfs.smudge=]===] [===[filter.lfs.process=]===] [===[filter.lfs.required=false]===]"
                " UPDATE_COMMAND [===[]===]"
            )
        endif()
    endif()

    cmake_language(EVAL CODE "_ExternalProject_Add(${_call_args})")
endfunction()

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_MODULES_DIR}/macros.cmake")
include("${LITERTLM_CMAKE_PACKAGES_DIR}/packages.cmake")

if("${LITERTLM_ORCHESTRATION_PHASE}" STREQUAL "litert_lm")
    set(LITERTLM_HOST_PROTOC "${LITERTLM_HOST_PROTOC}" CACHE PATH "")
    set(LITERTLM_HOST_PROTOC_BIN_DIR "${LITERTLM_HOST_PROTOC_BIN_DIR}" CACHE PATH "")
    set(LITERTLM_HOST_FLATC "${LITERTLM_HOST_FLATC}" CACHE PATH "")
    set(LITERTLM_HOST_FLATC_BIN_DIR "${LITERTLM_HOST_FLATC_BIN_DIR}" CACHE PATH "")
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
        absl_nonnull=)
    if(LITERT_LM_ENABLE_HUGGINGFACE_TOKENIZER)
        add_compile_definitions(ENABLE_HUGGINGFACE_TOKENIZER)
    endif()
    if(LITERT_LM_ENABLE_SENTENCEPIECE_TOKENIZER)
        add_compile_definitions(ENABLE_SENTENCEPIECE_TOKENIZER)
    endif()
    if(NOT LITERT_LM_ENABLE_CONSTRAINED_DECODING)
        add_compile_definitions(
            LITERT_LM_CONSTRAINED_DECODING_DISABLED
            LITERT_LM_FST_CONSTRAINTS_DISABLED)
    endif()
    if(NOT LITERT_LM_ENABLE_FST_CONSTRAINTS)
        add_compile_definitions(LITERT_LM_FST_CONSTRAINTS_DISABLED)
    endif()
    if(WIN32)
        # The ANTLR runtime is linked statically; without this its headers
        # declare every class __declspec(dllimport).
        add_compile_definitions(ANTLR4CPP_STATIC)
    endif()

    add_subdirectory("${LITERTLM_PROJECT_ROOT}/c" "${LITERTLM_GENERATED_SRC_DIR}/c")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/js" "${LITERTLM_GENERATED_SRC_DIR}/js")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/schema" "${LITERTLM_GENERATED_SRC_DIR}/schema")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/runtime" "${LITERTLM_GENERATED_SRC_DIR}/runtime")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/support" "${LITERTLM_GENERATED_SRC_DIR}/support")
    add_subdirectory("${LITERTLM_PROJECT_ROOT}/sync" "${LITERTLM_GENERATED_SRC_DIR}/sync")

    include("${LITERTLM_PACKAGE_DIR}/litert_lm_aggregate.cmake")
    generate_litertlm_aggregate()

    add_executable(litert_lm_main
        "${LITERTLM_GENERATED_SRC_DIR}/runtime/engine/litert_lm_main.cc")

    if(TARGET litertlm_anchor)
        add_dependencies(litert_lm_main litertlm_anchor)
    endif()

    target_include_directories(litert_lm_main PUBLIC
        ${LITERTLM_INCLUDE_PATHS})

    if(NOT LITERTLM_MSVC_FRONTEND)
    target_link_options(litert_lm_main PRIVATE
    "LINKER:--export-dynamic-symbol=LiteRt*")
    endif()

    if(LITERTLM_MSVC_FRONTEND)
    # --- Windows (cl.exe / clang-cl): see modules/windows_link.cmake ---
    include("${LITERTLM_MODULES_DIR}/windows_link.cmake")

    set(_litertlm_win_whole ${_litertlm_lib_paths})
    set(_litertlm_win_libs
        "${_litert_lib_paths}"
        "${_tflite_lib_paths}"
        "${_tokenizers_lib_paths}"
        "${_sentencepiece_lib_paths}"
        "${_re2_lib_paths}"
        "${_flatbuffers_lib_paths}"
        "${_protobuf_lib_paths}"
        "${_absl_lib_paths}"
        "${_cxxbridge_paths}"
    )

    if(LITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER)
        # Replace the stub provider with the prebuilt DLL for this architecture.
        litertlm_windows_prebuilt_gemma_provider(_litertlm_gemma_lib _litertlm_gemma_dll)
        list(FILTER _litertlm_win_whole EXCLUDE REGEX "gemma_model_constraint_provider")
        list(APPEND _litertlm_win_libs "${_litertlm_gemma_lib}")
        add_custom_command(TARGET litert_lm_main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_litertlm_gemma_dll}" "$<TARGET_FILE_DIR:litert_lm_main>"
            COMMENT "[LiteRTLM] Copying libGemmaModelConstraintProvider.dll next to litert_lm_main"
            VERBATIM)
    endif()

    if(QAIRT_HEADERS_DIR)
        # Copy the Qualcomm NPU dispatch library built by litert_external next
        # to litert_lm_main.exe so `--litert_dispatch_lib_dir` can point at the
        # output directory directly.
        set(_litertlm_qnn_dispatch_dll
            "${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/dispatch/LiteRtDispatch_Qualcomm.dll")
        add_custom_command(TARGET litert_lm_main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_litertlm_qnn_dispatch_dll}" "$<TARGET_FILE_DIR:litert_lm_main>"
            COMMENT "[LiteRTLM] Copying LiteRtDispatch_Qualcomm.dll next to litert_lm_main"
            VERBATIM)
    endif()

    litertlm_windows_link_executable(litert_lm_main
        WHOLE ${_litertlm_win_whole}
        LIBS ${_litertlm_win_libs}
        DEPENDS
            litertlm_anchor
            litert_external
            tflite_external
            tokenizers-cpp_external
            sentencepiece_external
            re2_external
            flatbuffers_external
            protobuf_external
            absl_external
            litertlm_cxx_bridge
    )

    # Rust: link Corrosion's imported targets rather than the archives by path,
    # so the system libraries the Rust standard library needs (ws2_32, userenv,
    # ntdll, bcrypt, ...) come with them.
    target_link_libraries(litert_lm_main
        PRIVATE
            litertlm_cxx_bridge
            litert_lm_deps

            libpng_lib
            kissfft_lib
            miniaudio_lib
            minizip_lib
            minja_lib
            highway_lib
            zlib_lib
            antlr_lib
            LiteRTLM::nlohmann_json::nlohmann_json
            opencl_headers_lib
    )
    else()
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
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
                execute_process(
                    COMMAND ${CMAKE_CXX_COMPILER} -fuse-ld=lld -Wl,--version
                    OUTPUT_QUIET ERROR_QUIET
                    RESULT_VARIABLE _LITERTLM_LLD_CHECK
                )
                if(_LITERTLM_LLD_CHECK EQUAL 0)
                    message(STATUS "[LiteRTLM] Using LLD linker")
                    set(_LITERTLM_LINK_MULTIDEF "-fuse-ld=lld ${_LITERTLM_LINK_MULTIDEF}")
                endif()
            endif()
            set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
            set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
            set(_LITERTLM_LINK_WHOLE_START "-Wl,--whole-archive")
            set(_LITERTLM_LINK_WHOLE_END "-Wl,--no-whole-archive")
            set(_LITERTLM_SYSLIBS "-lz" "-lrt" "-lpthread" "-ldl")
        endif()
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
            highway_lib
            zlib_lib
            antlr_lib
            LiteRTLM::nlohmann_json::nlohmann_json
            opencl_headers_lib

            ${PYTHON3_LIB}

            ${_LITERTLM_SYSLIBS}
            ${_LITERTLM_LINK_GROUP_END}
    )
    endif()
else()
    message(STATUS "[LiteRTLM] Skipping target layer build (LITERTLM_BUILD_TARGETS=OFF)")
endif()
