# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# CMakeLists.txt for LiteRT-LM-Orchestrator.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(LiteRT-LM-Orchestrator LANGUAGES NONE)

# Host tool paths are only relevant when cross-compiling — the prebuild phase
# below runs only if LITERTLM_TOOLCHAIN_ARGS is non-empty. For a native build,
# leaving LITERTLM_HOST_PROTOC / LITERTLM_HOST_FLATC undefined lets the inner
# project build its own protoc/flatc via ExternalProject.
if(LITERTLM_TOOLCHAIN_ARGS)
    if(NOT DEFINED LITERTLM_HOST_PROTOC)
        set(LITERTLM_HOST_PROTOC "${CMAKE_CURRENT_BINARY_DIR}/prebuild/build/external/protobuf/install/bin/protoc")
        set(LITERTLM_HOST_PROTOC_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/prebuild/build/external/protobuf/install/bin")
    endif()

    if(NOT DEFINED LITERTLM_HOST_FLATC)
        set(LITERTLM_HOST_FLATC "${CMAKE_CURRENT_BINARY_DIR}/prebuild/build/external/flatbuffers/install/bin/flatc")
        set(LITERTLM_HOST_FLATC_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/prebuild/build/external/flatbuffers/install/bin")
    endif()
endif()

include(ExternalProject)

set(LITERTLM_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "LiteRT-LM: Absolute path to the repository root")
set(LITERTLM_TOOLCHAIN_ARGS "" CACHE STRING "String used to define parameterized CMAKE_ARGS")

set(_PREBUILD_DEPENDENCY "")

if(LITERTLM_TOOLCHAIN_ARGS)
    message(STATUS "[LiteRTLM] Toolchain args detected. Generating Host Prebuild phase...")
    ExternalProject_Add(
        litert_lm_prebuild

        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/packages/litert_lm"

        PREFIX     "${CMAKE_CURRENT_BINARY_DIR}/prebuild"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/prebuild/build"
        STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/prebuild/stamps"
        TMP_DIR    "${CMAKE_CURRENT_BINARY_DIR}/prebuild/tmp"
        DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/prebuild/download"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/prebuild/install"

        CMAKE_ARGS
            "-DLITERTLM_PROJECT_ROOT=${LITERTLM_PROJECT_ROOT}"

        INSTALL_COMMAND ""
    )
    set(_PREBUILD_DEPENDENCY "litert_lm_prebuild")
else()
    message(STATUS "[LiteRTLM] Native build detected. Skipping Host Prebuild phase.")
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    set(LITERTLM_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
endif()

if(DEFINED LITERTLM_TOOLCHAIN_SCRIPT)
    message(STATUS "[LiteRTLM] Executing toolchain script -> ${LITERTLM_TOOLCHAIN_SCRIPT}")
    include("${LITERTLM_TOOLCHAIN_SCRIPT}")
endif()

ExternalProject_Add(
    litert_lm

    DEPENDS ${_PREBUILD_DEPENDENCY}

    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/packages/litert_lm"

    PREFIX     "${CMAKE_CURRENT_BINARY_DIR}/litert_lm"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_lm/build"
    STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/litert_lm/stamps"
    TMP_DIR    "${CMAKE_CURRENT_BINARY_DIR}/litert_lm/tmp"
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_lm/download"
    INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_lm/install"

    CMAKE_ARGS
        ${LITERTLM_TOOLCHAIN_FILE}
        "-DLITERTLM_TOOLCHAIN_FILE=${LITERTLM_TOOLCHAIN_FILE}"

        ${LITERTLM_TOOLCHAIN_ARGS}
        "-DLITERTLM_TOOLCHAIN_ARGS=${LITERTLM_TOOLCHAIN_ARGS}"

        "-DLITERTLM_PROJECT_ROOT=${LITERTLM_PROJECT_ROOT}"
        "-DLITERTLM_HOST_PROTOC=${LITERTLM_HOST_PROTOC}"
        "-DLITERTLM_HOST_PROTOC_BIN_DIR=${LITERTLM_HOST_PROTOC_BIN_DIR}"
        "-DLITERTLM_HOST_FLATC=${LITERTLM_HOST_FLATC}"
        "-DLITERTLM_HOST_FLATC_BIN_DIR=${LITERTLM_HOST_FLATC_BIN_DIR}"

    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
)
