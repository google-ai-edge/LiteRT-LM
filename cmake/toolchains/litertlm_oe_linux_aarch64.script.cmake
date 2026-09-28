# Copyright 2026 The ODML Authors.
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
# LiteRT-LM Qualcomm OE-Linux Orchestrator Script
# ==============================================================================

set(RUST_TARGET "aarch64-unknown-linux-gnu")
string(REPLACE "-" "_" RUST_TARGET_UNDERSCORE "${RUST_TARGET}")
set(CARGO_ENV "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER")

if(NOT RUST_LINKER_PATH)
    if(CMAKE_C_COMPILER)
        set(RUST_LINKER_PATH "${CMAKE_C_COMPILER}")
    elseif(DEFINED ENV{CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER})
        set(RUST_LINKER_PATH "$ENV{CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER}")
    else()
        find_program(RUST_LINKER_PATH NAMES aarch64-unknown-linux-gnu-gcc aarch64-linux-gnu-gcc aarch64-oe-linux-gcc aarch64-qcom-linux-gcc REQUIRED)
    endif()
endif()

get_filename_component(_LINKER_DIR "${RUST_LINKER_PATH}" DIRECTORY)
if(EXISTS "${_LINKER_DIR}")
    set(ENV{PATH} "${_LINKER_DIR}:$ENV{PATH}")
    set(ENV{CC_${RUST_TARGET_UNDERSCORE}} "${RUST_LINKER_PATH}")
    if(CMAKE_CXX_COMPILER)
        set(ENV{CXX_${RUST_TARGET_UNDERSCORE}} "${CMAKE_CXX_COMPILER}")
    endif()
endif()

if(NOT CMAKE_AR)
    find_program(CMAKE_AR NAMES aarch64-unknown-linux-gnu-ar aarch64-linux-gnu-ar aarch64-oe-linux-ar aarch64-qcom-linux-ar ar
        HINTS "${_LINKER_DIR}")
endif()
if(CMAKE_AR)
    set(ENV{AR_${RUST_TARGET_UNDERSCORE}} "${CMAKE_AR}")
endif()

set(LITERTLM_CCRS_CXXFLAGS_KEY "CXXFLAGS_${RUST_TARGET_UNDERSCORE}")
set(LITERTLM_CCRS_CXXFLAGS_VAL "--sysroot=${CMAKE_SYSROOT} -std=c++20")
set(LITERTLM_CCRS_CFLAGS_KEY "CFLAGS_${RUST_TARGET_UNDERSCORE}")
set(LITERTLM_CCRS_CFLAGS_VAL "--sysroot=${CMAKE_SYSROOT}")

string(TOUPPER "${RUST_TARGET_UNDERSCORE}" RUST_TARGET_UNDERSCORE_UPPER)
set(ENV{CARGO_TARGET_${RUST_TARGET_UNDERSCORE_UPPER}_RUSTFLAGS} "-C link-arg=--sysroot=${CMAKE_SYSROOT}")
set(ENV{CARGO_TARGET_${RUST_TARGET_UNDERSCORE_UPPER}_LINKER} "${RUST_LINKER_PATH}")

list(APPEND LITERTLM_ENV_WRAPPER "PATH=$ENV{PATH}")
list(APPEND LITERTLM_ENV_WRAPPER
    "CC_${RUST_TARGET_UNDERSCORE}=${RUST_LINKER_PATH}"
    "CXX_${RUST_TARGET_UNDERSCORE}=${CMAKE_CXX_COMPILER}"
    "CARGO_TARGET_${RUST_TARGET_UNDERSCORE_UPPER}_RUSTFLAGS=-C link-arg=--sysroot=${CMAKE_SYSROOT}"
    "CARGO_TARGET_${RUST_TARGET_UNDERSCORE_UPPER}_LINKER=${RUST_LINKER_PATH}"
)
if(CMAKE_AR)
    list(APPEND LITERTLM_ENV_WRAPPER "AR_${RUST_TARGET_UNDERSCORE}=${CMAKE_AR}")
endif()
if(DEFINED OE_ESDK_ROOT)
    list(APPEND LITERTLM_ENV_WRAPPER "OE_ESDK_ROOT=${OE_ESDK_ROOT}")
elseif(DEFINED ENV{OE_ESDK_ROOT})
    list(APPEND LITERTLM_ENV_WRAPPER "OE_ESDK_ROOT=$ENV{OE_ESDK_ROOT}")
endif()
if(DEFINED CMAKE_SYSROOT)
    list(APPEND LITERTLM_ENV_WRAPPER "SDKTARGETSYSROOT=${CMAKE_SYSROOT}")
endif()
set(LITERTLM_ENV_WRAPPER "${LITERTLM_ENV_WRAPPER}" CACHE INTERNAL "LiteRT-LM: Environment wrapper for external project" FORCE)

set(LITERTLM_RUST_LINKER_OVERRIDE "${RUST_LINKER_PATH}"
    CACHE STRING "Override the Rust linker for OE-Linux cross-compilation")
set(LITERTLM_RUST_CARGO_ENV_VAR "${CARGO_ENV}"
    CACHE STRING "Environment variable for Rust Cargo linker override")
set(Rust_CARGO_TARGET "${RUST_TARGET}"
    CACHE STRING "Rust triple for Corrosion")

find_program(CARGO_EXECUTABLE cargo
    HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin"
    DOC "Path to cargo executable")

if(CARGO_EXECUTABLE)
    execute_process(
        COMMAND "${CARGO_EXECUTABLE}" fetch
        WORKING_DIRECTORY "${LITERTLM_PROJECT_ROOT}"
        RESULT_VARIABLE _FETCH_RES
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _FETCH_RES EQUAL 0)
        message(STATUS "[LiteRTLM] Notice: 'cargo fetch' returned ${_FETCH_RES} (continuing with offline/local crates)")
    endif()
endif()
