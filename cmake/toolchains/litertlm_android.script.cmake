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
# LiteRT-LM Android Orchestrator Script
# ==============================================================================

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(NDK_HOST_TAG "linux-x86_64")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(NDK_HOST_TAG "darwin-x86_64")
else()
    message(FATAL_ERROR "[LiteRTLM] Unsupported host OS for Android cross-compilation.")
endif()

string(REPLACE "android-" "" API_LEVEL "${ANDROID_PLATFORM}")

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(RUST_TARGET "aarch64-linux-android")
    set(CARGO_ENV "CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER")
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(RUST_TARGET "x86_64-linux-android")
    set(CARGO_ENV "CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER")
else()
    message(WARNING "LiteRT-LM: Unmapped Rust target for ABI: ${ANDROID_ABI}")
endif()

string(REPLACE "-" "_" RUST_TARGET_UNDERSCORE "${RUST_TARGET}")
set(LITERTLM_CCRS_CXXFLAGS_KEY "CXXFLAGS_${RUST_TARGET_UNDERSCORE}")
set(LITERTLM_CCRS_CXXFLAGS_VAL "--target=${RUST_TARGET}${API_LEVEL} -std=c++20")
set(LITERTLM_CCRS_CFLAGS_KEY "CFLAGS_${RUST_TARGET_UNDERSCORE}")
set(LITERTLM_CCRS_CFLAGS_VAL "--target=${RUST_TARGET}${API_LEVEL}")

set(RUST_LINKER_PATH "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/${RUST_TARGET}${API_LEVEL}-clang")
set(LITERTLM_RUST_LINKER_OVERRIDE "${RUST_LINKER_PATH}"
    CACHE STRING "Override the Rust linker for Android cross-compilation")
set(LITERTLM_RUST_CARGO_ENV_VAR "${CARGO_ENV}"
    CACHE STRING "Environment variable for Rust Cargo linker override")


execute_process(
    COMMAND cargo fetch
    WORKING_DIRECTORY "${LITERTLM_PROJECT_ROOT}"
    COMMAND_ERROR_IS_FATAL ANY
)

if(DEFINED ENV{CARGO_HOME})
    set(CARGO_HOME "$ENV{CARGO_HOME}")
elseif(CMAKE_HOST_WIN32)
    set(CARGO_HOME "$ENV{USERPROFILE}/.cargo")
else()
    set(CARGO_HOME "$ENV{HOME}/.cargo")
endif()

file(GLOB CXX_H_FILES "${CARGO_HOME}/registry/src/*/cxx-*/include/cxx.h")

if(NOT CXX_H_FILES)
    message(FATAL_ERROR "[LiteRTLM] cxx.h not found in Cargo registry after 'cargo fetch'.")
endif()

foreach(CXX_H IN LISTS CXX_H_FILES)
    file(READ "${CXX_H}" CXX_CONTENT)
    string(FIND "${CXX_CONTENT}" "using element_type = T;" IS_PATCHED)

    if(NOT IS_PATCHED EQUAL -1)
        message(STATUS "[LiteRTLM] Already patched: ${CXX_H}")
    else()
        set(SEARCH_STR "using reference = typename std::add_lvalue_reference<T>::type;")
        set(REPLACE_STR "${SEARCH_STR}\n  using element_type = T;")

        string(REPLACE "${SEARCH_STR}" "${REPLACE_STR}" CXX_CONTENT "${CXX_CONTENT}")
        file(WRITE "${CXX_H}" "${CXX_CONTENT}")
        message(STATUS "[LiteRTLM] Successfully patched: ${CXX_H}")
    endif()
endforeach()
