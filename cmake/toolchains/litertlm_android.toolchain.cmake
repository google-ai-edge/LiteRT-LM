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
# LiteRT-LM Android Toolchain Wrapper
# ==============================================================================

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    ANDROID_NDK_ROOT
    ANDROID_ABI
    ANDROID_PLATFORM
    ANDROID_STL
    CMAKE_SYSTEM_NAME
    CMAKE_SYSTEM_VERSION
    CMAKE_ANDROID_ARCH_ABI
    CMAKE_ANDROID_NDK
    CMAKE_ANDROID_STL_TYPE
)

if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "Target Android Architecture")
    if(NOT CMAKE_C_COMPILER_LOADED)
        message(STATUS "[LiteRTLM] No ANDROID_ABI provided. Defaulting to ${ANDROID_ABI}.")
    endif()
endif()

if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "android-28" CACHE STRING "Target Android API Level")
    if(NOT CMAKE_C_COMPILER_LOADED)
        message(STATUS "[LiteRTLM] No ANDROID_PLATFORM provided. Defaulting to ${ANDROID_PLATFORM}.")
    endif()
endif()

if(NOT DEFINED ANDROID_STL)
    set(ANDROID_STL "c++_shared" CACHE STRING "Target Android STL")
endif()

if(NOT DEFINED CMAKE_SYSTEM_NAME)
    set(CMAKE_SYSTEM_NAME "Android" CACHE STRING "Target OS")
endif()

if(NOT DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(CMAKE_ANDROID_ARCH_ABI "${ANDROID_ABI}" CACHE STRING "Target Android Architecture (Native)")
endif()

if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    string(REPLACE "android-" "" _API_LEVEL "${ANDROID_PLATFORM}")
    set(CMAKE_SYSTEM_VERSION "${_API_LEVEL}" CACHE STRING "Target Android API Level (Native)")
endif()

if(NOT DEFINED CMAKE_ANDROID_STL_TYPE)
    set(CMAKE_ANDROID_STL_TYPE "${ANDROID_STL}" CACHE STRING "Target Android STL (Native)")
endif()

if(NOT DEFINED CMAKE_ANDROID_NDK)
    set(CMAKE_ANDROID_NDK "${_LITERTLM_NDK_ROOT}" CACHE PATH "Path to the Android NDK (Native)")
endif()

set(_REAL_NDK_TOOLCHAIN "${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake")

if(NOT EXISTS "${_REAL_NDK_TOOLCHAIN}")
    message(FATAL_ERROR "[LiteRTLM] Could not find the Android NDK toolchain at: ${_REAL_NDK_TOOLCHAIN}")
endif()

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

set(RUST_LINKER_PATH "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/${RUST_TARGET}${API_LEVEL}-clang")
set(LITERTLM_RUST_LINKER_OVERRIDE "${RUST_LINKER_PATH}"
    CACHE STRING "Override the Rust linker for Android cross-compilation")
set(LITERTLM_RUST_CARGO_ENV_VAR "${CARGO_ENV}"
    CACHE STRING "Environment variable for Rust Cargo linker override")



set(LITERTLM_TOOLCHAIN_ARGS 
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_STL=${ANDROID_STL}"
    "-DANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}"
    "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
    "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}"
    "-DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}"
    "-DLITERTLM_RUST_LINKER_OVERRIDE=${RUST_LINKER_PATH}"
    "-DLITERTLM_RUST_CARGO_ENV_VAR=${CARGO_ENV}"
    "-DLITERTLM_PROTOBUF_CXX_FLAGS_RELEASE=-O2 -DNDEBUG"
    CACHE INTERNAL "String used to define parameterized CMAKE_ARGS"
)

include("${_REAL_NDK_TOOLCHAIN}")

if(CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()
if(CMAKE_CXX_FLAGS_MINSIZEREL)
    string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif()
if(CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if(CMAKE_C_FLAGS_RELEASE)
    string(REPLACE "-O3" "-O2" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()
if(CMAKE_C_FLAGS_MINSIZEREL)
    string(REPLACE "-O3" "-O2" CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif()
if(CMAKE_C_FLAGS_RELWITHDEBINFO)
    string(REPLACE "-O3" "-O2" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()