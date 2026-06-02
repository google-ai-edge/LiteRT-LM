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
# Executes ONCE in the root to prepare Phase 2 variables
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

set(RUST_LINKER_PATH "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/${RUST_TARGET}${API_LEVEL}-clang")

list(APPEND LITERTLM_TOOLCHAIN_ARGS "-DLITERTLM_RUST_LINKER_OVERRIDE=${RUST_LINKER_PATH}")
list(APPEND LITERTLM_TOOLCHAIN_ARGS "-DLITERTLM_RUST_CARGO_ENV_VAR=${CARGO_ENV}")