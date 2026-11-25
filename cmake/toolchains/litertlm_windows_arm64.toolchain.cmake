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
# LiteRT-LM Windows ARM64 Toolchain
# ==============================================================================
#
# Used by the `windows-arm64` preset. The compiler itself is NOT selected here:
# the build is started from a Visual Studio developer shell that already
# targets ARM64 (`vcvarsall.bat arm64` on a Snapdragon device, or
# `vcvarsall.bat amd64_arm64` on an x64 machine), so `cl`, `link`, `rc` and `mt`
# on PATH are the ARM64 tools.
#
# What this file does is tell CMake that the *target* is ARM64. Without it
# CMAKE_SYSTEM_PROCESSOR describes the build machine (AMD64 on an x64 host, and
# also under an emulated x64 CMake on an ARM64 device), and dependencies that
# pick their sources from it -- XNNPACK, cpuinfo, ruy, libpng -- would compile
# x86 kernels with an ARM64 compiler.
#
# Setting CMAKE_SYSTEM_NAME puts CMake in cross-compiling mode on every host,
# which is what routes the build through the host protoc / flatc
# (LITERTLM_HOST_PROTOC, LITERTLM_HOST_FLATC) instead of the target ones.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)
if(NOT CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0)
endif()

# Only read by clang-cl; cl.exe is already an ARM64 compiler and ignores it.
set(CMAKE_C_COMPILER_TARGET arm64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET arm64-pc-windows-msvc)

# Host programs (python, java, cargo, git) come from the build machine; nothing
# else is looked up outside the build tree.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
