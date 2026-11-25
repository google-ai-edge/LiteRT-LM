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
# LiteRT-LM Orchestrator: Windows host support
# ==============================================================================
#
# Included by the root CMakeLists.txt only when the build machine runs Windows.
# It fills in the variables the orchestrator splices into its two phases:
#
#   LITERTLM_PREBUILD_PLATFORM_ARGS   extra configure arguments, host phase
#   LITERTLM_TARGET_PLATFORM_ARGS     extra configure arguments, target phase
#   LITERTLM_PREBUILD_CXX_FLAGS_ARGS  compiler flags, host phase
#   LITERTLM_TARGET_CXX_FLAGS_ARGS    compiler flags, target phase
#   LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR / LITERTLM_PREBUILD_TARGETS
#
# The build is expected to run from a Visual Studio developer shell for the
# TARGET architecture, with the Ninja generator:
#
#   windows-x64    vcvarsall.bat amd64
#   windows-arm64  vcvarsall.bat arm64        (on a Windows ARM64 device)
#                  vcvarsall.bat amd64_arm64  (cross compiling on an x64 host)

# --- Generator ----------------------------------------------------------------
# Every nested project forwards a single CMAKE_BUILD_TYPE and looks its outputs
# up without a per-configuration directory, so multi-config generators (Visual
# Studio, the default on Windows) cannot work.
get_property(_litertlm_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_litertlm_multi_config)
    message(FATAL_ERROR
        "[LiteRTLM] The '${CMAKE_GENERATOR}' generator is not supported on Windows. "
        "Use Ninja from a Visual Studio developer shell, e.g.\n"
        "  cmake --preset windows-arm64   (or windows-x64)")
endif()

# --- Build type and C runtime -------------------------------------------------
# Rust's MSVC targets link the release DLL runtime (/MD) unless told
# otherwise, and that is what the Rust static libraries (litert_lm_deps,
# tokenizers_c) are built with here. Every C++ object in the final link has to
# agree, so the same runtime is forced into both phases and into each external
# project.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING
        "MSVC runtime library; must stay the release DLL runtime to link Rust" FORCE)
endif()
if(NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDLL")
    message(WARNING
        "[LiteRTLM] CMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}: the Rust "
        "components link the release DLL runtime (MultiThreadedDLL); mixing runtimes "
        "usually fails at link time.")
endif()

set(_litertlm_win_generator_args "-G" "${CMAKE_GENERATOR}")
if(CMAKE_MAKE_PROGRAM)
    list(APPEND _litertlm_win_generator_args "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
endif()

set(_litertlm_win_msvc_args
    ${_litertlm_win_generator_args}
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
    "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
)

# --- Compiler flags -----------------------------------------------------------
# MSVC spelling of the `-w` the other platforms pass. CMAKE_CXX_FLAGS replaces
# CMake's defaults, so /EHsc and the WIN32 defines are restated. The remaining
# switches mirror `build:windows` in .bazelrc.
set(LITERTLM_MSVC_EXTRA_FLAGS "" CACHE STRING
    "Extra cl.exe / clang-cl flags appended to every LiteRT-LM compile on Windows")
set(_litertlm_msvc_defines
    "/DNOMINMAX /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /D_ENABLE_EXTENDED_ALIGNED_STORAGE")
set(_litertlm_msvc_cxx_flags
    "/DWIN32 /D_WINDOWS /EHsc /W0 /utf-8 /bigobj /Zc:__cplusplus ${_litertlm_msvc_defines}")
set(_litertlm_msvc_c_flags
    "/DWIN32 /D_WINDOWS /W0 /utf-8 ${_litertlm_msvc_defines}")
if(LITERTLM_MSVC_EXTRA_FLAGS)
    string(APPEND _litertlm_msvc_cxx_flags " ${LITERTLM_MSVC_EXTRA_FLAGS}")
    string(APPEND _litertlm_msvc_c_flags " ${LITERTLM_MSVC_EXTRA_FLAGS}")
endif()
set(_litertlm_msvc_flag_args
    "-DCMAKE_CXX_FLAGS=${_litertlm_msvc_cxx_flags}"
    "-DCMAKE_C_FLAGS=${_litertlm_msvc_c_flags}"
)

# The host phase always builds with the Windows host compiler.
list(APPEND LITERTLM_PREBUILD_PLATFORM_ARGS ${_litertlm_win_msvc_args})
set(LITERTLM_PREBUILD_CXX_FLAGS_ARGS ${_litertlm_msvc_flag_args})

# --- Short build paths --------------------------------------------------------
# MSVC tools are limited to MAX_PATH (260 characters) and the dependency trees
# nest deeply, XNNPACK in particular. Keep the external projects close to the
# root of the build directory: <build>/ext for the host phase, <build>/xt for a
# cross-compiled target phase.
set(LITERTLM_PREBUILD_EXTERNAL_PROJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ext")
if(LITERTLM_CROSSCOMPILING)
    list(APPEND LITERTLM_TARGET_PLATFORM_ARGS
        "-DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}/xt")
endif()

# --- User-provided paths ------------------------------------------------------
# These are forwarded through several generated command lines and .cmake files;
# a backslash there is read as an escape sequence (C:\t... becomes a tab).
foreach(_litertlm_path_var LITERTLM_HOST_PROTOC LITERTLM_HOST_FLATC QAIRT_HEADERS_DIR)
    if(DEFINED ${_litertlm_path_var} AND NOT "${${_litertlm_path_var}}" STREQUAL "")
        file(TO_CMAKE_PATH "${${_litertlm_path_var}}" _litertlm_path_value)
        set(${_litertlm_path_var} "${_litertlm_path_value}" CACHE PATH "" FORCE)
    endif()
endforeach()

if(NOT WIN32)
    # Windows host, non-Windows target (e.g. the Android NDK). Only the
    # generator has to be pinned for the target phase.
    list(APPEND LITERTLM_TARGET_PLATFORM_ARGS ${_litertlm_win_generator_args})
    return()
endif()

# ==============================================================================
# Windows target
# ==============================================================================
list(APPEND LITERTLM_TARGET_PLATFORM_ARGS ${_litertlm_win_msvc_args})
set(LITERTLM_TARGET_CXX_FLAGS_ARGS ${_litertlm_msvc_flag_args})

# --- Target architecture ------------------------------------------------------
function(_litertlm_windows_normalize_arch OUT_VAR ARCH)
    string(TOUPPER "${ARCH}" _arch)
    if(_arch MATCHES "^(ARM64|AARCH64)$")
        set(${OUT_VAR} "ARM64" PARENT_SCOPE)
    elseif(_arch MATCHES "^(X64|AMD64|X86_64)$")
        set(${OUT_VAR} "x64" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "${ARCH}" PARENT_SCOPE)
    endif()
endfunction()

set(LITERTLM_WINDOWS_TARGET_ARCH "" CACHE STRING
    "Windows target architecture: ARM64 or x64. Empty detects it from the developer shell.")

# vcvarsall.bat exports the architecture the shell targets.
_litertlm_windows_normalize_arch(_litertlm_shell_arch "$ENV{VSCMD_ARG_TGT_ARCH}")

_litertlm_windows_normalize_arch(_litertlm_target_arch "${LITERTLM_WINDOWS_TARGET_ARCH}")
if(NOT _litertlm_target_arch)
    if(CMAKE_TOOLCHAIN_FILE AND CMAKE_SYSTEM_PROCESSOR)
        _litertlm_windows_normalize_arch(_litertlm_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
    elseif(_litertlm_shell_arch)
        set(_litertlm_target_arch "${_litertlm_shell_arch}")
    else()
        _litertlm_windows_normalize_arch(_litertlm_target_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
endif()

if(NOT _litertlm_target_arch MATCHES "^(ARM64|x64)$")
    message(FATAL_ERROR
        "[LiteRTLM] Unsupported Windows target architecture '${_litertlm_target_arch}'. "
        "Set -DLITERTLM_WINDOWS_TARGET_ARCH=ARM64 or x64.")
endif()

if(NOT _litertlm_shell_arch)
    message(WARNING
        "[LiteRTLM] VSCMD_ARG_TGT_ARCH is not set: this does not look like a Visual Studio "
        "developer shell. Run vcvarsall.bat for ${_litertlm_target_arch} first so that cl, "
        "link, rc and mt for the target are on PATH.")
elseif(NOT _litertlm_shell_arch STREQUAL _litertlm_target_arch)
    message(FATAL_ERROR
        "[LiteRTLM] This developer shell targets ${_litertlm_shell_arch} but the build targets "
        "${_litertlm_target_arch}. Open a shell for the target first:\n"
        "  x64:   vcvarsall.bat amd64\n"
        "  ARM64: vcvarsall.bat arm64 (Windows ARM64 device) or amd64_arm64 (x64 host)")
endif()
message(STATUS "[LiteRTLM] Windows target architecture: ${_litertlm_target_arch}")

# --- Rust ---------------------------------------------------------------------
set(LITERTLM_RUST_TARGET "" CACHE STRING
    "Rust target triple (e.g. aarch64-pc-windows-msvc). Empty follows the target architecture.")
if(NOT LITERTLM_RUST_TARGET)
    if(_litertlm_target_arch STREQUAL "ARM64")
        set(_litertlm_rust_target "aarch64-pc-windows-msvc")
    else()
        set(_litertlm_rust_target "x86_64-pc-windows-msvc")
    endif()
else()
    set(_litertlm_rust_target "${LITERTLM_RUST_TARGET}")
endif()

list(APPEND LITERTLM_TARGET_PLATFORM_ARGS
    "-DLITERTLM_WINDOWS_TARGET_ARCH=${_litertlm_target_arch}"
    "-DLITERTLM_RUST_TARGET=${_litertlm_rust_target}"
)

# --- Options of the target phase ------------------------------------------------
# Forwarded only when set here, so their defaults stay with the inner project.
foreach(_litertlm_forwarded_option
        LITERTLM_LITERT_WINDOWS_BUILD_ALL
        LITERTLM_WINDOWS_EXPORT_LITERT_SYMBOLS
        LITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER)
    if(DEFINED ${_litertlm_forwarded_option})
        list(APPEND LITERTLM_TARGET_PLATFORM_ARGS
            "-D${_litertlm_forwarded_option}=${${_litertlm_forwarded_option}}")
    endif()
endforeach()

# --- ARM64 --------------------------------------------------------------------
# True on Snapdragon hardware, also when this CMake is an emulated x64 binary.
set(_litertlm_host_is_arm64 FALSE)
if("$ENV{PROCESSOR_ARCHITECTURE}" STREQUAL "ARM64"
   OR "$ENV{PROCESSOR_ARCHITEW6432}" STREQUAL "ARM64")
    set(_litertlm_host_is_arm64 TRUE)
endif()

if(_litertlm_target_arch STREQUAL "ARM64")
    if(NOT CMAKE_TOOLCHAIN_FILE)
        message(FATAL_ERROR
            "[LiteRTLM] A Windows ARM64 build needs the ARM64 toolchain file so that every "
            "dependency sees CMAKE_SYSTEM_PROCESSOR=ARM64. Use\n"
            "  cmake --preset windows-arm64\n"
            "or add -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchains/litertlm_windows_arm64.toolchain.cmake")
    endif()

    if(_litertlm_host_is_arm64)
        set(_litertlm_cross_arm64_default OFF)
    else()
        set(_litertlm_cross_arm64_default ON)
    endif()
    option(LITERTLM_WINDOWS_CROSS_ARM64
        "Cross compile for Windows ARM64 on a host that cannot run ARM64 binaries (requires LITERTLM_HOST_PROTOC and LITERTLM_HOST_FLATC)"
        ${_litertlm_cross_arm64_default})

    if(LITERTLM_WINDOWS_CROSS_ARM64)
        # The ARM64 developer shell cannot produce tools that run on this
        # machine, so protoc and flatc must come from an x64 build.
        foreach(_litertlm_tool PROTOC FLATC)
            if(NOT LITERTLM_HOST_${_litertlm_tool} OR NOT EXISTS "${LITERTLM_HOST_${_litertlm_tool}}")
                message(FATAL_ERROR
                    "[LiteRTLM] Cross compiling for Windows ARM64 on an x64 host needs host "
                    "tools: LITERTLM_HOST_${_litertlm_tool}='${LITERTLM_HOST_${_litertlm_tool}}' "
                    "does not exist.\nBuild them once from an x64 developer shell "
                    "(vcvarsall.bat amd64):\n"
                    "  cmake --preset windows-x64\n"
                    "  cmake --build cmake/build/windows-x64 --target prebuild\n"
                    "then configure from the ARM64 shell (vcvarsall.bat amd64_arm64) with\n"
                    "  -DLITERTLM_HOST_PROTOC=<repo>/cmake/build/windows-x64/ext/protobuf/install/bin/protoc.exe\n"
                    "  -DLITERTLM_HOST_FLATC=<repo>/cmake/build/windows-x64/ext/flatbuffers/install/bin/flatc.exe")
            endif()
        endforeach()
        # Nothing built by the host phase would be runnable; skip its packages.
        set(LITERTLM_PREBUILD_TARGETS "")
        message(STATUS "[LiteRTLM] Cross compiling for Windows ARM64; host tools:")
        message(STATUS "[LiteRTLM]   protoc: ${LITERTLM_HOST_PROTOC}")
        message(STATUS "[LiteRTLM]   flatc:  ${LITERTLM_HOST_FLATC}")
    endif()
elseif(_litertlm_host_is_arm64)
    message(WARNING
        "[LiteRTLM] Building x64 binaries on a Windows ARM64 host is not covered by the "
        "presets: dependencies that read CMAKE_SYSTEM_PROCESSOR may select ARM64 sources.")
endif()
