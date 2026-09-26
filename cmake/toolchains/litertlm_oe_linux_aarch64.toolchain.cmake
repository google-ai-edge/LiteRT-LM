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
# LiteRT-LM Qualcomm OE-Linux (aarch64) Toolchain Wrapper
# ==============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Locate OE-Linux eSDK root
if(DEFINED ENV{OE_ESDK_ROOT})
    set(_ESDK_ROOT "$ENV{OE_ESDK_ROOT}")
elseif(DEFINED ENV{LINUX_AARCH64_ESDK})
    set(_ESDK_ROOT "$ENV{LINUX_AARCH64_ESDK}")
elseif(DEFINED OE_ESDK_ROOT)
    set(_ESDK_ROOT "${OE_ESDK_ROOT}")
elseif(DEFINED LINUX_AARCH64_ESDK)
    set(_ESDK_ROOT "${LINUX_AARCH64_ESDK}")
elseif(DEFINED ENV{OECORE_NATIVE_SYSROOT} AND EXISTS "$ENV{OECORE_NATIVE_SYSROOT}")
    get_filename_component(_ESDK_ROOT "$ENV{OECORE_NATIVE_SYSROOT}/../.." ABSOLUTE)
elseif(DEFINED ENV{SDKTARGETSYSROOT} AND EXISTS "$ENV{SDKTARGETSYSROOT}")
    get_filename_component(_ESDK_ROOT "$ENV{SDKTARGETSYSROOT}/../.." ABSOLUTE)
elseif(EXISTS "/opt/qcom/esdk")
    set(_ESDK_ROOT "/opt/qcom/esdk")
endif()

if(NOT _ESDK_ROOT OR NOT EXISTS "${_ESDK_ROOT}")
    message(FATAL_ERROR "[LiteRTLM] OE_ESDK_ROOT or LINUX_AARCH64_ESDK must be set to the Qualcomm OE-Linux eSDK root directory.")
endif()

set(OE_ESDK_ROOT "${_ESDK_ROOT}" CACHE PATH "Qualcomm OE-Linux eSDK root" FORCE)
set(ENV{OE_ESDK_ROOT} "${_ESDK_ROOT}")

# Locate toolchain bin directory
if(DEFINED ENV{OE_TOOLCHAIN_DIR})
    set(_TOOLCHAIN_DIR "$ENV{OE_TOOLCHAIN_DIR}")
elseif(DEFINED ENV{QCOM_TOOLCHAIN_DIR})
    set(_TOOLCHAIN_DIR "$ENV{QCOM_TOOLCHAIN_DIR}")
elseif(DEFINED OE_TOOLCHAIN_DIR)
    set(_TOOLCHAIN_DIR "${OE_TOOLCHAIN_DIR}")
elseif(DEFINED QCOM_TOOLCHAIN_DIR)
    set(_TOOLCHAIN_DIR "${QCOM_TOOLCHAIN_DIR}")
elseif(DEFINED TOOLCHAIN_DIR)
    set(_TOOLCHAIN_DIR "${TOOLCHAIN_DIR}")
endif()

if(NOT _TOOLCHAIN_DIR)
    set(_SDK_SYSROOTS "")
    if(DEFINED ENV{OECORE_NATIVE_SYSROOT} AND EXISTS "$ENV{OECORE_NATIVE_SYSROOT}")
        list(APPEND _SDK_SYSROOTS "$ENV{OECORE_NATIVE_SYSROOT}")
    endif()
    file(GLOB _ESDK_SYSROOTS "${_ESDK_ROOT}/sysroots/*sdk-linux")
    list(APPEND _SDK_SYSROOTS ${_ESDK_SYSROOTS})

    set(_CANDIDATE_DIRS "")
    foreach(_SYSROOT IN LISTS _SDK_SYSROOTS)
        file(GLOB _ARCH_DIRS "${_SYSROOT}/usr/bin/*linux")
        list(FILTER _ARCH_DIRS EXCLUDE REGEX "musl")
        list(APPEND _CANDIDATE_DIRS ${_ARCH_DIRS})
        list(APPEND _CANDIDATE_DIRS "${_SYSROOT}/usr/bin")
    endforeach()

    foreach(_DIR IN LISTS _CANDIDATE_DIRS)
        if(IS_DIRECTORY "${_DIR}")
            if(EXISTS "${_DIR}/aarch64-qcom-linux-gcc" OR
               EXISTS "${_DIR}/aarch64-oe-linux-gcc" OR
               EXISTS "${_DIR}/aarch64-linux-gnu-gcc" OR
               EXISTS "${_DIR}/aarch64-unknown-linux-gnu-gcc")
                set(_TOOLCHAIN_DIR "${_DIR}")
                break()
            endif()
        endif()
    endforeach()

    if(NOT _TOOLCHAIN_DIR)
        foreach(_DIR IN LISTS _CANDIDATE_DIRS)
            if(IS_DIRECTORY "${_DIR}")
                set(_TOOLCHAIN_DIR "${_DIR}")
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT _TOOLCHAIN_DIR AND EXISTS "/opt/qcom_rust_toolchain/bin")
    set(_TOOLCHAIN_DIR "/opt/qcom_rust_toolchain/bin")
endif()

if(NOT CMAKE_C_COMPILER)
    find_program(CMAKE_C_COMPILER
        NAMES aarch64-unknown-linux-gnu-gcc aarch64-linux-gnu-gcc aarch64-oe-linux-gcc aarch64-qcom-linux-gcc
        HINTS "${_TOOLCHAIN_DIR}"
        REQUIRED)
endif()

if(NOT CMAKE_CXX_COMPILER)
    find_program(CMAKE_CXX_COMPILER
        NAMES aarch64-unknown-linux-gnu-g++ aarch64-linux-gnu-g++ aarch64-oe-linux-g++ aarch64-qcom-linux-g++
        HINTS "${_TOOLCHAIN_DIR}"
        REQUIRED)
endif()

if(NOT CMAKE_AR)
    find_program(CMAKE_AR
        NAMES aarch64-unknown-linux-gnu-ar aarch64-linux-gnu-ar aarch64-oe-linux-ar aarch64-qcom-linux-ar ar
        HINTS "${_TOOLCHAIN_DIR}"
        REQUIRED)
endif()

if(NOT CMAKE_RANLIB)
    find_program(CMAKE_RANLIB
        NAMES aarch64-unknown-linux-gnu-ranlib aarch64-linux-gnu-ranlib aarch64-oe-linux-ranlib aarch64-qcom-linux-ranlib ranlib
        HINTS "${_TOOLCHAIN_DIR}"
        REQUIRED)
endif()

# Note: Top-level CMakeLists.txt checks LITERTLM_CC_NDK / LITERTLM_CXX_NDK to identify
# target cross-compilers for the ExternalProject litert_lm configuration step.
set(LITERTLM_CC_NDK "${CMAKE_C_COMPILER}" CACHE FILEPATH "Target C cross-compiler")
set(LITERTLM_CXX_NDK "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Target CXX cross-compiler")

# Ensure host compilers for prebuild (protoc/flatc) default to host gcc/g++ if not specified.
if(NOT DEFINED LITERTLM_CC)
    set(LITERTLM_CC "gcc" CACHE STRING "Host C compiler for prebuild phase")
endif()
if(NOT DEFINED LITERTLM_CXX)
    set(LITERTLM_CXX "g++" CACHE STRING "Host CXX compiler for prebuild phase")
endif()

get_filename_component(_COMPILER_DIR "${CMAKE_C_COMPILER}" DIRECTORY)

# Dynamically locate sysroot architecture subdirectory
if(NOT CMAKE_SYSROOT)
    if(DEFINED ENV{SDKTARGETSYSROOT} AND EXISTS "$ENV{SDKTARGETSYSROOT}")
        set(CMAKE_SYSROOT "$ENV{SDKTARGETSYSROOT}")
    elseif(EXISTS "${_ESDK_ROOT}/sysroots/armv8-2a-qcom-linux")
        set(CMAKE_SYSROOT "${_ESDK_ROOT}/sysroots/armv8-2a-qcom-linux")
    else()
        file(GLOB _MATCHED_SYSROOTS "${_ESDK_ROOT}/sysroots/*-linux*")
        list(FILTER _MATCHED_SYSROOTS EXCLUDE REGEX "x86_64|sdk")
        if(_MATCHED_SYSROOTS)
            list(GET _MATCHED_SYSROOTS 0 CMAKE_SYSROOT)
        endif()
    endif()
endif()

if(NOT CMAKE_SYSROOT OR NOT EXISTS "${CMAKE_SYSROOT}")
    message(FATAL_ERROR "[LiteRTLM] Unable to locate target sysroot under ${_ESDK_ROOT}/sysroots.")
endif()

set(CMAKE_SYSROOT "${CMAKE_SYSROOT}" CACHE PATH "Target sysroot" FORCE)
set(ENV{SDKTARGETSYSROOT} "${CMAKE_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(RUST_LINKER_PATH "${CMAKE_C_COMPILER}")
set(CARGO_ENV "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER")

set(LITERTLM_RUST_LINKER_OVERRIDE "${RUST_LINKER_PATH}"
    CACHE STRING "Override the Rust linker for OE-Linux cross-compilation")
set(LITERTLM_RUST_CARGO_ENV_VAR "${CARGO_ENV}"
    CACHE STRING "Environment variable for Rust Cargo linker override")
set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu"
    CACHE STRING "Rust target triple for Corrosion")

# Optional QAIRT/QNN headers directory
set(_QAIRT_HEADERS_DIR "")
if(DEFINED ENV{QAIRT_HEADERS_DIR})
    set(_QAIRT_HEADERS_DIR "$ENV{QAIRT_HEADERS_DIR}")
elseif(DEFINED QAIRT_HEADERS_DIR)
    set(_QAIRT_HEADERS_DIR "${QAIRT_HEADERS_DIR}")
elseif(DEFINED ENV{QNN_SDK_ROOT})
    set(_QAIRT_HEADERS_DIR "$ENV{QNN_SDK_ROOT}/include")
elseif(DEFINED QNN_SDK_ROOT)
    set(_QAIRT_HEADERS_DIR "${QNN_SDK_ROOT}/include")
endif()

set(LITERTLM_TOOLCHAIN_ARGS
    "-DCMAKE_SYSTEM_NAME=Linux"
    "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_AR=${CMAKE_AR}"
    "-DCMAKE_RANLIB=${CMAKE_RANLIB}"
    "-DCMAKE_SYSROOT=${CMAKE_SYSROOT}"
    "-DCMAKE_FIND_ROOT_PATH=${CMAKE_SYSROOT}"
    "-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER"
    "-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY"
    "-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY"
    "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY"
    "-DTOOLCHAIN_DIR=${_COMPILER_DIR}/"
    "-DOE_ESDK_ROOT=${_ESDK_ROOT}"
    "-DRust_CARGO_TARGET=aarch64-unknown-linux-gnu"
    "-DLITERTLM_RUST_LINKER_OVERRIDE=${RUST_LINKER_PATH}"
    "-DLITERTLM_RUST_CARGO_ENV_VAR=${CARGO_ENV}"
    "-DLITERTLM_PROTOBUF_CXX_FLAGS_RELEASE=-O2 -DNDEBUG"
    "-DENABLE_UV=OFF"
)

if(_QAIRT_HEADERS_DIR AND EXISTS "${_QAIRT_HEADERS_DIR}")
    list(APPEND LITERTLM_TOOLCHAIN_ARGS "-DQAIRT_HEADERS_DIR=${_QAIRT_HEADERS_DIR}")
endif()

set(LITERTLM_TOOLCHAIN_ARGS "${LITERTLM_TOOLCHAIN_ARGS}"
    CACHE INTERNAL "String used to define parameterized CMAKE_ARGS")
