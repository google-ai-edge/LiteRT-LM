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
# LiteRT-LM FetchContent: Windows preamble
# ==============================================================================
#
# Included at the top of fetch_content.cmake when the target is Windows. It
# pre-seeds Windows-specific static/CRT options for the dependencies declared
# there and populates zlib before libpng/minizip call find_package(ZLIB).

# --- ANTLR ---
# Every object in the final link must use the DLL C runtime (/MD). Make sure
# the ANTLR C++ runtime does not select the static one (/MT), and skip its
# shared library, which LiteRT-LM does not link.
set(WITH_STATIC_CRT OFF CACHE BOOL "ANTLR: use the DLL C runtime like the rest of the build" FORCE)
set(ANTLR_BUILD_SHARED OFF CACHE BOOL "ANTLR: only the static runtime is linked" FORCE)

# --- KissFFT ---
# Request a static library: a kissfft DLL would have to be deployed next to
# litert_lm_main.exe.
set(KISSFFT_STATIC ON CACHE BOOL "KissFFT: build a static library" FORCE)

# --- LibPNG ---
# Build the portable C implementation on ARM64 rather than risk the NEON
# sources with cl.exe; PNG decoding speed does not matter here.
string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" _litertlm_win_processor)
if(_litertlm_win_processor MATCHES "^(ARM64|AARCH64)$"
   OR "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" MATCHES "^(ARM64|arm64)$")
  set(PNG_ARM_NEON "off" CACHE STRING "LibPNG: portable C implementation on Windows ARM64" FORCE)
endif()

# --- ZLIB ---
# Linux and macOS resolve find_package(ZLIB) in libpng and minizip against the
# system zlib. Windows has none, so zlib is fetched first (the declaration
# further down in fetch_content.cmake then finds it already populated) and
# find_package(ZLIB) is pointed at that target.
set(LITERTLM_ZLIB_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/zlib CACHE PATH "Path to zlib source directory")
set(LITERTLM_ZLIB_TAG "v1.3.2" CACHE STRING "ZLIB git tag")
FetchContent_Declare(
  zlib_lib
  GIT_REPOSITORY https://github.com/madler/zlib
  GIT_TAG ${LITERTLM_ZLIB_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_ZLIB_SRC_DIR}
)
block()
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(zlib_lib)
endblock()

set(LITERTLM_WINDOWS_ZLIB_TARGET "")
if(TARGET zlibstatic)
  set(LITERTLM_WINDOWS_ZLIB_TARGET zlibstatic)
elseif(TARGET zlib)
  set(LITERTLM_WINDOWS_ZLIB_TARGET zlib)
endif()

if(LITERTLM_WINDOWS_ZLIB_TARGET)
  FetchContent_GetProperties(zlib_lib
    SOURCE_DIR LITERTLM_WINDOWS_ZLIB_SOURCE_DIR
    BINARY_DIR LITERTLM_WINDOWS_ZLIB_BINARY_DIR)
  if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS ${LITERTLM_WINDOWS_ZLIB_TARGET})
  endif()
  message(STATUS "[LiteRTLM] find_package(ZLIB) resolves to the bundled '${LITERTLM_WINDOWS_ZLIB_TARGET}' target")
else()
  message(WARNING "[LiteRTLM] The zlib project defined neither 'zlibstatic' nor 'zlib'; find_package(ZLIB) will fail.")
endif()

# litertlm_windows_zlib_hints()
#
# Sets, in the caller's scope, the result variables FindZLIB looks at before it
# searches. A macro, because fetch_content.cmake calls it inside block().
macro(litertlm_windows_zlib_hints)
  if(LITERTLM_WINDOWS_ZLIB_TARGET)
    set(ZLIB_LIBRARY ${LITERTLM_WINDOWS_ZLIB_TARGET})
    set(ZLIB_LIBRARIES ${LITERTLM_WINDOWS_ZLIB_TARGET})
    set(ZLIB_INCLUDE_DIR "${LITERTLM_WINDOWS_ZLIB_SOURCE_DIR}")
    set(ZLIB_INCLUDE_DIRS
        "${LITERTLM_WINDOWS_ZLIB_BINARY_DIR}"
        "${LITERTLM_WINDOWS_ZLIB_SOURCE_DIR}")
    set(ZLIB_ROOT "${LITERTLM_WINDOWS_ZLIB_SOURCE_DIR}")
  endif()
endmacro()

# minizip and any other project added from this directory.
litertlm_windows_zlib_hints()
