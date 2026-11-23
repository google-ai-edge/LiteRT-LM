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
# LiteRT-LM platform helpers
#
# Shared by the inner LiteRT-LM project, by every package shim that is injected
# into a third-party build, and by the cmake -P helper scripts. Everything in
# here is a no-op unless the build targets the MSVC ABI (cl.exe or clang-cl), so
# including this file never changes a Linux, macOS or Android build.
#
# WARNING: Do not use anything from macros.cmake in this file.
# ==============================================================================

# --- Windows system libraries -------------------------------------------------
# Needed by the Rust standard library (ws2_32, userenv, ntdll, bcrypt, ...) and
# by Abseil (dbghelp, bcrypt, advapi32) once their static archives are linked
# by path instead of through an imported target.
set(LITERTLM_WINDOWS_SYSTEM_LIBS
    advapi32.lib
    bcrypt.lib
    dbghelp.lib
    iphlpapi.lib
    ntdll.lib
    ole32.lib
    psapi.lib
    shell32.lib
    shlwapi.lib
    synchronization.lib
    userenv.lib
    ws2_32.lib
)

# litertlm_is_msvc_frontend(<out-var>)
#
# TRUE when the active compiler uses the MSVC command line: cl.exe, and also
# clang-cl, which CMake reports as CMAKE_CXX_COMPILER_ID "Clang" with MSVC set.
# Callers must test this BEFORE any `CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"`
# check, otherwise clang-cl is handed GNU ld options such as -Wl,--start-group.
function(litertlm_is_msvc_frontend OUT_VAR)
    set(_is_msvc FALSE)
    if(MSVC
       OR "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC"
       OR "${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
        set(_is_msvc TRUE)
    elseif(CMAKE_SCRIPT_MODE_FILE AND CMAKE_HOST_WIN32 AND NOT MINGW)
        # cmake -P scripts have no compiler; assume the MSVC ABI on Windows.
        set(_is_msvc TRUE)
    endif()
    set(${OUT_VAR} ${_is_msvc} PARENT_SCOPE)
endfunction()

# litertlm_normalize_static_lib_path(<out-var> <path>)
#
# The target maps under cmake/packages/*/ are generated on Linux and spell
# every archive as `lib<name>.a` (or `lib<name>${CMAKE_STATIC_LIBRARY_SUFFIX}`).
# The MSVC ABI has no `lib` prefix and uses `.lib`, except for projects that
# add the prefix themselves. This maps a Linux-style path to the name used on
# Windows:
#
#   1. If `<dir>/<name>.lib` or `<dir>/lib<name>.lib` exists, that file wins.
#   2. Otherwise, if the directory already holds other `.lib` files the archive
#      is not produced on this platform and "" is returned so the caller can
#      drop it (disable with LITERTLM_TARGET_MAP_PRUNE_MISSING=OFF).
#   3. Otherwise nothing has been built yet and the name is guessed.
#
# Paths are returned unchanged for every other toolchain.
function(litertlm_normalize_static_lib_path OUT_VAR LIB_PATH)
    litertlm_is_msvc_frontend(_is_msvc)
    if(NOT _is_msvc OR NOT "${LIB_PATH}" MATCHES "\\.(a|lib)$")
        set(${OUT_VAR} "${LIB_PATH}" PARENT_SCOPE)
        return()
    endif()

    get_filename_component(_dir "${LIB_PATH}" DIRECTORY)
    get_filename_component(_name "${LIB_PATH}" NAME)
    string(REGEX REPLACE "\\.(a|lib)$" "" _stem "${_name}")
    if(_stem MATCHES "^lib(.+)$")
        set(_bare "${CMAKE_MATCH_1}")
        set(_prefixed "${_stem}")
    else()
        set(_bare "${_stem}")
        set(_prefixed "lib${_stem}")
    endif()

    if(EXISTS "${_dir}/${_bare}.lib")
        set(${OUT_VAR} "${_dir}/${_bare}.lib" PARENT_SCOPE)
        return()
    endif()
    if(EXISTS "${_dir}/${_prefixed}.lib")
        set(${OUT_VAR} "${_dir}/${_prefixed}.lib" PARENT_SCOPE)
        return()
    endif()

    if(NOT DEFINED LITERTLM_TARGET_MAP_PRUNE_MISSING OR LITERTLM_TARGET_MAP_PRUNE_MISSING)
        file(GLOB _siblings LIST_DIRECTORIES false "${_dir}/*.lib")
        if(_siblings)
            message(STATUS
                "[LiteRTLM] ${_name} is not built on this platform; dropping it from the link line.")
            set(${OUT_VAR} "" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Nothing is built yet, so this is a guess: Protobuf is expected to keep
    # the `lib` prefix on MSVC, everything else to follow the CMake default.
    # The guess only has to be good enough for arguments passed to dependency
    # builds; the final link re-resolves every archive against the file system
    # (scripts/windows_write_link_rsp.cmake).
    if(_prefixed MATCHES "^lib(protobuf|protobuf-lite|protoc|upb)$")
        set(${OUT_VAR} "${_dir}/${_prefixed}.lib" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "${_dir}/${_bare}.lib" PARENT_SCOPE)
    endif()
endfunction()

# litertlm_set_msvc_link_flags()
#
# Sets the _LITERTLM_LINK_* variables the package shims splice into
# CMAKE_CXX_STANDARD_LIBRARIES. link.exe and lld-link resolve archives in any
# order, so there is no group; duplicate definitions across the aggregated
# archives are tolerated the same way --allow-multiple-definition does on ELF.
macro(litertlm_set_msvc_link_flags)
    set(_LITERTLM_LINK_MULTIDEF "/FORCE:MULTIPLE")
    set(_LITERTLM_LINK_GROUP_START "")
    set(_LITERTLM_LINK_GROUP_END "")
    set(_LITERTLM_LINK_WHOLE_START "")
    set(_LITERTLM_LINK_WHOLE_END "")
    string(REPLACE ";" " " _LITERTLM_SYSLIBS "${LITERTLM_WINDOWS_SYSTEM_LIBS}")
endmacro()

# litertlm_windows_rust_target(<out-var>)
#
# Rust target triple matching the C/C++ compiler in use. An explicit
# LITERTLM_RUST_TARGET wins; otherwise the triple follows the architecture the
# compiler targets, which is what distinguishes an ARM64 from an x64 developer
# shell (CMAKE_SYSTEM_PROCESSOR only describes the build machine there).
function(litertlm_windows_rust_target OUT_VAR)
    set(_triple "")
    if(LITERTLM_RUST_TARGET)
        set(_triple "${LITERTLM_RUST_TARGET}")
    else()
        set(_arch "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
        if(NOT _arch)
            set(_arch "${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
        endif()
        if(NOT _arch)
            set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        string(TOUPPER "${_arch}" _arch)
        if(_arch MATCHES "^(ARM64|AARCH64)$")
            set(_triple "aarch64-pc-windows-msvc")
        elseif(_arch MATCHES "^(X64|AMD64|X86_64)$")
            set(_triple "x86_64-pc-windows-msvc")
        endif()
    endif()
    set(${OUT_VAR} "${_triple}" PARENT_SCOPE)
endfunction()
