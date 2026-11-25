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
# LiteRT-LM: linking the final executable with the MSVC ABI (cl.exe, clang-cl)
# ==============================================================================
#
# On ELF the executable links every dependency archive by absolute path, taken
# from the target maps at configure time, inside --start-group/--whole-archive.
# That does not carry over to link.exe / lld-link:
#
#   * `/WHOLEARCHIVE` and `/FORCE:MULTIPLE` passed through
#     target_link_libraries() start with '/', so CMake treats them as absolute
#     library paths instead of options.
#   * The archives belong to ExternalProjects and do not exist when Ninja
#     starts; as link items they fail with "missing and no known rule to make
#     it".
#   * The archive names differ per project on Windows (`absl_base.lib` but
#     `libprotobuf.lib`), so a configure-time guess can be wrong.
#
# Instead, the candidate archives are written to a spec file at configure time
# and resolved against the file system by scripts/windows_write_link_rsp.cmake
# right before the link, which produces a linker response file.
#
# LiteRT-LM's own archives are linked with /WHOLEARCHIVE, as on ELF, so that
# objects reached only through static registration are kept. Every other
# archive is resolved on demand and needs no group.

include_guard(GLOBAL)

option(LITERTLM_WINDOWS_EXPORT_LITERT_SYMBOLS
    "Export the LiteRt C API from the executable through LiteRT's windows_exported_symbols.def (needed by accelerator DLLs that resolve LiteRt* from the host process)"
    OFF)

option(LITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER
    "Windows: link prebuilt/windows_<arch>/libGemmaModelConstraintProvider instead of the stub provider"
    OFF)

# _litertlm_windows_append_spec(<spec-var> <WHOLE|LIB> <space or ; separated paths>)
#
# One spec line per archive: `<mode>|<path>[|<alternative path>...]`. The first
# existing path wins when the response file is written.
function(_litertlm_windows_append_spec SPEC_VAR MODE)
    set(_spec "${${SPEC_VAR}}")
    foreach(_group IN LISTS ARGN)
        separate_arguments(_paths NATIVE_COMMAND "${_group}")
        foreach(_path IN LISTS _paths)
            if(NOT "${_path}" STREQUAL "")
                string(APPEND _spec "${MODE}|${_path}\n")
            endif()
        endforeach()
    endforeach()
    set(${SPEC_VAR} "${_spec}" PARENT_SCOPE)
endfunction()

# litertlm_windows_prebuilt_gemma_provider(<out-import-lib> <out-dll>)
#
# Picks prebuilt/windows_arm64 or prebuilt/windows_x86_64 from the architecture
# the compiler targets and fails early, with the reason, if the binaries for
# that architecture are missing or are still Git LFS pointer files.
function(litertlm_windows_prebuilt_gemma_provider OUT_LIB OUT_DLL)
    string(TOUPPER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" _arch)
    if(_arch MATCHES "^(ARM64|AARCH64)$")
        set(_subdir "windows_arm64")
    elseif(_arch MATCHES "^(X64|AMD64|X86_64)$")
        set(_subdir "windows_x86_64")
    else()
        message(FATAL_ERROR
            "[LiteRTLM] No prebuilt GemmaModelConstraintProvider for architecture "
            "'${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}'. Configure with "
            "-DLITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER=OFF to use the stub.")
    endif()

    set(_dir "${LITERTLM_PROJECT_ROOT}/prebuilt/${_subdir}")
    set(_lib "${_dir}/libGemmaModelConstraintProvider.lib")
    set(_dll "${_dir}/libGemmaModelConstraintProvider.dll")
    foreach(_file IN ITEMS "${_lib}" "${_dll}")
        if(NOT EXISTS "${_file}")
            message(FATAL_ERROR
                "[LiteRTLM] LITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER=ON but ${_file} "
                "does not exist. The prebuilt for this architecture is not available in "
                "this checkout; configure with "
                "-DLITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER=OFF to build with the "
                "stub provider (constrained decoding for Gemma tools is then disabled).")
        endif()
        file(SIZE "${_file}" _size)
        if(_size LESS 1024)
            message(FATAL_ERROR
                "[LiteRTLM] ${_file} is ${_size} bytes: it is a Git LFS pointer, not the "
                "binary. Run `git lfs pull`, or configure with "
                "-DLITERTLM_USE_PREBUILT_GEMMA_CONSTRAINT_PROVIDER=OFF.")
        endif()
    endforeach()
    set(${OUT_LIB} "${_lib}" PARENT_SCOPE)
    set(${OUT_DLL} "${_dll}" PARENT_SCOPE)
endfunction()

# litertlm_windows_link_executable(<target>
#     WHOLE <archive lists...>     # linked with /WHOLEARCHIVE
#     LIBS <archive lists...>      # resolved on demand
#     DEPENDS <targets...>)        # targets that produce those archives
#
# Each archive list is a string of space separated paths or a CMake list, as
# produced by kvp_parse_map() and the archive registry.
function(litertlm_windows_link_executable TARGET)
    cmake_parse_arguments(PARSE_ARGV 1 _LINK "" "" "WHOLE;LIBS;DEPENDS")

    set(_spec "")
    _litertlm_windows_append_spec(_spec "WHOLE" ${_LINK_WHOLE})
    _litertlm_windows_append_spec(_spec "LIB" ${_LINK_LIBS})

    set(_spec_file "${CMAKE_BINARY_DIR}/${TARGET}.link_spec.txt")
    set(_rsp_file "${CMAKE_BINARY_DIR}/${TARGET}.link_libs.rsp")
    set(_stamp_file "${CMAKE_BINARY_DIR}/${TARGET}.link_libs.stamp")
    file(GENERATE OUTPUT "${_spec_file}" CONTENT "${_spec}")

    # Runs on every build: it is cheap, and it only rewrites the response file
    # and the stamp when the resolved archives or their timestamps change, so
    # the executable relinks exactly when a dependency was rebuilt.
    add_custom_target(${TARGET}_link_inputs
        COMMAND ${CMAKE_COMMAND}
            "-DSPEC_FILE=${_spec_file}"
            "-DRSP_FILE=${_rsp_file}"
            "-DSTAMP_FILE=${_stamp_file}"
            -P "${LITERTLM_SCRIPTS_DIR}/windows_write_link_rsp.cmake"
        BYPRODUCTS "${_rsp_file}" "${_stamp_file}"
        COMMENT "[LiteRTLM] Resolving link inputs for ${TARGET}"
        VERBATIM
    )
    foreach(_dep IN LISTS _LINK_DEPENDS)
        if(TARGET ${_dep})
            add_dependencies(${TARGET}_link_inputs ${_dep})
        endif()
    endforeach()
    add_dependencies(${TARGET} ${TARGET}_link_inputs)

    set_property(TARGET ${TARGET} APPEND PROPERTY LINK_DEPENDS "${_stamp_file}")
    target_link_options(${TARGET} PRIVATE
        "/FORCE:MULTIPLE"
        "@${_rsp_file}"
    )

    if(LITERTLM_WINDOWS_EXPORT_LITERT_SYMBOLS)
        target_link_options(${TARGET} PRIVATE
            "/DEF:${LITERTLM_LITERT_SRC_DIR}/c/windows_exported_symbols.def")
    endif()

    target_link_libraries(${TARGET} PRIVATE ${LITERTLM_WINDOWS_SYSTEM_LIBS})
endfunction()
