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

# WARNING: Do not use anything from macros.cmake in this file.

function(verify_install target_name config_path)
    ExternalProject_Add_Step(${target_name} step_verify_install
        COMMAND ${CMAKE_COMMAND} -E echo "Verifying installation..."
        COMMAND ${CMAKE_COMMAND} -DFILE_TO_CHECK=${config_path} -P "${LITERTLM_SCRIPTS_DIR}/verify_install.cmake"
        DEPENDEES install
        COMMENT "Ensuring ${config_path} was actually generated."
    )
endfunction()

function(patch_file_content FILE_PATH MATCH_STR REPLACE_STR IS_REGEX)
    if(EXISTS "${FILE_PATH}")
        file(READ "${FILE_PATH}" CONTENT)
        if(IS_REGEX)
            string(REGEX REPLACE "${MATCH_STR}" "${REPLACE_STR}" CONTENT "${CONTENT}")
        else()
            string(REPLACE "${MATCH_STR}" "${REPLACE_STR}" CONTENT "${CONTENT}")
        endif()
        file(WRITE "${FILE_PATH}" "${CONTENT}")
    endif()
endfunction()

# --- cmake_checkpoint_target ---
#
# Synopsis:
#   cmake_checkpoint_target(<name>
#       [TYPE <Interface|Object|Static|Shared|Executable|Custom>]
#       [GLOBAL]
#       [QUIET]
#       [PROPERTIES <prop> <value>...]
#   )
#
# Description:
#   Enforces the existence of a logical target in the current build graph.
#   If the target does not exist, it creates a "Contract Shim" (Imported Target)
#   to satisfy downstream dependencies without defining an implementation.
#
#   This separates the *Declaration* of a dependency from its *Definition*.
#
# Options:
#   TYPE     : The CMake target type. Defaults to INTERFACE.
#   GLOBAL   : Promotes the shim to Global Scope (visible to all directories).
#              Default is Directory Scope (standard CMake visibility).
#   QUIET    : Suppresses status messages.
#   PROPERTIES: List of properties to apply to the shim.
function(cmake_checkpoint_target TARGET_NAME)
    set(options GLOBAL QUIET)
    set(oneValueArgs TYPE)
    set(multiValueArgs PROPERTIES)
    cmake_parse_arguments(CHK "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(TARGET ${TARGET_NAME})
        if(NOT CHK_QUIET)
             message(STATUS "[LiteRTLM] Target '${TARGET_NAME}' satisfied (Existing).")
        endif()
        return()
    endif()

    if(NOT CHK_TYPE)
        set(CHK_TYPE "INTERFACE")
    endif()
    string(TOUPPER "${CHK_TYPE}" CHK_TYPE)

    set(SCOPE_FLAG "")
    if(CHK_GLOBAL)
        set(SCOPE_FLAG "GLOBAL")
    endif()

    if(CHK_TYPE STREQUAL "CUSTOM")
        add_custom_target(${TARGET_NAME})

    elseif(CHK_TYPE STREQUAL "INTERFACE")
        add_library(${TARGET_NAME} INTERFACE IMPORTED ${SCOPE_FLAG})

    elseif(CHK_TYPE MATCHES "^(STATIC|SHARED|MODULE|UNKNOWN)$")
        add_library(${TARGET_NAME} ${CHK_TYPE} IMPORTED ${SCOPE_FLAG})

    elseif(CHK_TYPE STREQUAL "EXECUTABLE")
        add_executable(${TARGET_NAME} IMPORTED ${SCOPE_FLAG})

    else()
        message(FATAL_ERROR "cmake_checkpoint_target: Unsupported TYPE '${CHK_TYPE}'")
    endif()

    if(CHK_PROPERTIES)
        set_target_properties(${TARGET_NAME} PROPERTIES ${CHK_PROPERTIES})
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        CHECKPOINT_TYPE "SHIM"
        CHECKPOINT_ORIGIN "${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE}"
    )

    if(NOT CHK_QUIET)
        message(STATUS "[Checkpoint] Created shim for '${TARGET_NAME}' (${CHK_TYPE} ${SCOPE_FLAG})")
    endif()

endfunction()


# ==============================================================================
# kvp_parse_map
# Purpose: Always extracts Keys (Target Names) and Values (File Paths).
# Usage: kvp_parse_map(ABSL_TARGET_MAP _KEYS _VALS)
# ==============================================================================
function(kvp_parse_map MAP_VAR KEYS_OUT VALS_OUT)
    set(_KVP_KEYS "")
    set(_KVP_VALS "")

    foreach(_entry IN ITEMS ${MAP_VAR})
        # Find the delimiter
        string(FIND "${_entry}" "=" _pos)

        if(_pos EQUAL -1)
            message(FATAL_ERROR "[LiteRTLM] Malformed KVP entry: '${_entry}'. Expected 'Key|Value'")
        endif()

        # 1. Extract Key (Left of |)
        string(SUBSTRING "${_entry}" 0 ${_pos} _key)
        string(STRIP "${_key}" _key)

        # 2. Extract Value (Right of |)
        math(EXPR _val_start "${_pos} + 1")
        string(SUBSTRING "${_entry}" ${_val_start} -1 _val)
        string(STRIP "${_val}" _val)

        # 3. Synchronized storage
        list(APPEND _KVP_KEYS "${_key}")
        list(APPEND _KVP_VALS "${_val}")
    endforeach()

    string(JOIN " " _FLAT_KEYS ${_KVP_KEYS})
    string(JOIN " " _FLAT_VALS ${_KVP_VALS})

    set(${KEYS_OUT} "${_FLAT_KEYS}" PARENT_SCOPE)
    set(${VALS_OUT} "${_FLAT_VALS}" PARENT_SCOPE)
endfunction()



# ==============================================================================
# setup_external_install_structure
# Purpose: Pre-creates the standard GNU directory tree for an external project.
# Rationale: Satisfies CMake's strict path validation for IMPORTED targets
#            before the ExternalProject build phase actually runs.
# ==============================================================================
function(setup_external_install_structure prefix_path)
    if(NOT prefix_path)
        message(FATAL_ERROR "[LiteRTLM] setup_external_install_structure called with empty path.")
    endif()

    set(_subdirs
        "${prefix_path}/include"
        "${prefix_path}/lib"
        "${prefix_path}/bin"
    )

    foreach(_dir IN LISTS _subdirs)
        if(NOT EXISTS "${_dir}")
            message(STATUS "[LiteRTLM] Pre-creating install directory: ${_dir}")
            file(MAKE_DIRECTORY "${_dir}")
        endif()
    endforeach()
endfunction()

function(cmake_to_c_bool CMAKE_VAL OUTPUT_VAR)
    if(${CMAKE_VAL})
        set(${OUTPUT_VAR} 1 PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} 0 PARENT_SCOPE)
    endif()
endfunction()

function(patch_delete_block TARGET_FILE START_PATTERN END_PATTERN)
    if(EXISTS "${TARGET_FILE}")

        file(STRINGS "${TARGET_FILE}" FILE_LINES)
        set(NEW_LINES "")
        set(IN_BLOCK FALSE)

        foreach(LINE IN LISTS FILE_LINES)
            if(LINE MATCHES "${START_PATTERN}")
                set(IN_BLOCK TRUE)
            endif()

            if(NOT IN_BLOCK)
                list(APPEND NEW_LINES "${LINE}")
            endif()

            if(IN_BLOCK AND LINE MATCHES "${END_PATTERN}")
                set(IN_BLOCK FALSE)
            endif()
        endforeach()
        string(REPLACE ";" "\n" NEW_CONTENT "${NEW_LINES}")
        file(WRITE "${TARGET_FILE}" "${NEW_CONTENT}\n")
    endif()
endfunction()