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

# Writes the linker response file for a Windows (MSVC ABI) executable.
#
# Expected inputs via -D:
#   SPEC_FILE   One archive per line: `<WHOLE|LIB>|<path>[|<alternative>...]`.
#               Paths use the Linux spelling from the target maps
#               (`lib<name>.a` / `lib<name>.lib`).
#   RSP_FILE    Response file to write for link.exe / lld-link.
#   STAMP_FILE  Rewritten whenever an archive is added, removed or rebuilt, so
#               the executable can depend on it.
#
# For every line the first existing file among `<dir>/<name>.lib`,
# `<dir>/lib<name>.lib` and the alternatives is used. Archives that were not
# built on this platform are reported and skipped; if one of them is actually
# needed the link fails with the unresolved symbols that name it.

foreach(_required SPEC_FILE RSP_FILE STAMP_FILE)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "[LiteRTLM] windows_write_link_rsp.cmake: ${_required} is not set")
    endif()
endforeach()

if(NOT EXISTS "${SPEC_FILE}")
    message(FATAL_ERROR "[LiteRTLM] Link spec not found: ${SPEC_FILE}")
endif()

# Candidate file names for one spec path, most likely first.
function(_litertlm_lib_candidates OUT_VAR LIB_PATH)
    get_filename_component(_dir "${LIB_PATH}" DIRECTORY)
    get_filename_component(_name "${LIB_PATH}" NAME)
    string(REGEX REPLACE "\\.(a|lib)$" "" _stem "${_name}")
    if(_stem MATCHES "^lib(.+)$")
        set(_bare "${CMAKE_MATCH_1}")
    else()
        set(_bare "${_stem}")
    endif()
    set(${OUT_VAR}
        "${_dir}/${_bare}.lib"
        "${_dir}/lib${_bare}.lib"
        "${LIB_PATH}"
        PARENT_SCOPE)
endfunction()

file(STRINGS "${SPEC_FILE}" _spec_lines)

set(_rsp_content "")
set(_stamp_content "")
set(_seen "")
set(_missing "")
set(_whole_count 0)
set(_lib_count 0)

foreach(_line IN LISTS _spec_lines)
    if("${_line}" STREQUAL "")
        continue()
    endif()
    string(REPLACE "|" ";" _fields "${_line}")
    list(POP_FRONT _fields _mode)

    set(_resolved "")
    foreach(_field IN LISTS _fields)
        _litertlm_lib_candidates(_candidates "${_field}")
        foreach(_candidate IN LISTS _candidates)
            if(EXISTS "${_candidate}" AND NOT IS_DIRECTORY "${_candidate}")
                set(_resolved "${_candidate}")
                break()
            endif()
        endforeach()
        if(_resolved)
            break()
        endif()
    endforeach()

    if(NOT _resolved)
        list(GET _fields 0 _first)
        list(APPEND _missing "${_first}")
        continue()
    endif()

    # An archive listed twice would be whole-archived twice.
    string(TOLOWER "${_resolved}" _key)
    if(_key IN_LIST _seen)
        continue()
    endif()
    list(APPEND _seen "${_key}")

    if(_mode STREQUAL "WHOLE")
        string(APPEND _rsp_content "\"/WHOLEARCHIVE:${_resolved}\"\n")
        math(EXPR _whole_count "${_whole_count} + 1")
    else()
        string(APPEND _rsp_content "\"${_resolved}\"\n")
        math(EXPR _lib_count "${_lib_count} + 1")
    endif()

    file(TIMESTAMP "${_resolved}" _timestamp UTC)
    string(APPEND _stamp_content "${_mode}|${_resolved}|${_timestamp}\n")
endforeach()

foreach(_lib IN LISTS _missing)
    message(STATUS "[LiteRTLM] Not built on this platform, skipped: ${_lib}")
endforeach()
message(STATUS
    "[LiteRTLM] Link inputs: ${_whole_count} whole archives, ${_lib_count} archives -> ${RSP_FILE}")

set(_previous_stamp "")
if(EXISTS "${STAMP_FILE}")
    file(READ "${STAMP_FILE}" _previous_stamp)
endif()
if(NOT EXISTS "${RSP_FILE}" OR NOT _previous_stamp STREQUAL _stamp_content)
    file(WRITE "${RSP_FILE}" "${_rsp_content}")
    file(WRITE "${STAMP_FILE}" "${_stamp_content}")
endif()
