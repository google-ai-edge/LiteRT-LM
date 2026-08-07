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

include(${LITERTLM_SENTENCEPIECE_PACKAGE_DIR}/sentencepiece_target_map.cmake)

macro(generate_sentencepiece_aggregate)
    if(NOT TARGET LiteRTLM::sentencepiece::sentencepiece)
        message(STATUS "[LiteRTLM] Generating the litert aggregate...")

        set(_sentencepiece_lib_names "")
        set(_sentencepiece_lib_paths "")
        kvp_parse_map("${LITERTLM_SENTENCEPIECE_TARGET_MAP}" _sentencepiece_lib_names _sentencepiece_lib_paths)

        add_library(LiteRTLM::sentencepiece::sentencepiece INTERFACE IMPORTED GLOBAL)

        set_target_properties(LiteRTLM::sentencepiece::sentencepiece PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_sentencepiece_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_sentencepiece_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_sentencepiece_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_SENTENCEPIECE_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::sentencepiece::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::sentencepiece::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_SENTENCEPIECE_INCLUDE_DIR};${LITERTLM_SENTENCEPIECE_BUILD_DIR}"
        )

        foreach(_comp_target IN LISTS ${_sentencepiece_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::sentencepiece::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to sentencepiece shim")
            endif()
        endforeach()

        if(NOT TARGET sentencepiece_libs)
            add_library(sentencepiece_libs ALIAS LiteRTLM::sentencepiece::sentencepiece)
        endif()

        get_target_property(_SENTENCEPIECE_PAYLOAD LiteRTLM::sentencepiece::sentencepiece INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _SENTENCEPIECE_LINK_FLAGS "${_SENTENCEPIECE_PAYLOAD}")

        set(SENTENCEPIECE_FOUND TRUE CACHE BOOL "" FORCE)
        message(STATUS "[LiteRTLM] litert aggregate has been generated.")
    endif()
endmacro()
