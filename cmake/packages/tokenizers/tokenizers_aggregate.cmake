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

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_TOKENIZERS_TARGET_MAP_PATH}")


macro(generate_tokenizers_aggregate)
    if(NOT TARGET LiteRTLM::tokenizers::tokenizers)
        message(STATUS "[LiteRTLM] Generating the Tokenizers-Cpp aggregate...")

        set(_tokenizers_names "")
        set(_tokenizers_paths "")
        kvp_parse_map("${LITERTLM_TOKENIZERS_TARGET_MAP}" _tokenizers_names _tokenizers_paths)

        add_library(LiteRTLM::tokenizers::tokenizers INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::tokenizers::tokenizers PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_tokenizers_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_tokenizers_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_tokenizers_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_TOKENIZERS_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::tokenizers::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::tokenizers::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_TOKENIZERS_INCLUDE_DIR};${LITERTLM_TFLITE_BUILD_DIR}"
        )

        foreach(_comp_target IN LISTS ${_tokenizers_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::tokenizers::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to Tokenizers-Cpp aggregate")
            endif()
        endforeach()

        get_target_property(_TOKENIZERS_PAYLOAD LiteRTLM::tokenizers::tokenizers INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _TOKENIZERS_LINK_FLAGS "${_TOKENIZERS_PAYLOAD}")

        if(NOT TARGET tokenizers_libs)
            add_library(tokenizers_libs ALIAS LiteRTLM::tokenizers::shim)
        endif()
        if(NOT TARGET tensorflow-lite)
            add_library(tensorflow-lite ALIAS LiteRTLM::tokenizers::shim)
        endif()

        set(tokenizers_FOUND TRUE CACHE BOOL "" FORCE)
        message(STATUS "[LiteRTLM] Tokenizers-Cpp aggregate has been generated.")
    endif()
endmacro()
