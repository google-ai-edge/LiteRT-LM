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
include("${LITERTLM_TFLITE_PACKAGE_DIR}/tflite_target_map.cmake")

macro(generate_tflite_aggregate)
    if(NOT TARGET LiteRTLM::tflite::tflite)
        message(STATUS "[LiteRTLM] Generating the TFLite aggregate...")

        set(_tflite_lib_names "")
        set(_tflite_lib_paths "")
        kvp_parse_map("${LITERTLM_TFLITE_TARGET_MAP}" _tflite_lib_names _tflite_lib_paths)

        add_library(LiteRTLM::tflite::tflite INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::tflite::tflite PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_tflite_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_tflite_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_tflite_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_TFLITE_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::tflite::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::tflite::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_TFLITE_INCLUDE_DIR};${LITERTLM_TFLITE_BUILD_DIR}"
        )

        foreach(_comp_target IN LISTS ${_tflite_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::tflite::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to TFLite aggregate")
            endif()
        endforeach()

        get_target_property(_TFLITE_PAYLOAD LiteRTLM::tflite::tflite INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _TFLITE_LINK_FLAGS "${_TFLITE_PAYLOAD}")

        if(NOT TARGET tflite_libs)
            add_library(tflite_libs ALIAS LiteRTLM::tflite::shim)
        endif()
        if(NOT TARGET tensorflow-lite)
            add_library(tensorflow-lite ALIAS LiteRTLM::tflite::shim)
        endif()

        set(tflite_FOUND TRUE CACHE BOOL "" FORCE)
        message(STATUS "[LiteRTLM] TFLite aggregate has been generated.")
    endif()
endmacro()
