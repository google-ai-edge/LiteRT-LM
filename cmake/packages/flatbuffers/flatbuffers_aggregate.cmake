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

include("${LITERTLM_FLATBUFFERS_PACKAGE_DIR}/flatbuffers_target_map.cmake")

macro(generate_flatbuffers_aggregate)
    if(NOT TARGET LiteRTLM::flatbuffers::flatbuffers)
        message(STATUS "[LiteRTLM] Generating the Flatbuffers aggregate...")
        set(_flatbuffers_lib_names "")
        set(_flatbuffers_lib_paths "")
        kvp_parse_map("${LITERTLM_FLATBUFFERS_TARGET_MAP}" _flatbuffers_lib_names _flatbuffers_lib_paths)

        add_library(LiteRTLM::flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::flatbuffers::flatbuffers PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_flatbuffers_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_flatbuffers_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_flatbuffers_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::flatbuffers::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::flatbuffers::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
        )

        foreach(_comp_target IN LISTS ${_flatbuffers_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::flatbuffers::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to aggregate")
            endif()
        endforeach()

        get_target_property(_FLATBUFFERS_PAYLOAD LiteRTLM::flatbuffers::flatbuffers INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _FLATBUFFERS_LINK_FLAGS "${_FLATBUFFERS_PAYLOAD}")

        set(FlatBuffers_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
        set(flatbuffers_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
        message(STATUS "[LiteRTLM] Flatbuffers aggregate has been generated.")
    endif()
endmacro()


macro(generate_flatc_aggregate)
    if(NOT TARGET LiteRTLM::flatbuffers::flatc)
        message(STATUS "[LiteRTLM] Generating the flatc aggregate...")

        add_executable(LiteRTLM::flatbuffers::flatc IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::flatbuffers::flatc PROPERTIES
            IMPORTED_LOCATION
                "${LITERTLM_FLATC_EXECUTABLE}"
        )

        add_library(LiteRTLM::flatc::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::flatc::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_FLATC_INCLUDE_DIR}"
        )

        set(_flatc_lib_names "")
        set(_flatc_lib_paths "")
        kvp_parse_map("${LITERTLM_FLATC_TARGET_MAP}" _flatc_lib_names _flatc_lib_paths)

        foreach(_comp_target IN LISTS ${_flatc_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::flatc::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to aggregate")
            endif()
        endforeach()

        set(FlatC_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
        set(flatc_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
        message(STATUS "[LiteRTLM] flatc aggregate has been generated.")
    endif()
endmacro()