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

include("${LITERTLM_PROTOBUF_TARGET_MAP_PATH}")

macro(generate_protobuf_aggregate)
    if(NOT TARGET LiteRTLM::protobuf::libprotobuf)
        message(STATUS "[LiteRTLM] Generating the Protobuf aggregate...")
        set(_protobuf_lib_names "")
        set(_protobuf_lib_paths "")
        kvp_parse_map("${LITERTLM_PROTOBUF_TARGET_MAP}" _protobuf_lib_names _protobuf_lib_paths)

        add_library(LiteRTLM::protobuf::libprotobuf INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::protobuf::libprotobuf PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_protobuf_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_protobuf_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_protobuf_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_PROTOBUF_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::protobuf::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::protobuf::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_PROTO_INCLUDE_DIR}"
        )

        foreach(_comp_target IN LISTS ${_protobuf_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::protobuf::shim)
                message(VERBOSE "[LiteRTLM] Redirected ${_comp_target} to aggregate")
            endif()
        endforeach()

        if(NOT TARGET protobuf::libprotobuf)
            add_library(protobuf::libprotobuf ALIAS LiteRTLM::protobuf::shim)
        endif()

        if(NOT TARGET protobuf::protoc)
            add_executable(protobuf::protoc IMPORTED GLOBAL)
            set_target_properties(protobuf::protoc PROPERTIES
                IMPORTED_LOCATION "${LITERTLM_PROTOC_EXECUTABLE}"
            )
        endif()

        get_target_property(_PROTOBUF_PAYLOAD LiteRTLM::protobuf::libprotobuf INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _PROTOBUF_LINK_FLAGS "${_PROTOBUF_PAYLOAD}")

        message(STATUS "[LiteRTLM] Protobuf aggregate has been generated.")
    endif()
endmacro()

