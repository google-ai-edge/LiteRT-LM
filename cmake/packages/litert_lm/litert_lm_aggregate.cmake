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

macro(generate_litertlm_aggregate)
    if(NOT TARGET LiteRTLM::Aggregate)
        message(STATUS "[LiteRTLM] Generating Local Aggregate...")

        get_property(_litertlm_lib_paths GLOBAL PROPERTY LITERTLM_ARCHIVE_REGISTRY)
        get_property(_litertlm_targets GLOBAL PROPERTY LITERTLM_TARGET_REGISTRY)

        add_library(LiteRTLM::Aggregate INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::Aggregate PROPERTIES
            INTERFACE_LINK_LIBRARIES
                "${_litertlm_lib_paths} ${_cxxbridge_paths} ${_litert_lib_paths} ${_tflite_lib_paths} ${_tokenizers_lib_paths} ${_sentencepiece_lib_paths} ${_re2_lib_paths} ${_flatbuffers_lib_paths} ${_protobuf_lib_paths} ${_absl_lib_paths}"
            INTERFACE_LINK_LIBRARIES_CORE
                "${_tokenizers_lib_paths} ${_sentencepiece_lib_paths} ${_re2_lib_paths} ${_flatbuffers_lib_paths} ${_protobuf_lib_paths} ${_absl_lib_paths}"
            INTERFACE_LINK_LIBRARIES_ODML
                "${_litertlm_lib_paths} ${_cxxbridge_paths} ${_llguidance_lib_paths} ${_litert_lib_paths} ${_tflite_lib_paths}"
        )
        if(NOT TARGET litertlm_anchor)
            add_custom_target(litertlm_anchor DEPENDS ${_litertlm_targets})
        endif()
        add_dependencies(LiteRTLM::Aggregate litertlm_anchor)

        get_target_property(_LITERTLM_PAYLOAD LiteRTLM::Aggregate INTERFACE_LINK_LIBRARIES)
        get_target_property(_LITERTLM_CORE_PAYLOAD LiteRTLM::Aggregate INTERFACE_LINK_LIBRARIES_CORE)
        get_target_property(_LITERTLM_ODML_PAYLOAD LiteRTLM::Aggregate INTERFACE_LINK_LIBRARIES_ODML)

        string(REPLACE ";" " " _LITERTLM_LINK_FLAGS "${_LITERTLM_PAYLOAD}")
        message(STATUS "[LiteRTLM] Local Aggregate generated with ${LITERTLM_ARCHIVE_REGISTRY_LENGTH} targets.")
    else()
        message(WARNING "[LiteRTLM] LiteRTLM::Aggregate already exists!")
    endif()
endmacro()