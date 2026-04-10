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


include_guard(GLOBAL)
include(${LITERTLM_MODULES_DIR}/utils.cmake)
include(${ABSL_PACKAGE_DIR}/absl_target_map.cmake)

macro(generate_absl_aggregate)
    if(NOT TARGET LiteRTLM::absl::absl)
        message(STATUS "[LiteRTLM] Generating the Abseil aggregate...")
        set(_absl_lib_names "")
        set(_absl_lib_paths "")
        kvp_parse_map("${ABSL_TARGET_MAP}" _absl_lib_names _absl_lib_paths)

        add_library(LiteRTLM::absl::absl INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::absl::absl PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_absl_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_absl_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_absl_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${ABSL_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::absl::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::absl::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${ABSL_INCLUDE_DIR}"
            )

        foreach(_comp_target IN LISTS ${_absl_lib_names})
            if(NOT TARGET ${_comp_target})
                add_library(${_comp_target} ALIAS LiteRTLM::absl::shim)
                message(STATUS "[LiteRTLM] Redirected ${_comp_target} to aggregate")
            endif()
        endforeach()

        foreach(_target IN ITEMS ${_absl_exhaustive_targets})
            if(NOT TARGET ${_target})
                add_library(${_target} ALIAS LiteRTLM::absl::shim)
            endif()
        endforeach()

        set(absl_FOUND TRUE CACHE BOOL "${ABSL_CONFIG_CMAKE_FILE}" FORCE)
        set(Abseil_FOUND TRUE CACHE BOOL "${ABSL_CONFIG_CMAKE_FILE}" FORCE)
        set(absl_DIR "Abseil merged archive" CACHE PATH "" FORCE)

        get_target_property(_ABSL_PAYLOAD LiteRTLM::absl::absl INTERFACE_LINK_LIBRARIES)
        string(REPLACE ";" " " _ABSL_LINK_FLAGS "${_ABSL_PAYLOAD}")
        message(STATUS "[LiteRTLM] Abseil aggregate has been generated.")
    endif()
endmacro()

