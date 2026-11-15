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

include("${LITERTLM_RE2_PACKAGE_DIR}/re2_target_map.cmake")

macro(generate_re2_aggregate)
    if(NOT TARGET LiteRTLM::re2::re2)
        set(_re2_lib_names "")
        set(_re2_lib_paths "")
        kvp_parse_map("${LITERTLM_RE2_TARGET_MAP}" _re2_lib_names _re2_lib_paths)

        add_library(LiteRTLM::re2::re2 INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::re2::re2 PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_re2_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_re2_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_re2_lib_paths}"
        )

        add_library(LiteRTLM::re2::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::re2::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_RE2_INCLUDE_DIR}"
        )

        set(re2_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
        set(re2_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
    endif()
endmacro()
