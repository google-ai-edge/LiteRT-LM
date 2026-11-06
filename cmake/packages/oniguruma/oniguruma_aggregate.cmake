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

include("${LITERTLM_ONIGURUMA_PACKAGE_DIR}/oniguruma_target_map.cmake")

macro(generate_oniguruma_aggregate)
    if(NOT TARGET LiteRTLM::oniguruma::oniguruma)
        set(_oniguruma_lib_names "")
        set(_oniguruma_lib_paths "")
        kvp_parse_map("${LITERTLM_ONIGURUMA_TARGET_MAP}" _oniguruma_lib_names _oniguruma_lib_paths)

        add_library(LiteRTLM::oniguruma::oniguruma INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::oniguruma::oniguruma PROPERTIES
            INTERFACE_LIBRARY_NAMES
                "${_oniguruma_lib_names}"
            INTERFACE_LIBRARY_PATHS
                "${_oniguruma_lib_paths}"
            INTERFACE_LINK_LIBRARIES
                "${_oniguruma_lib_paths}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_ONIGURUMA_INCLUDE_DIR}"
        )

        add_library(LiteRTLM::oniguruma::shim INTERFACE IMPORTED GLOBAL)
        set_target_properties(LiteRTLM::oniguruma::shim PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${LITERTLM_ONIGURUMA_INCLUDE_DIR}"
            # Oniguruma is linked statically; without ONIG_STATIC its header
            # declares the API with __declspec(dllimport) on MSVC.
            INTERFACE_COMPILE_DEFINITIONS
                "ONIG_STATIC"
        )

        set(oniguruma_FOUND TRUE CACHE INTERNAL "Forced by LiteRTLM" FORCE)
    endif()
endmacro()
