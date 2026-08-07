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

message(STATUS "[LiteRTLM] Executing native XNNPACK path neutralization...")

if(DEFINED LITERTLM_XNNPACK_SRC_DIR AND EXISTS "${LITERTLM_XNNPACK_SRC_DIR}")
    file(GLOB_RECURSE _xnnpack_files LIST_DIRECTORIES false 
        "${xnnpack_SOURCE_DIR}/CMakeLists.txt"
        "${xnnpack_SOURCE_DIR}/*.cmake"
    )
    foreach(_f IN LISTS _xnnpack_files)
        file(READ "${_f}" _content)
        if(_content MATCHES "flatbuffers-flatc/bin/flatc")
            string(REPLACE "flatbuffers-flatc/bin/flatc" 
                "${FLATBUFFERS_FLATC_EXECUTABLE}" _new_content "${_content}")
            file(WRITE "${_f}" "${_new_content}")
        endif()
    endforeach()
else()
    message(WARNING "[LiteRTLM] xnnpack_SOURCE_DIR not defined. Skipping neutralization.")
endif()
