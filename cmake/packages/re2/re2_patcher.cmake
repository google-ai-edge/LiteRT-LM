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


message(STATUS "[LiteRTLM] Patching RE2...")

set(ROOT_LIST "${LITERTLM_RE2_SRC_DIR}/CMakeLists.txt")

if(EXISTS "${ROOT_LIST}")
    file(READ "${ROOT_LIST}" ROOT_CONTENT)

    string(REPLACE 
        "set(RE2_CXX_VERSION cxx_std_17)"
        "set(RE2_CXX_VERSION cxx_std_20)\ninclude(${LITERTLM_RE2_SHIM_PATH})"
        ROOT_CONTENT "${ROOT_CONTENT}"
    )

    file(WRITE "${ROOT_LIST}" "${ROOT_CONTENT}")
    message(STATUS "[LiteRTLM] RE2 CMakeLists.txt patching successful.")
else()
    message(FATAL_ERROR "Could not find RE2 CMakeLists.txt at ${ROOT_LIST}")
endif()