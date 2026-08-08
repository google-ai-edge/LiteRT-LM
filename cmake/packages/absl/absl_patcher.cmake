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
include("${LITERTLM_ABSL_CONFIG_PATH}")

message(STATUS "[LiteRTLM] Injecting shim into Abseil-cpp root...")

set(ROOT_LIST "${LITERTLM_ABSL_SRC_DIR}/absl/CMakeLists.txt")
if(EXISTS "${ROOT_LIST}")
    file(READ "${ROOT_LIST}" ROOT_CONTENT)

    string(PREPEND ROOT_CONTENT "include(${LITERTLM_ABSL_SHIM_PATH})\n")

    file(WRITE "${ROOT_LIST}" "${ROOT_CONTENT}")
    message(STATUS "[LiteRTLM] Injection successful.")
else()
    message(FATAL_ERROR "Could not find Abseil-cpp CMakeLists.txt at ${ROOT_LIST}")
endif()