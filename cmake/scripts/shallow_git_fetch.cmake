# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Expected inputs via -D: SRC_DIR, REPO_URL, TAG

if(NOT DEFINED SRC_DIR OR NOT DEFINED REPO_URL OR NOT DEFINED TAG)
  message(FATAL_ERROR "[LiteRTLM] shallow_git_fetch requires SRC_DIR, REPO_URL, and TAG")
endif()

if(NOT EXISTS "${SRC_DIR}/.git")
  file(MAKE_DIRECTORY "${SRC_DIR}")
  execute_process(
    COMMAND git init
    WORKING_DIRECTORY "${SRC_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND git remote add origin "${REPO_URL}"
    WORKING_DIRECTORY "${SRC_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
else()
  execute_process(
    COMMAND git remote set-url origin "${REPO_URL}"
    WORKING_DIRECTORY "${SRC_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

message(STATUS "[LiteRTLM] Fetching shallow ref ${TAG} from ${REPO_URL} into ${SRC_DIR}")
execute_process(
  COMMAND git fetch --depth 1 origin "${TAG}"
  WORKING_DIRECTORY "${SRC_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND git checkout --force FETCH_HEAD
  WORKING_DIRECTORY "${SRC_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND git clean -dfx
  WORKING_DIRECTORY "${SRC_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

if(EXISTS "${SRC_DIR}/.gitmodules")
  message(STATUS "[LiteRTLM] Updating submodules with depth 1 in ${SRC_DIR}")
  execute_process(
    COMMAND git submodule update --init --recursive --depth 1
    WORKING_DIRECTORY "${SRC_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()
