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

include(ExternalProject)

set(LITERTLM_OPENCL_EXT_PREFIX ${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/opencl_headers CACHE INTERNAL "")
set(LITERTLM_OPENCL_SRC_DIR ${LITERTLM_OPENCL_EXT_PREFIX}/src/opencl_headers_external)
set(LITERTLM_OPENCL_INCLUDE_DIR ${LITERTLM_OPENCL_SRC_DIR} CACHE INTERNAL "" FORCE)
set(LITERTLM_OPENCL_TAG "v2024.05.08" CACHE STRING "OpenCL headers git tag")

ExternalProject_Add(
  opencl_headers_external
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG        ${LITERTLM_OPENCL_TAG}
  PREFIX         ${LITERTLM_OPENCL_EXT_PREFIX}
  UPDATE_COMMAND
      git fetch origin ${LITERTLM_OPENCL_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    PATCH_COMMAND
      ${CMAKE_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  GIT_SHALLOW       TRUE
)

add_library(opencl_headers_lib INTERFACE)
add_dependencies(opencl_headers_lib opencl_headers_external)
target_include_directories(opencl_headers_lib SYSTEM INTERFACE ${LITERTLM_OPENCL_INCLUDE_DIR})