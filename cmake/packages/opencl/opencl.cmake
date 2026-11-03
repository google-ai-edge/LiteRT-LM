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
set(LITERTLM_OPENCL_EXTERNAL_DONE "${LITERTLM_OPENCL_EXT_PREFIX}/src/opencl_headers_external-stamp/opencl_headers_external-done" CACHE INTERNAL "")

if(NOT EXISTS "${LITERTLM_OPENCL_EXTERNAL_DONE}")
  ExternalProject_Add(
    opencl_headers_external
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG        ${LITERTLM_OPENCL_TAG}
    PREFIX         ${LITERTLM_OPENCL_EXT_PREFIX}
    UPDATE_COMMAND ""
    PATCH_COMMAND
      ${CMAKE_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    GIT_SHALLOW       TRUE
  )
else()
  message(STATUS "[LiteRTLM] OpenCL headers already present at: ${LITERTLM_OPENCL_INCLUDE_DIR}")
  if(NOT TARGET opencl_headers_external)
    add_custom_target(opencl_headers_external)
  endif()
endif()

add_library(opencl_headers_lib INTERFACE)
add_dependencies(opencl_headers_lib opencl_headers_external)
target_include_directories(opencl_headers_lib SYSTEM INTERFACE ${LITERTLM_OPENCL_INCLUDE_DIR})