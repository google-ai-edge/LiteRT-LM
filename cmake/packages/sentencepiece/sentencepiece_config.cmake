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

set(LITERTLM_SENTENCEPIECE_EXT_PREFIX
  "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/sentencepiece"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_INSTALL_PREFIX
  "${LITERTLM_SENTENCEPIECE_EXT_PREFIX}/install"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_SRC_DIR
  "${LITERTLM_SENTENCEPIECE_EXT_PREFIX}/src/sentencepiece_external"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_INCLUDE_DIR
  "${LITERTLM_SENTENCEPIECE_INSTALL_PREFIX}/include"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_BUILD_DIR 
  "${LITERTLM_SENTENCEPIECE_EXT_PREFIX}/src/sentencepiece_external-build"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_INCLUDE_PATHS
  "${LITERTLM_SENTENCEPIECE_INCLUDE_DIR}"
  "${LITERTLM_SENTENCEPIECE_BUILD_DIR}/src"
  "${LITERTLM_SENTENCEPIECE_BUILD_DIR}/src/builtin_pb"
  CACHE PATH "")

if(EXISTS "${LITERTLM_SENTENCEPIECE_INSTALL_PREFIX}/lib64")
  set(LITERTLM_SENTENCEPIECE_LIB_DIR
  "${LITERTLM_SENTENCEPIECE_INSTALL_PREFIX}/lib64"
   CACHE PATH "")
else()
  set(LITERTLM_SENTENCEPIECE_LIB_DIR
  "${LITERTLM_SENTENCEPIECE_INSTALL_PREFIX}/lib"
   CACHE PATH "")
endif()

set(LITERTLM_SENTENCEPIECE_LIBRARY_STATIC
  "${LITERTLM_SENTENCEPIECE_LIB_DIR}/libsentencepiece.a"
  CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_LIBRARY_TRAIN 
  "${LITERTLM_SENTENCEPIECE_LIB_DIR}/libsentencepiece_train.a"
  CACHE PATH "")

set(LITERTLM_SENTENCEPIECE_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/sentencepiece"
    CACHE PATH "Path to SentencePiece related build scripts")
set(LITERTLM_SENTENCEPIECE_ROOT_SHIM_PATH
    "${LITERTLM_SENTENCEPIECE_PACKAGE_DIR}/sentencepiece_root_shim.cmake"
    CACHE PATH "")
set(LITERTLM_SENTENCEPIECE_SRC_SHIM_PATH
    "${LITERTLM_SENTENCEPIECE_PACKAGE_DIR}/sentencepiece_src_shim.cmake"
    CACHE PATH "")

set(LITERTLM_SENTENCEPIECE_AGGREGATE_PATH
    "${LITERTLM_SENTENCEPIECE_PACKAGE_DIR}/sentencepiece_aggregate.cmake"
    CACHE PATH "")