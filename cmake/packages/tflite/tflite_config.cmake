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

set(LITERTLM_TFLITE_EXT_PREFIX       "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/tensorflow" CACHE PATH "")
set(LITERTLM_TFLITE_INSTALL_PREFIX   "${LITERTLM_TFLITE_EXT_PREFIX}/install" CACHE PATH "")
set(LITERTLM_TFLITE_PACKAGE_DIR      "${LITERTLM_CMAKE_PACKAGES_DIR}/tflite" CACHE PATH "Path to TFLite related build scripts")
set(LITERTLM_TFLITE_CONFIG_PATH      "${LITERTLM_TFLITE_PACKAGE_DIR}/tflite_config.cmake" CACHE PATH "")
set(LITERTLM_TFLITE_AGGREGATE_PATH   "${LITERTLM_TFLITE_PACKAGE_DIR}/tflite_aggregate.cmake" CACHE PATH "")
set(LITERTLM_TFLITE_INCLUDE_DIR      "${LITERTLM_TFLITE_INSTALL_PREFIX}/include" CACHE PATH "")
set(LITERTLM_TFLITE_LIB_DIR          "${LITERTLM_TFLITE_INSTALL_PREFIX}/lib" CACHE PATH "")
set(LITERTLM_TFLITE_SRC_DIR          "${LITERTLM_TFLITE_EXT_PREFIX}/src/tflite_external/tensorflow/lite" CACHE PATH "")
set(LITERTLM_TFLITE_BUILD_DIR        "${LITERTLM_TFLITE_EXT_PREFIX}/src/tflite_external-build" CACHE PATH "TFLite Build Directory")
set(LITERTLM_TENSORFLOW_SRC_DIR      "${LITERTLM_TFLITE_EXT_PREFIX}/src/tflite_external" CACHE PATH "")
set(LITERTLM_TFLITE_STATIC_LIB       "${LITERTLM_TFLITE_BUILD_DIR}/libtensorflow-lite.a" CACHE PATH "")
set(LITERTLM_RUY_INCLUDE_DIR         "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/tflite_external-build/ruy" CACHE PATH "")
set(LITERTLM_TFLITE_CMAKELISTS       "${LITERTLM_TFLITE_SRC_DIR}/CMakeLists.txt" CACHE PATH "")

set(LITERTLM_XNNPACK_SRC_DIR "${LITERTLM_TFLITE_BUILD_DIR}/xnnpack" CACHE PATH "")
set(LITERTLM_XNNPACK_BINARY_DIR "${LITERTLM_TFLITE_BUILD_DIR}/_deps/xnnpack-build" CACHE PATH "")
set(LITERTLM_XNNPACK_INCLUDE_DIR "${LITERTLM_TFLITE_BUILD_DIR}/xnnpack/include" CACHE PATH "")
set(LITERTLM_XNNPACK_LIB_DIR "${LITERTLM_TFLITE_BUILD_DIR}/_deps/xnnpack-build" CACHE PATH "")
set(LITERTLM_XNNPACK_MOD_FILE "${LITERTLM_TFLITE_SRC_DIR}/tools/cmake/modules/xnnpack/CMakeLists.txt")
set(LITERTLM_XNNPACK_DELEGATE_CMAKELISTS "${LITERTLM_TFLITE_CMAKELISTS}" CACHE PATH "")
set(LITERTLM_XNNPACK_CMAKELISTS "${LITERTLM_TFLITE_CMAKELISTS}" CACHE PATH "")
