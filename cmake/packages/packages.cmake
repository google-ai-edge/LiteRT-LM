# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set(LITERTLM_ABSL_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/absl"
    CACHE PATH "Path to absl related build scripts")

set(LITERTLM_GTEST_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/gtest"
    CACHE PATH "Path to gtest related build scripts")

set(LITERTLM_FLATBUFFERS_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/flatbuffers"
    CACHE PATH "Path to Flatbuffers related build scripts")

set(LITERTLM_PROTOBUF_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/protobuf"
    CACHE PATH "Path to Protobuf related build scripts")

set(LITERTLM_RE2_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/re2"
    CACHE PATH "Path to RE2 related build scripts")

set(LITERTLM_SENTENCEPIECE_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/sentencepiece"
    CACHE PATH "Path to SentencePiece related build scripts")

set(LITERTLM_TOKENIZERS_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/tokenizers"
    CACHE PATH "Path to Tokenizers-cpp related build scripts")

set(LITERTLM_TFLITE_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/tflite"
    CACHE PATH "Path to TFLite related build scripts")

set(LITERTLM_LITERT_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/litert"
    CACHE PATH "Path to LiteRT related build scripts")

set(LITERTLM_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/litert_lm"
    CACHE PATH "Path to LiteRT-LM related build scripts")
