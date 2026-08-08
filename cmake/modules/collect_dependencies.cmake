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

list(APPEND LITERTLM_INCLUDE_PATHS
  "${LITERTLM_OPENCL_HDR_DIR}"
  "${LITERTLM_ABSL_INCLUDE_DIR}"
  "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
  "${LITERTLM_GTEST_INCLUDE_DIR}"
  "${LITERTLM_PROTOBUF_INCLUDE_DIR}"
  "${LITERTLM_RE2_INCLUDE_DIR}"
  "${LITERTLM_SENTENCEPIECE_INCLUDE_PATHS}"
  "${LITERTLM_TOKENIZERS_INCLUDE_DIR}"
  "${LITERTLM_TFLITE_INCLUDE_DIR}"
  "${LITERTLM_TFLITE_SRC_DIR}"
  "${LITERTLM_RUY_INCLUDE_DIR}"
  "${LITERTLM_TENSORFLOW_SRC_DIR}"
  "${LITERTLM_LITERT_INCLUDE_PATHS}"
  "${LITERTLM_FETCHCONTENT_MODULE_SRC_DIRS}"
  "${LITERTLM_FETCHCONTENT_MODULE_INCLUDE_DIR}"
  "${CMAKE_BINARY_DIR}/antlr_generated"
)

add_library(LITERTLM_DEPS INTERFACE)
add_dependencies(LITERTLM_DEPS
    litert_external
    tflite_external
    opencl_headers_external
    re2_external
    litertlm_cxx_bridge
    tokenizers-cpp_external
    sentencepiece_external
    flatbuffers_external
    litertlm_generated_protobuf
    protobuf_external
    absl_external
)
target_link_libraries(LITERTLM_DEPS INTERFACE
    libpng_lib
    kissfft_lib
    miniaudio_lib
    minizip_lib
    minja_lib
    antlr_lib
    zlib_lib

    litertlm_cxx_bridge
    litert_lm_deps

    LiteRTLM::litert::shim
    LiteRTLM::tflite::shim
    LiteRTLM::tokenizers::tokenizers
    LiteRTLM::sentencepiece::shim
    LiteRTLM::re2::shim
    LiteRTLM::flatbuffers::shim
    LiteRTLM::protobuf::shim
    LiteRTLM::absl::shim

    json_lib
    opencl_headers_lib
)

target_include_directories(LITERTLM_DEPS SYSTEM INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)