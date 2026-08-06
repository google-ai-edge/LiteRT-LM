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

# ==============================================================================
# Audio Preprocessor MiniAudio
# ==============================================================================
add_library(support_preprocessor_audio_preprocessor_miniaudio STATIC
  audio_preprocessor_miniaudio.cc
)
add_library(LiteRT::Support::Preprocessor::AudioMiniAudio ALIAS support_preprocessor_audio_preprocessor_miniaudio)

target_include_directories(support_preprocessor_audio_preprocessor_miniaudio
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PUBLIC
    ${LITERT_KISSFFT_SRC_DIR}
)

target_link_libraries(support_preprocessor_audio_preprocessor_miniaudio
  PUBLIC
    absl::memory
    absl::status
    absl::strings
    litert_cc_api
    LiteRT::Support::Preprocessor::MelFilterBank
    LiteRT::Support::Util::IoTypes
    kissfft
)

# ==============================================================================
# Image Preprocessor Util
# ==============================================================================
add_library(support_preprocessor_image_preprocessor_utils STATIC
  image_preprocessor_utils.cc
)
add_library(LiteRT::Support::Preprocessor::ImagePreprocessorUtils ALIAS support_preprocessor_image_preprocessor_utils)

target_include_directories(support_preprocessor_image_preprocessor_utils
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(support_preprocessor_image_preprocessor_utils
  PUBLIC
    LiteRT::Support::Util::IoTypes   
    litert_cc_api
    absl::log
    absl::status
    absl::strings
)


# ==============================================================================
# Mel Filterbank
# ==============================================================================
add_library(support_preprocessor_mel_filterbank STATIC
  mel_filterbank.cc
)
add_library(LiteRT::Support::Preprocessor::MelFilterBank ALIAS support_preprocessor_mel_filterbank)

target_include_directories(support_preprocessor_mel_filterbank
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(support_preprocessor_mel_filterbank
  PUBLIC
  absl::log
  absl::status
  absl::strings
)

# ==============================================================================
# STB Image Preprocessor
# ==============================================================================
add_library(support_preprocessor_stb_image_preprocessor STATIC
  stb_image_preprocessor.cc
)
add_library(LiteRT::Support::Preprocessor::StbImage ALIAS support_preprocessor_stb_image_preprocessor)

target_include_directories(support_preprocessor_stb_image_preprocessor
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PUBLIC
    ${LITERT_STBLIB_SRC_DIR}
)

target_link_libraries(support_preprocessor_stb_image_preprocessor
  PUBLIC
    LiteRT::Support::Util::IoTypes
    litert_cc_api
    absl::algorithm
    absl::log
    absl::status
    absl::strings
    stblib
)

# ==============================================================================
# Stub Image Preprocessor
# ==============================================================================
add_library(support_preprocessor_stub_image_preprocessor STATIC
  stub_image_preprocessor.cc
)
add_library(LiteRT::Support::Preprocessor::StubImage ALIAS support_preprocessor_stub_image_preprocessor)

target_include_directories(support_preprocessor_stub_image_preprocessor
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(support_preprocessor_stub_image_preprocessor
  PUBLIC
    LiteRT::Support::Util::IoTypes
    litert_cc_api
    absl::status
    absl::strings
)

