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
add_litertlm_library(support_preprocessor_audio_preprocessor_miniaudio STATIC
  audio_preprocessor_miniaudio.cc
)
add_library(LiteRTLM::Support::Preprocessor::AudioMiniAudio ALIAS support_preprocessor_audio_preprocessor_miniaudio)

target_include_directories(support_preprocessor_audio_preprocessor_miniaudio
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_preprocessor_audio_preprocessor_miniaudio
  PUBLIC
    LiteRTLM::Support::Preprocessor::MelFilterBank
    LiteRTLM::Support::Preprocessor::AudioUtils
    LiteRTLM::Support::Util::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Audio Preprocessor Utils
# ==============================================================================
add_litertlm_library(support_preprocessor_audio_preprocessor_utils STATIC
  audio_preprocessor_utils.cc
)
add_library(LiteRTLM::Support::Preprocessor::AudioUtils ALIAS support_preprocessor_audio_preprocessor_utils)

target_include_directories(support_preprocessor_audio_preprocessor_utils
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_preprocessor_audio_preprocessor_utils
  PUBLIC
  LiteRTLM::Support::Preprocessor::MelFilterBank
  LiteRTLM::Support::Util::IoTypes
  LITERTLM_DEPS
)

# ==============================================================================
# Image Preprocessor Util
# ==============================================================================
add_litertlm_library(support_preprocessor_image_preprocessor_utils STATIC
  image_preprocessor_utils.cc
)
add_library(LiteRTLM::Support::Preprocessor::ImagePreprocessorUtils ALIAS support_preprocessor_image_preprocessor_utils)

target_include_directories(support_preprocessor_image_preprocessor_utils
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS} 
)

target_link_libraries(support_preprocessor_image_preprocessor_utils
  PUBLIC
    LiteRTLM::Support::Util::IoTypes   
    LITERTLM_DEPS
)

# ==============================================================================
# Mel Filterbank
# ==============================================================================
add_litertlm_library(support_preprocessor_mel_filterbank STATIC
  mel_filterbank.cc
)
add_library(LiteRTLM::Support::Preprocessor::MelFilterBank ALIAS support_preprocessor_mel_filterbank)

target_include_directories(support_preprocessor_mel_filterbank
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_preprocessor_mel_filterbank
  PUBLIC
    LiteRTLM::Support::Preprocessor::AudioUtils
    LITERTLM_DEPS
)

# ==============================================================================
# STB Image Preprocessor
# ==============================================================================
add_litertlm_library(support_preprocessor_stb_image_preprocessor STATIC
  stb_image_preprocessor.cc
)
add_library(LiteRTLM::Support::Preprocessor::StbImage ALIAS support_preprocessor_stb_image_preprocessor)

target_include_directories(support_preprocessor_stb_image_preprocessor
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_preprocessor_stb_image_preprocessor
  PUBLIC
    LiteRTLM::Support::Util::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(support_processor_libs INTERFACE)
add_library(LiteRTLM::Support::Preprocessor
  ALIAS support_processor_libs)

target_link_libraries(support_processor_libs INTERFACE
  LiteRTLM::Support::Preprocessor::AudioMiniAudio
  LiteRTLM::Support::Preprocessor::AudioUtils
  LiteRTLM::Support::Preprocessor::ImagePreprocessorUtils
  LiteRTLM::Support::Preprocessor::MelFilterBank
  LiteRTLM::Support::Preprocessor::StbImage  
)
