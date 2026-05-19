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


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# 1. Audio Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_audio_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::Audio ALIAS runtime_components_preprocessor_audio_preprocessor)

target_include_directories(runtime_components_preprocessor_audio_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
)

# ==============================================================================
# 2. Audio Preprocessor MiniAudio
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_audio_preprocessor_miniaudio STATIC
  audio_preprocessor_miniaudio.cc
)
add_library(LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio ALIAS runtime_components_preprocessor_audio_preprocessor_miniaudio)

target_include_directories(runtime_components_preprocessor_audio_preprocessor_miniaudio
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_preprocessor_audio_preprocessor_miniaudio
  PUBLIC
    LiteRTLM::Runtime::Components::Preprocessor::Audio
    LiteRTLM::Runtime::Components::Preprocessor::MelFilterBank
    runtime_engine_io_types
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# 3. Bypass Audio Preprocessor
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_by_pass_audio_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::AudioBypass ALIAS runtime_components_preprocessor_by_pass_audio_preprocessor)

target_include_directories(runtime_components_preprocessor_by_pass_audio_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
)

# ==============================================================================
# 4. Bypass Image Preprocessor
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_by_pass_image_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::ImageBypass ALIAS runtime_components_preprocessor_by_pass_image_preprocessor)

target_include_directories(runtime_components_preprocessor_by_pass_image_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
)

# ==============================================================================
# 5. Image Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_image_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::Image ALIAS runtime_components_preprocessor_image_preprocessor)

target_include_directories(runtime_components_preprocessor_image_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

# ==============================================================================
# 6. Mel Filterbank
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_mel_filterbank STATIC
  mel_filterbank.cc
)
add_library(LiteRTLM::Runtime::Components::Preprocessor::MelFilterBank ALIAS runtime_components_preprocessor_mel_filterbank)

target_include_directories(runtime_components_preprocessor_mel_filterbank
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_preprocessor_mel_filterbank
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# 7. Signal Vector Util
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_signal_vector_util INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::SignalVectorUtil ALIAS runtime_components_preprocessor_signal_vector_util)

target_include_directories(runtime_components_preprocessor_signal_vector_util
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# 8. STB Image Preprocessor
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_stb_image_preprocessor STATIC
  stb_image_preprocessor.cc
  image_preprocessor_utils.cc
)
add_library(LiteRTLM::Runtime::Components::Preprocessor::StbImage ALIAS runtime_components_preprocessor_stb_image_preprocessor)

target_include_directories(runtime_components_preprocessor_stb_image_preprocessor
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_preprocessor_stb_image_preprocessor
  PUBLIC
    LiteRTLM::Runtime::Components::Preprocessor::Image
    runtime_engine_io_types
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util
    LITERTLM_DEPS
    stb_lib
)

# ==============================================================================
# 9. Folder Facade
# ==============================================================================
add_library(runtime_components_preprocessor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor ALIAS runtime_components_preprocessor_libs)

target_link_libraries(runtime_components_preprocessor_libs INTERFACE
  LiteRTLM::Runtime::Components::Preprocessor::Audio
  LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio
  LiteRTLM::Runtime::Components::Preprocessor::AudioBypass
  LiteRTLM::Runtime::Components::Preprocessor::ImageBypass
  LiteRTLM::Runtime::Components::Preprocessor::Image
  LiteRTLM::Runtime::Components::Preprocessor::MelFilterBank
  LiteRTLM::Runtime::Components::Preprocessor::SignalVectorUtil
  LiteRTLM::Runtime::Components::Preprocessor::StbImage
)