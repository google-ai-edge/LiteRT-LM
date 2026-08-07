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
# Audio Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_audio_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::AudioInterface
  ALIAS runtime_components_preprocessor_audio_preprocessor)

target_include_directories(runtime_components_preprocessor_audio_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
)

# ==============================================================================
# Bypass Audio Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_by_pass_audio_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::AudioBypassInterface
  ALIAS runtime_components_preprocessor_by_pass_audio_preprocessor)

target_include_directories(runtime_components_preprocessor_by_pass_audio_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
)

# ==============================================================================
# Bypass Image Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_by_pass_image_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::BypassImageInterface
  ALIAS runtime_components_preprocessor_by_pass_image_preprocessor)

target_include_directories(runtime_components_preprocessor_by_pass_image_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
)

# ==============================================================================
# Image Preprocessor Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_image_preprocessor INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::ImageInterface
  ALIAS runtime_components_preprocessor_image_preprocessor)

target_include_directories(runtime_components_preprocessor_image_preprocessor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

# ==============================================================================
# Signal Vector Util Interface
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_signal_vector_util INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor::SignalVectorUtilInterface
  ALIAS runtime_components_preprocessor_signal_vector_util)

target_include_directories(runtime_components_preprocessor_signal_vector_util
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_components_preprocessor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::Preprocessor
  ALIAS runtime_components_preprocessor_libs)

target_link_libraries(runtime_components_preprocessor_libs INTERFACE
  LiteRTLM::Runtime::Components::Preprocessor::AudioInterface
  LiteRTLM::Runtime::Components::Preprocessor::AudioBypassInterface
  LiteRTLM::Runtime::Components::Preprocessor::BypassImageInterface
  LiteRTLM::Runtime::Components::Preprocessor::ImageInterface
  LiteRTLM::Runtime::Components::Preprocessor::SignalVectorUtilInterface
)