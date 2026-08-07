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
# LiteRT-LM Web
# ==============================================================================
add_litertlm_library(js_core_litertlm_web STATIC
  litertlm_web.cc
)
add_library(LiteRTLM::JS::Core::LitertLmWeb ALIAS js_core_litertlm_web)

target_include_directories(js_core_litertlm_web
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(js_core_litertlm_web
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# LiteRT-LM Web
# ==============================================================================
add_litertlm_library(js_core_readable_stream_data_stream STATIC
  readable_stream_data_stream.cc
)
add_library(LiteRTLM::JS::Core::ReadableStreamDataStream ALIAS js_core_readable_stream_data_stream)

target_include_directories(js_core_readable_stream_data_stream
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(js_core_readable_stream_data_stream
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(js_core_libs INTERFACE)
add_library(LiteRTLM::JS::Core ALIAS js_core_libs)

target_link_libraries(js_core_libs INTERFACE
  LiteRTLM::JS::Core::LitertLmWeb
  LiteRTLM::JS::Core::ReadableStreamDataStream
)