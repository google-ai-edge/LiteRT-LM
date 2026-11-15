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
# Context Handler
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_context_handler STATIC
  context_handler.cc
)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::ContextHandler ALIAS runtime_framework_resource_manager_context_handler)

target_include_directories(runtime_framework_resource_manager_context_handler
  PUBLIC
  ${LITERTLM_GENERATED_SRC_DIR}
  ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_resource_manager_context_handler
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

