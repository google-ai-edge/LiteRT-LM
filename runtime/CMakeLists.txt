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

add_subdirectory(components)
add_subdirectory(conversation)
add_subdirectory(core)
add_subdirectory(engine)
add_subdirectory(executor)
add_subdirectory(framework)
add_subdirectory(util)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_libs INTERFACE)
add_library(LiteRTLM::Runtime ALIAS runtime_libs)

target_link_libraries(runtime_libs INTERFACE
  LiteRTLM::Runtime::Components
  LiteRTLM::Runtime::Conversation
  LiteRTLM::Runtime::Core
  LiteRTLM::Runtime::Engine
  LiteRTLM::Runtime::Executor
  LiteRTLM::Runtime::Framework
  LiteRTLM::Runtime::Util
)