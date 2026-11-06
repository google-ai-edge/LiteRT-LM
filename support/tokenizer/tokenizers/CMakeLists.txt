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

add_subdirectory(utils)
add_subdirectory(unicode)
add_subdirectory(tokenizer)
add_subdirectory(models)
add_subdirectory(normalizers)
add_subdirectory(pre_tokenizers)
add_subdirectory(decoders)
add_subdirectory(processors)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(support_tokenizer_tokenizers_libs INTERFACE)
add_library(LiteRTLM::Support::Tokenizer::Tokenizers
  ALIAS support_tokenizer_tokenizers_libs)

target_link_libraries(support_tokenizer_tokenizers_libs INTERFACE
  LiteRTLM::Support::Tokenizer::Tokenizers::Utils
  LiteRTLM::Support::Tokenizer::Tokenizers::Unicode
  LiteRTLM::Support::Tokenizer::Tokenizers::Tokenizer
  LiteRTLM::Support::Tokenizer::Tokenizers::Models
  LiteRTLM::Support::Tokenizer::Tokenizers::Normalizers
  LiteRTLM::Support::Tokenizer::Tokenizers::PreTokenizers
  LiteRTLM::Support::Tokenizer::Tokenizers::Decoders
  LiteRTLM::Support::Tokenizer::Tokenizers::Processors
)
