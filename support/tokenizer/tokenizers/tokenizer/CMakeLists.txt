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

add_litertlm_library(support_tokenizer_tokenizers_tokenizer STATIC
  added_vocabulary.cc
  aho_corasick.cc
  api.cc
  encoding.cc
  normalizer.cc
  padding.cc
  pattern.cc
  pre_tokenizer.cc
  serialization.cc
  tokenizer.cc
  truncation.cc
)
add_library(LiteRTLM::Support::Tokenizer::Tokenizers::Tokenizer
  ALIAS support_tokenizer_tokenizers_tokenizer)

target_include_directories(support_tokenizer_tokenizers_tokenizer
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_tokenizers_tokenizer
  PUBLIC
    LITERTLM_DEPS
)
