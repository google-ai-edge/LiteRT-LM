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

add_litertlm_library(support_tokenizer_tokenizers_pre_tokenizers STATIC
  bert.cc
  byte_level.cc
  delimiter.cc
  digits.cc
  fixed_length.cc
  metaspace.cc
  pre_tokenizers.cc
  punctuation.cc
  sequence.cc
  split.cc
  unicode_scripts/pre_tokenizer.cc
  unicode_scripts/scripts.cc
  unicode_scripts/unicode_scripts.cc
  whitespace.cc
)
add_library(LiteRTLM::Support::Tokenizer::Tokenizers::PreTokenizers
  ALIAS support_tokenizer_tokenizers_pre_tokenizers)

target_include_directories(support_tokenizer_tokenizers_pre_tokenizers
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_tokenizers_pre_tokenizers
  PUBLIC
    LITERTLM_DEPS
)
