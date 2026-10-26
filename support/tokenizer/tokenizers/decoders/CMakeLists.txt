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

add_litertlm_library(support_tokenizer_tokenizers_decoders STATIC
  bpe.cc
  byte_fallback.cc
  ctc.cc
  decoders.cc
  fuse.cc
  sequence.cc
  strip.cc
  wordpiece.cc
)
add_library(LiteRTLM::Support::Tokenizer::Tokenizers::Decoders
  ALIAS support_tokenizer_tokenizers_decoders)

target_include_directories(support_tokenizer_tokenizers_decoders
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_tokenizers_decoders
  PUBLIC
    LITERTLM_DEPS
)
