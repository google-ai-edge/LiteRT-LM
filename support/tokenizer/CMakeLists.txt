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
# SentencePiece Tokenizers
# ==============================================================================
add_litertlm_library(support_tokenizer_sentencepiece_tokenizer STATIC
  sentencepiece_tokenizer.cc
  ${LITERTLM_TOKENIZERS_SRC_SENTENCEPIECE_DIR}/src/error.cc
)
add_library(LiteRTLM::Support::Tokenizer::SentencePieceTokenizer
  ALIAS support_tokenizer_sentencepiece_tokenizer)

target_include_directories(support_tokenizer_sentencepiece_tokenizer
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_sentencepiece_tokenizer
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# HuggingFace Tokenizers
# ==============================================================================
add_litertlm_library(support_tokenizer_huggingface_tokenizer STATIC
  huggingface_tokenizer.cc
)
add_library(LiteRTLM::Support::Tokenizer::HuggingFaceTokenizer
  ALIAS support_tokenizer_huggingface_tokenizer)

target_include_directories(support_tokenizer_huggingface_tokenizer
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_huggingface_tokenizer
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Buffered Streaming Detokenizer
# ==============================================================================
add_litertlm_library(support_tokenizer_buffered_streaming_detokenizer STATIC
  buffered_streaming_detokenizer.cc
)
add_library(LiteRTLM::Support::Tokenizer::BufferedStreamingDetokenizer
  ALIAS support_tokenizer_buffered_streaming_detokenizer)

target_include_directories(support_tokenizer_buffered_streaming_detokenizer
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(support_tokenizer_buffered_streaming_detokenizer
  PUBLIC
    LiteRTLM::Support::Util::MemoryMappedFile
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(support_tokenizer_libs INTERFACE)
add_library(LiteRTLM::Support::Tokenizer
  ALIAS support_tokenizer_libs)

target_link_libraries(support_tokenizer_libs INTERFACE
  LiteRTLM::Support::Tokenizer::SentencePieceTokenizer
  LiteRTLM::Support::Tokenizer::HuggingFaceTokenizer
  LiteRTLM::Support::Tokenizer::BufferedStreamingDetokenizer
)
