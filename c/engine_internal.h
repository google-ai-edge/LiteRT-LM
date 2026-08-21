// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "c/engine.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/token.pb.h"

struct LiteRtLmInputData {
  explicit LiteRtLmInputData(litert::lm::InputData d) : data(std::move(d)) {}
  litert::lm::InputData data;
};

struct LiteRtLmSamplerParams {
  LiteRtLmSamplerType type;
  int32_t top_k;
  float top_p;
  float temperature;
  int32_t seed;
};

struct LiteRtLmStreamChunk {
  const char* text = nullptr;
  bool is_final = false;
  const char* error_msg = nullptr;
};

struct LiteRtLmRepetitionPenaltyConfig {
  litert::lm::RepetitionPenaltyConfig repetition_penalty_config;
};

struct LiteRtLmNoRepeatNgramConfig {
  litert::lm::NoRepeatNgramConfig no_repeat_ngram_config;
};

struct LiteRtLmSuppressTokensConfig {
  litert::lm::SuppressTokensConfig suppress_tokens_config;
};

struct LiteRtLmEngineSettings {
  std::unique_ptr<litert::lm::EngineSettings> settings;
};

struct LiteRtLmEngine {
  std::unique_ptr<litert::lm::Engine> engine;
};

struct LiteRtLmSession {
  std::unique_ptr<litert::lm::Engine::Session> session;
};

struct LiteRtLmResponses {
  litert::lm::Responses responses;
};

struct LiteRtLmBenchmarkInfo {
  litert::lm::BenchmarkInfo benchmark_info;
};

// TODO: b/483172229 - Migrate to use SessionConfig instead of unique_ptr to
// SessionConfig for consistency and efficiency.
struct LiteRtLmSessionConfig {
  std::unique_ptr<litert::lm::SessionConfig> config;
};

struct LiteRtLmDetokenizeResult {
  std::string text;
};

struct LiteRtLmTokenizeResult {
  std::vector<int> tokens;
};

struct LiteRtLmTokenUnion {
  litert::lm::proto::TokenUnion token_union;
};

struct LiteRtLmTokenUnions {
  std::vector<litert::lm::proto::TokenUnion> tokens;
};

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_
