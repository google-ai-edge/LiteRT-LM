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

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "c/engine.h"

#ifdef _WIN32
std::pair<double, double> GetMemoryUsage() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return {pmc.WorkingSetSize / (1024.0 * 1024.0),
            pmc.PagefileUsage / (1024.0 * 1024.0)};
  }
  return {0.0, 0.0};
}
#else
long GetCurrentVmSwapKb() {
#ifdef __linux__
  std::ifstream f("/proc/self/status");
  if (!f.is_open()) return 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmSwap:", 0) == 0) {
      std::string label;
      long value;
      std::string unit;
      std::stringstream ss(line);
      if (ss >> label >> value >> unit) return value;
    }
  }
#endif
  return 0;
}

std::pair<double, double> GetMemoryUsage() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return {0.0, 0.0};
  double maxrss = usage.ru_maxrss;
#ifdef __APPLE__
  maxrss /= (1024.0 * 1024.0);
#else
  maxrss /= 1024.0;
#endif
  double swap_mb = GetCurrentVmSwapKb() / 1024.0;
  return {maxrss, maxrss + swap_mb};
}
#endif

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string backend = "cpu";
  int prefill_tokens = 256, decode_tokens = 256, max_num_tokens = -1;
  std::string cache_dir, spec_dec = "auto";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--model_path=", 0) == 0) {
      model_path = arg.substr(13);
    } else if (arg.rfind("--backend=", 0) == 0) {
      backend = arg.substr(10);
    } else if (arg.rfind("--prefill_tokens=", 0) == 0) {
      prefill_tokens = std::stoi(arg.substr(17));
    } else if (arg.rfind("--decode_tokens=", 0) == 0) {
      decode_tokens = std::stoi(arg.substr(16));
    } else if (arg.rfind("--max_num_tokens=", 0) == 0) {
      max_num_tokens = std::stoi(arg.substr(17));
    } else if (arg.rfind("--cache_dir=", 0) == 0) {
      cache_dir = arg.substr(12);
    } else if (arg.rfind("--speculative_decoding=", 0) == 0) {
      spec_dec = arg.substr(23);
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model_path is required." << std::endl;
    return 1;
  }

  LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(
      model_path.c_str(), backend.c_str(), nullptr, nullptr);
  if (!settings) return 1;

  litert_lm_engine_settings_enable_benchmark(settings);
  if (max_num_tokens > 0) {
    litert_lm_engine_settings_set_max_num_tokens(settings, max_num_tokens);
  }
  litert_lm_engine_settings_set_num_prefill_tokens(settings, prefill_tokens);
  litert_lm_engine_settings_set_num_decode_tokens(settings, decode_tokens);
  if (!cache_dir.empty()) {
    litert_lm_engine_settings_set_cache_dir(settings, cache_dir.c_str());
  }

  if (spec_dec == "true") {
    litert_lm_engine_settings_set_enable_speculative_decoding(settings, true);
  } else if (spec_dec == "false") {
    litert_lm_engine_settings_set_enable_speculative_decoding(settings, false);
  }

  LiteRtLmEngine* engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine) return 1;

  LiteRtLmSession* session = litert_lm_engine_create_session(engine, nullptr);
  if (!session) {
    litert_lm_engine_delete(engine);
    return 1;
  }

  const char* dummy_prompt = "benchmark";
  LiteRtLmInputData input_data = {kLiteRtLmInputDataTypeText, dummy_prompt,
                                  strlen(dummy_prompt)};
  LiteRtLmResponses* responses =
      litert_lm_session_generate_content(session, &input_data, 1);
  if (responses) litert_lm_responses_delete(responses);

  auto [peak_mem, peak_private] = GetMemoryUsage();

  LiteRtLmBenchmarkInfo* info = litert_lm_session_get_benchmark_info(session);
  if (!info) {
    litert_lm_session_delete(session);
    litert_lm_engine_delete(engine);
    return 1;
  }

  std::cout << "{"
            << "\"init_time_in_second\":"
            << litert_lm_benchmark_info_get_total_init_time_in_second(info)
            << ","
            << "\"time_to_first_token_in_second\":"
            << litert_lm_benchmark_info_get_time_to_first_token(info) << ","
            << "\"last_prefill_token_count\":"
            << litert_lm_benchmark_info_get_prefill_token_count_at(info, 0)
            << ","
            << "\"last_prefill_tokens_per_second\":"
            << litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, 0)
            << ","
            << "\"last_decode_token_count\":"
            << litert_lm_benchmark_info_get_decode_token_count_at(info, 0)
            << ","
            << "\"last_decode_tokens_per_second\":"
            << litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, 0)
            << ","
            << "\"peak_mem_mb\":" << peak_mem << ","
            << "\"peak_private_mb\":" << peak_private << "}" << std::endl;

  litert_lm_benchmark_info_delete(info);
  litert_lm_session_delete(session);
  litert_lm_engine_delete(engine);
  return 0;
}
