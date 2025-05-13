// A simple command line tool to run the litert LLM engine on NPU.

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"

ABSL_FLAG(std::string, gemma3_path, "", "Path to the Gemma3 model.");
ABSL_FLAG(std::string, embedder_path, "", "Path to the embedder model.");
ABSL_FLAG(std::string, auxiliary_path, "", "Path to the auxiliary model.");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer model.");
ABSL_FLAG(std::string, litert_dispatch_lib_path, "",
          "Path to the LiteRT dispatch library.");
ABSL_FLAG(std::string, prompt, "", "Prompt to run.");
ABSL_FLAG(bool, gemma_only_quantized, true,
          "If only Gemma3 is quantized. If false, all models except embedder "
          "are quantized (*except the embedder for now).");
ABSL_FLAG(int, num_runs, 1, "Number of times to run the benchmark.");

using odml::infra::LlmLiteRtNpuCompiledModelExecutor;

using litert::lm::ThreadOptions;
using litert::lm::ThreadPool;
using odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization::
    kAllQuantized;
using odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization::
    kTransformerStackOnlyQuantized;
using litert::lm::ThreadPool;
using litert::lm::ThreadOptions;

odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization
GetQuantizationSchema() {
  if (absl::GetFlag(FLAGS_gemma_only_quantized)) {
    return kTransformerStackOnlyQuantized;
  } else {
    return kAllQuantized;
  }
}

float GetToksPrefill(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  return ((latency_stats.prefill_num_tokens * 1000 * 1000) /
          (float)latency_stats.prefill_e2e_latency_us);
}

float GetToksDecode(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  return ((latency_stats.decode_num_tokens * 1000 * 1000) /
          (float)latency_stats.decode_e2e_latency_us);
}

void PrintLatencyStats(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  std::cout << "\n" << "====== PREFILL STATS ======";
  std::cout << "\n"
            << "Total prefill latency [us]: "
            << latency_stats.prefill_e2e_latency_us;
  std::cout << "\n"
            << "(e2e) Prefill num tokens: " << latency_stats.prefill_num_tokens;
  std::cout << "\n"
            << "(e2e) Prefill tokens per second: "
            << GetToksPrefill(latency_stats);
  std::cout << "\n"
            << "(TransformerStackOnly) Prefill tokens per second: "
            << ((latency_stats.prefill_num_tokens * 1000 * 1000) /
                (float)latency_stats.prefill_llm_inference_latency_us);

  std::cout << "\n"
            << "====== [Excluding (de)quantization and buffer copying] "
               "PREFILL STATS ======";
  std::cout << "\n"
            << "(*) Prefill latency [us]: "
            << (latency_stats.prefill_e2e_latency_us -
                latency_stats.prefill_quantization_latency_us);
  std::cout << "\n"
            << "(*) Prefill num tokens: " << latency_stats.prefill_num_tokens;
  std::cout << "\n"
            << "(*) Prefill tokens per second: "
            << ((latency_stats.prefill_num_tokens * 1000 * 1000) /
                (float)(latency_stats.prefill_e2e_latency_us -
                        latency_stats.prefill_quantization_latency_us));

  std::cout << "\n" << "------ Prefill breakdown ------";
  std::cout << "\n"
            << "Total prefill prepare input tensors latency [us]: "
            << latency_stats.prefill_prepare_input_latency_us << " ("
            << ((latency_stats.prefill_prepare_input_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill embedder inference latency [us]: "
            << latency_stats.prefill_embedder_inference_latency_us << " ("
            << ((latency_stats.prefill_embedder_inference_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill rope inference latency [us]: "
            << latency_stats.prefill_rope_inference_latency_us << " ("
            << ((latency_stats.prefill_rope_inference_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill mask inference latency [us]: "
            << latency_stats.prefill_mask_inference_latency_us << " ("
            << ((latency_stats.prefill_mask_inference_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill (de)quantization and copy buffer latency [us]: "
            << latency_stats.prefill_quantization_latency_us << " ("
            << ((latency_stats.prefill_quantization_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill LLM inference latency [us]: "
            << latency_stats.prefill_llm_inference_latency_us << " ("
            << ((latency_stats.prefill_llm_inference_latency_us * 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total prefill cache update inference latency [us]: "
            << latency_stats.prefill_cache_update_inference_latency_us << " ("
            << ((latency_stats.prefill_cache_update_inference_latency_us *
                 100) /
                (float)latency_stats.prefill_e2e_latency_us)
            << "%)";

  std::cout << "\n" << "\n====== DECODE STATS ======";
  std::cout << "\n"
            << "Total decode latency [us]: "
            << latency_stats.decode_e2e_latency_us;
  std::cout << "\n"
            << "Decode num tokens: " << latency_stats.decode_num_tokens;
  std::cout << "\n"
            << "Decode tokens per second: " << GetToksDecode(latency_stats);
  std::cout << "\n"
            << "(TransformerStackOnly) Decode tokens per second: "
            << ((latency_stats.decode_num_tokens * 1000 * 1000) /
                (float)latency_stats.decode_llm_inference_latency_us);

  std::cout << "\n"
            << "====== [Excluding (de)quantization and buffer copying] "
               "DECODE STATS ======";
  std::cout << "\n"
            << "(*) Decode latency [us]: "
            << (latency_stats.decode_e2e_latency_us -
                latency_stats.decode_quantization_latency_us);
  std::cout << "\n"
            << "(*) Decode num tokens: " << latency_stats.decode_num_tokens;
  std::cout << "\n"
            << "(*) Decode tokens per second: "
            << ((latency_stats.decode_num_tokens * 1000 * 1000) /
                (float)(latency_stats.decode_e2e_latency_us -
                        latency_stats.decode_quantization_latency_us));

  std::cout << "\n" << "------ Decode breakdown ------";
  std::cout << "\n"
            << "Total decode prepare input tensors latency [us]: "
            << latency_stats.decode_prepare_input_latency_us << " ("
            << ((latency_stats.decode_prepare_input_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode embedder inference latency [us]: "
            << latency_stats.decode_embedder_inference_latency_us << " ("
            << ((latency_stats.decode_embedder_inference_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode rope inference latency [us]: "
            << latency_stats.decode_rope_inference_latency_us << " ("
            << ((latency_stats.decode_rope_inference_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode mask inference latency [us]: "
            << latency_stats.decode_mask_inference_latency_us << " ("
            << ((latency_stats.decode_mask_inference_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode (de)quantization and copy buffer latency [us]: "
            << latency_stats.decode_quantization_latency_us << " ("
            << ((latency_stats.decode_quantization_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode LLM inference latency [us]: "
            << latency_stats.decode_llm_inference_latency_us << " ("
            << ((latency_stats.decode_llm_inference_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode cache update inference latency [us]: "
            << latency_stats.decode_cache_update_inference_latency_us << " ("
            << ((latency_stats.decode_cache_update_inference_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)";
  std::cout << "\n"
            << "Total decode sampling latency [us]: "
            << latency_stats.decode_sampling_latency_us << " ("
            << ((latency_stats.decode_sampling_latency_us * 100) /
                (float)latency_stats.decode_e2e_latency_us)
            << "%)\n";
}

struct RunStats {
  int64_t executor_creation_latency_us = 0;
  float prefill_toks = 0;
  float decode_toks = 0;
};

RunStats CreateAndRun(const std::string& prompt) {
  // Create the tokenizer.
  auto tokenizer_or = litert::lm::SentencePieceTokenizer::CreateFromFile(
      absl::GetFlag(FLAGS_tokenizer_path));
  if (tokenizer_or.ok()) {
    ABSL_LOG(INFO) << "tokenizer created successfully";
  } else {
    ABSL_LOG(ERROR) << "tokenizer creation failed: " << tokenizer_or.status();
  }
  std::shared_ptr<litert::lm::Tokenizer> tokenizer =
      std::move(tokenizer_or.value());

  // Create the executor.
  auto start = absl::Now();
  ABSL_LOG(INFO) << "Creating executor";
  auto executor_or = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
      GetQuantizationSchema(), absl::GetFlag(FLAGS_gemma3_path),
      absl::GetFlag(FLAGS_embedder_path), absl::GetFlag(FLAGS_auxiliary_path),
      absl::GetFlag(FLAGS_litert_dispatch_lib_path));
  auto end = absl::Now();

  int64_t executor_creation_latency_us = absl::ToInt64Microseconds(end - start);
  ABSL_LOG(INFO) << "executor creation took " << executor_creation_latency_us
                 << " us";
  if (executor_or.ok()) {
    ABSL_LOG(INFO) << "executor created successfully";
  } else {
    ABSL_LOG(ERROR) << "executor creation failed: " << executor_or.status();
  }
  std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor =
      std::move(executor_or.value());
  std::shared_ptr<LlmLiteRtNpuCompiledModelExecutor> executor_shared =
      std::move(executor);

  auto worker_thread_pool_or =
      ThreadPool::CreateThreadPool(ThreadOptions(),
                                   /*name_prefix=*/"engine",
                                   /*num_threads=*/1);
  std::shared_ptr<ThreadPool> worker_thread_pool =
      std::move(*worker_thread_pool_or);
  worker_thread_pool->StartWorkers();

  // Create the session.
  constexpr int kEndOfTurnTokenId = 106;
  std::vector<int> stop_token_ids = {kEndOfTurnTokenId};
  // TODO(b/405424188): The NPU executor currently uses direct sampling on the
  // int16 logits. Use SamplerParameters::TYPE_UNSPECIFIED to avoid using the
  // default float CPU sampler.
  // See the description of cl/752681117 that justifies the usage of a
  // custom sampler (decode performance speed-up). We should extend the
  // 'litert::lm' Sampler to support this natively and remove the custom sampler
  // of the NPU executor.
  // TODO(b/415915773): update the session config to use the provided sampler
  // once it supports the int16 logits.
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams().set_type(
      litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED);
  auto session = litert::lm::SessionBasic::Create(
      executor_shared, tokenizer, stop_token_ids, session_config, std::nullopt,
      worker_thread_pool);

  // Run the session.
  ABSL_LOG(INFO) << "Prompt: " << prompt;
  start = absl::Now();
  auto status = (*session)->RunPrefill(prompt);
  end = absl::Now();
  ABSL_LOG(INFO) << "RunPrefill took " << absl::ToInt64Microseconds(end - start)
                 << " us";

  start = absl::Now();
  auto responses = (*session)->RunDecode();
  end = absl::Now();

  ABSL_LOG(INFO) << "RunDecode took " << absl::ToInt64Microseconds(end - start)
                 << " us";
  if (responses.ok()) {
    for (int i = 0; i < responses->GetNumOutputCandidates(); ++i) {
      auto response_text = responses->GetResponseTextAt(i);
      ABSL_LOG(INFO) << "Generated response: " << (*response_text);
    }
  } else {
    ABSL_LOG(ERROR) << "response failed: " << responses.status();
  }

  LlmLiteRtNpuCompiledModelExecutor::LatencyStats latency_stats =
      executor_shared->GetLatencyStats();
  PrintLatencyStats(latency_stats);

  return RunStats{.executor_creation_latency_us = executor_creation_latency_us,
                  .prefill_toks = GetToksPrefill(latency_stats),
                  .decode_toks = GetToksDecode(latency_stats)};
}

void PrintStats(std::vector<float>& values, const std::string& stat_name) {
  if (values.empty()) {
    std::cout << stat_name << " is empty.\n";
    return;
  }
  float sum = std::accumulate(values.begin(), values.end(), 0.0);
  float average = sum / values.size();
  std::sort(values.begin(), values.end());
  float median = values[values.size() / 2];
  float min = values.front();
  float max = values.back();
  std::cout << "===== " << stat_name << " =====\n";
  std::cout << "Average: " << average << "\n";
  std::cout << "Median: " << median << "\n";
  std::cout << "Min: " << min << "\n";
  std::cout << "Max: " << max << "\n";
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  const int num_runs = absl::GetFlag(FLAGS_num_runs);

  std::vector<float> executor_creation_latency_us_values;
  std::vector<float> prefill_toks_values;
  std::vector<float> decode_toks_values;
  for (int i = 0; i < num_runs; ++i) {
    auto stats = CreateAndRun(prompt);
    executor_creation_latency_us_values.push_back(
        stats.executor_creation_latency_us);
    prefill_toks_values.push_back(stats.prefill_toks);
    decode_toks_values.push_back(stats.decode_toks);
  }

  PrintStats(executor_creation_latency_us_values,
             "Executor Creation Latency (us)");
  PrintStats(prefill_toks_values, "Prefill Toks");
  PrintStats(decode_toks_values, "Decode Toks");

  return 0;
}
