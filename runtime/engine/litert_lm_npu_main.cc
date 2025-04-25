// A simple command line tool to run the litert LLM engine on NPU.

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/flags/parse.h"  // from @abseil-cpp
#include "absl/log/absl_log.h"  // from @abseil-cpp
#include "third_party/odml/infra/genai/inference/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/proto/sampler_params.proto.h"

constexpr char kModelPathLlm[] = "gemma3_npu_f32_ekv1280.tflite";
constexpr char kModelPathTokenizer[] = "gemma3_tokenizer.spiece";
constexpr char kModelPathEmbedder[] = "gemma3_npu_embedder.tflite";
constexpr char kModelPathAuxiliary[] = "gemma3_npu_auxiliary.tflite";

ABSL_FLAG(std::string, binary_path, "", "Path to the binary.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Construct the paths to the models.
  auto binary_path = std::filesystem::path(absl::GetFlag(FLAGS_binary_path));
  std::string model_path = (binary_path / kModelPathLlm).string();
  std::string tokenizer_path = (binary_path / kModelPathTokenizer).string();
  std::string embedder_path = (binary_path / kModelPathEmbedder).string();
  std::string auxiliary_path = (binary_path / kModelPathAuxiliary).string();

  // Create the tokenizer.
  auto tokenizer_or =
      litert::lm::SentencePieceTokenizer::CreateFromFile(tokenizer_path);
  if (tokenizer_or.ok()) {
    ABSL_LOG(INFO) << "tokenizer created successfully";
  } else {
    ABSL_LOG(ERROR) << "tokenizer creation failed: " << tokenizer_or.status();
  }
  std::shared_ptr<litert::lm::Tokenizer> tokenizer =
      std::move(tokenizer_or.value());

  // Create the executor.
  auto executor_or = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
      model_path, embedder_path, auxiliary_path);
  if (executor_or.ok()) {
    ABSL_LOG(INFO) << "executor created successfully";
  } else {
    ABSL_LOG(ERROR) << "executor creation failed: " << executor_or.status();
  }
  std::unique_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor> executor =
      std::move(executor_or.value());
  std::shared_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor>
      executor_shared = std::move(executor);

  // Create the session.
  std::vector<int> stop_token_ids = {1};
  auto session = litert::lm::SessionBasic::Create(
      executor_shared, tokenizer, stop_token_ids,
      litert::lm::proto::SamplerParameters());

  // Run the session.
  auto status = (*session)->AddTextPrompt(
      "Write a poem about the greatness of the gemma LLM");
  auto responses = (*session)->PredictSync();
  if (responses.ok()) {
    for (int i = 0; i < responses->GetNumOutputCandidates(); ++i) {
      auto response_text = responses->GetResponseTextAt(i);
      ABSL_LOG(INFO) << "Generated response: " << (*response_text);
    }
  } else {
    ABSL_LOG(ERROR) << "response failed: " << responses.status();
  }

  return 0;
}
