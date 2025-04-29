#include "runtime/core/session_basic.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/core/pipeline.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// Default batch size for the output. This should be configurable in the
// future.
constexpr int kOutputBatchSize = 1;

}  // namespace

absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids,
    const proto::SamplerParameters& sampler_params) {
  std::unique_ptr<Sampler> sampler;
  // TODO(b/407086356): Add test or factor out the logic to create the sampler.
  switch (sampler_params.type()) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED:
      ABSL_LOG(INFO) << "Sampler type is unspecified. Assume the LLM Executor "
                        "handles the sampling logic.";
      break;
    case proto::SamplerParameters::TOP_P: {
      absl::StatusOr<std::unique_ptr<Sampler>> sampler_or;
      sampler_or = TopPSampler::Create(
          sampler_params.k(), sampler_params.p(), sampler_params.temperature(),
          /*batch_size=*/kOutputBatchSize, sampler_params.seed());
      if (!sampler_or.ok()) {
        return sampler_or.status();
      }
      sampler = std::move(*sampler_or);
    } break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Sampler type: ", sampler_params.type(), " not implemented yet."));
  }
  return absl::WrapUnique(new SessionBasic(executor, tokenizer, stop_token_ids,
                                           std::move(sampler)));
}

absl::Status SessionBasic::PrefillInternal(absl::string_view input,
                                           bool wait_for_completion) {
  // TODO(b/397975034): factor out the prompt formatting logic into a
  // separate library/class.
  const std::string prompt = absl::StrCat(
      "<start_of_turn>user\n", input, "<end_of_turn>\n<start_of_turn>model\n");
  ABSL_LOG(INFO) << "RunPrefillSync: " << prompt;
  ASSIGN_OR_RETURN(last_prefill_token_id_,
                   Prefill(executor_, tokenizer_, prompt, /*bos_token_id=*/2,
                           wait_for_completion));
  ABSL_LOG(INFO) << "Prefill done";
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(absl::string_view input) {
  return PrefillInternal(input, /*wait_for_completion=*/true);
}

absl::Status SessionBasic::RunPrefillAsync(absl::string_view input) {
  return PrefillInternal(input, /*wait_for_completion=*/false);
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  ABSL_LOG(INFO) << "RunDecodeSync";
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(auto responses,
                     Decode(executor_, tokenizer_, stop_token_ids_));
    return responses;
  } else {
    std::vector<int> decoded_ids(kOutputBatchSize, last_prefill_token_id_);
    auto decoded_ids_buffer =
        CopyToTensorBuffer<int>(decoded_ids, {kOutputBatchSize, 1});
    ASSIGN_OR_RETURN(auto responses, DecodeCustomSampling(
                                         executor_, tokenizer_, stop_token_ids_,
                                         /*num_output_candidates=*/1, *sampler_,
                                         *decoded_ids_buffer));
    return responses;
  }
}

}  // namespace litert::lm
