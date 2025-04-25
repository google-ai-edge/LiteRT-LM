#include "third_party/odml/litert_lm/runtime/core/session_factory.h"

#include <memory>
#include <vector>

#include "third_party/absl/status/statusor.h"
#include "third_party/odml/litert_lm/runtime/components/tokenizer.h"
#include "third_party/odml/litert_lm/runtime/core/session_basic.h"
#include "third_party/odml/litert_lm/runtime/engine/engine.h"
#include "third_party/odml/litert_lm/runtime/executor/llm_executor.h"
#include "third_party/odml/litert_lm/runtime/proto/sampler_params.proto.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Engine::Session>> InitializeSession(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids,
    const proto::SamplerParameters& sampler_params) {
  return SessionBasic::Create(executor, tokenizer, stop_token_ids,
                              sampler_params);
}

}  // namespace litert::lm
