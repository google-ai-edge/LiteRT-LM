#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_

#include <memory>
#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"

namespace litert::lm {

class TopPSampler : public Sampler {
 public:
  // Creates a TopPSampler with the given input arguments:
  // - k: The number of top logits to consider.
  // - p: The top-p probability mass to consider.
  // - batch_size: The batch size of the input logits.
  // - seed: The seed for the random number generator.
  static absl::StatusOr<std::unique_ptr<TopPSampler>> Create(int k, float p,
                                                             float temperature,
                                                             int batch_size,
                                                             int seed);

  // Given a batch of logits, samples a batch of token ids.
  // The expected shape of the logits is [batch_size, vocab_size].
  // The output ids_tensor is a 1D litert::TensorBuffer of shape [batch_size].
  // The output scores_tensor is optional. If it is not nullptr, the sampled
  // scores are also written to it (in the same shape as the ids_tensor). The
  // scores are the log of the probability of the sampled token.
  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override;

 private:
  explicit TopPSampler(int k, float p, float temperature, int batch_size,
                       int seed)
      : k_(k),
        p_(p),
        temperature_(temperature),
        batch_size_(batch_size),
        generator_(std::make_unique<std::mt19937>(seed)) {}

  // The parameters for the sampler.
  const int k_;
  const float p_;
  const float temperature_;
  const int batch_size_;
  std::unique_ptr<std::mt19937> generator_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_
