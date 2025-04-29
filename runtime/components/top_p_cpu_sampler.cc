#include "runtime/components/top_p_cpu_sampler.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

absl::Status ValidateTensor(const TensorBuffer& tensor, int max_num_dims,
                            int batch_size, const std::string& tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  if (NumSignificantDims(tensor) > max_num_dims) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output ", tensor_name, " tensor must have <=", max_num_dims,
        " significant dimension, but got ", NumSignificantDims(tensor)));
  }
  if (dims[0] != batch_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The output ", tensor_name,
                     " tensor must have the same batch size as the input "
                     "logits tensor, but got ",
                     dims[0], " vs ", batch_size));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<TopPSampler>> TopPSampler::Create(
    int k, float p, float temperature, int batch_size, int seed) {
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (p < 0.0f || p > 1.0f) {
    return absl::InvalidArgumentError("p must be in [0, 1].");
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (temperature <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be positive, but got ", temperature));
  }
  return absl::WrapUnique(new TopPSampler(k, p, temperature, batch_size, seed));
}

absl::Status TopPSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          logits_tensor_type.Layout().NumElements());
  const int vocab_size = num_elements / batch_size_;
  auto status = ValidateTensor(logits_tensor, /*max_num_dims=*/2, batch_size_,
                               "input logits");
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateTensor(ids_tensor, /*max_num_dims=*/1, batch_size_, "output ids");
  if (!status.ok()) {
    return status;
  }

  auto logits_data = CopyFromTensorBuffer<float>(logits_tensor);
  auto probabilities =
      Softmax(absl::MakeConstSpan((*logits_data)), temperature_, batch_size_);
  if (!probabilities.ok()) {
    return probabilities.status();
  }
  auto sampled_ids = TopKTopPSampling(absl::MakeConstSpan(*probabilities), k_,
                                      p_, *generator_, batch_size_);
  if (!sampled_ids.ok()) {
    return sampled_ids.status();
  }
  ids_tensor.Write(absl::MakeConstSpan(*sampled_ids));
  if (scores_tensor != nullptr) {
    status = ValidateTensor(*scores_tensor, /*max_num_dims=*/1, batch_size_,
                            "output scores");
    if (!status.ok()) {
      return status;
    }
    std::vector<float> scores(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
      // The scores are the log of the probability of the sampled token.
      scores[i] =
          std::log((*probabilities)[i * vocab_size + (*sampled_ids)[i]]);
    }
    scores_tensor->Write(absl::MakeConstSpan(scores));
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
