#ifndef THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/model_asset_bundle_resources.h"

namespace litert::lm {

// Prefill signature map for LiteRt APIs.
using SortedPrefillSignatureMap =
    absl::btree_map<int, std::string, std::greater<int>>;

// The data type of the attention mask.
// BOOLEAN: The attention mask is a boolean tensor.
// FLOAT: The attention mask is a float tensor.
enum class AttentionMaskDataType { BOOLEAN, FLOAT };

// All the loaded model resources the executor needs to hold to avoid the model
// being destroyed.
struct ExecutorModelResources {
  // The litert model, used by the litert APIs.
  ::litert::Model litert_model;
  // The model asset bundle resources produced by reading task bundle. Not null
  // only when the model is provided through .task format. If the model is
  // retrieved from this resource, releasing this resource will also invalidate
  // the model.
  std::unique_ptr<litert::lm::ModelAssetBundleResources>
      model_asset_bundle_resources;
};

// A struct holding a set of model signatures used for doing inference on a
// conversion path Gemini/Gemma model.
// For now, this struct supports Gemini V1.5 and Gemma2 only.
// TODO: b/375276056 - Support Gemini V2 signatures.
struct ModelSignatures {
  // Input token signature name. For both prefill and decode.
  std::string input_tokens;
  // Input position signature name. For both prefill and decode.
  std::string input_positions;
  // Input attention mask signature name. For both prefill and decode.
  // Not all models require this input.
  std::optional<std::string> input_attn_mask;
  // The data type of the attention mask.
  std::optional<AttentionMaskDataType> input_attn_mask_data_type;
  // Output logits signature name. Necessary for decode.
  std::string output_logits;
};

// Get the corresponding ModelSignatures struct for the given model using
// the signature runner. Returns an error if the the runner's signature does not
// match any of the predefined signature set.
// For now, we should use decode runner, since it contains all input and output
// signatures of the model.
absl::StatusOr<ModelSignatures> GetModelSignaturesFromInputOutputNames(
    std::vector<absl::string_view> input_names,
    std::vector<absl::string_view> output_names);

// Gets a set of prefill signature runners from the interpreter.
// The signature runners are sorted by the input tokens dimension.
// signature_name_base is the prefix of the prefill signature names, e.g.
// "prefill".
// input_tokens_name is the name of the input tokens signature, e.g. "token_ids"
// for Gemma2 JAX and "tokens" for Gemma2 PyTorch.
absl::StatusOr<SortedPrefillSignatureMap> GetPrefillRunnerSetFromModel(
    ::litert::Model& model, const std::string& signature_name_base,
    const std::string& input_tokens_name);

// Get a list of prefill work groups, each of which contains the signature
// runner and prefill length for a single prefill call.
// The work groups are calculated to maximize prefill performance.
// Output: A vector of std::pair<SignatureRunner*, int>
// SignatureRunner* - the prefill runner to be used for current prefill call.
// int - the prefill length for current prefill call.
absl::StatusOr<std::vector<std::pair<std::string, int>>>
GetOptimizedPrefillWorkGroups(SortedPrefillSignatureMap prefill_runner_set,
                              int input_length);

// Initializes the attention mask tensor for prefill/decode.
// The mask is a 4D tensor with shape [batch=1, seq_len, 1, max_kv_len].
// The default value for mask is different for different mask data types, and
// different calculation precisions.
absl::Status InitializeAttentionMask(::litert::TensorBuffer& mask,
                                     AttentionMaskDataType mask_data_type,
                                     bool is_f16);

// Fill attention mask for a given range of timesteps.
// The mask is a 4D tensor with shape [batch=1, seq_len, 1, max_kv_len].
// mask - The attention mask tensor to be filled.
// start_timestep - The starting timestep to be filled at seq = 1.
// steps - The number of steps to fill (the number of sequences to be filled).
// mask_data_type - The data type of the attention mask (e.g. boolean, float).
absl::Status FillAttentionMask(::litert::TensorBuffer& mask, int start_timestep,
                               int steps, AttentionMaskDataType mask_data_type);

// Builds the model resources from the model_path for compiled model only.
// Supports .tflite and .task format.
absl::StatusOr<std::unique_ptr<ExecutorModelResources>>
BuildLiteRtCompiledModelResources(const std::string& model_path);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
