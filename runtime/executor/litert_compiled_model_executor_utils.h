// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Prefill signature map for LiteRt APIs.
using SortedPrefillSignatureMap =
    absl::btree_map<int, std::string, std::greater<int>>;

// The data type of the attention mask.
// BOOLEAN: The attention mask is a boolean tensor.
// FLOAT: The attention mask is a float tensor.
enum class AttentionMaskDataType { BOOLEAN, FLOAT };

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
  // Input attention mask signature name for local attention. For both prefill
  // and decode. Not all models require this input.
  std::optional<std::string> input_attn_mask_local;
  // Input embeddings signature name. For both prefill and decode. When this
  // is provided, the embedding model will be used to look up the embeddings and
  // the input_tokens value must not be set.
  std::optional<std::string> input_embeddings;
  // Input per layer embeddings signature name. For both prefill and decode.
  // When this is provided, the per layer embedding model will be used to look
  // up the per layer embeddings.
  std::optional<std::string> input_per_layer_embeddings;
  // Input int32 param signature name. For both prefill and decode.
  std::optional<std::string> input_int32_param;
  // Output logits signature name. Necessary for decode.
  std::string output_logits;
};

// Get the corresponding ModelSignatures struct for the given model using
// the signature runner. Returns an error if the runner's signature does not
// match any of the predefined signature set.
// For now, we should use decode runner, since it contains all input and output
// signatures of the model.
// If strict is true, we will check that: `input_tokens` or `input_embeddings`
// is provided, `input_positions` is provided, and `output_logits` is provided.
absl::StatusOr<ModelSignatures> GetModelSignaturesFromInputOutputNames(
    const std::vector<absl::string_view>& input_names,
    const std::vector<absl::string_view>& output_names, bool strict = true);

// Returns the cache root names from the input names or output names.
// The cache root names are the names of the inputs that are used to store the
// KV cache. The root names are the names without the index suffix.
// For example, if the input names are ["kv_cache_k_0", "kv_cache_v_0"], then
// the k_root_name will be "kv_cache_k_" and the v_root_name will be
// "kv_cache_v_".
absl::Status GetKVCacheRootNames(std::vector<absl::string_view> input_names,
                                 std::vector<absl::string_view> output_names,
                                 std::string& k_root_name,
                                 std::string& v_root_name);

// Gets a set of prefill signature runners from the interpreter.
// The signature runners are sorted by the input tokens dimension.
// signature_name_base is the prefix of the prefill signature names, e.g.
// "prefill".
// input_tokens_name is the name of the input tokens signature, e.g. "token_ids"
// for Gemma2 JAX and "tokens" for Gemma2 PyTorch.
absl::StatusOr<SortedPrefillSignatureMap> GetPrefillRunnerSetFromModel(
    const ::litert::Model& model, absl::string_view signature_name_base,
    absl::string_view input_positions_name);

// Get a list of prefill work groups, each of which contains the signature
// runner and prefill length for a single prefill call.
// The work groups are calculated to maximize prefill performance.
//
// Arguments:
// - prefill_runner_set: Map of sequence length to prefill signature runner
//   name.
// - input_length: Total number of input tokens to prefill.
// - max_prefill_sequence_length: Optional remaining capacity of KV cache /
//   state
//   entries. If specified, prefill runners with sequence length >
//   max_prefill_sequence_length are filtered out, and work groups are
//   constrained so that the allocated prefill runners do not exceed remaining
//   capacity.
//
// Caveats:
// - If max_prefill_sequence_length filters out all available runners or if the
//   input length cannot be accommodated within the remaining state capacity, an
//   absl::InvalidArgumentError status is returned.
// - When max_prefill_sequence_length is provided, prefill runner selection
//   adapts to use smaller runners that fit in the remaining capacity instead of
//   larger ones.
//
// - use_greedy_chunking: If true, uses strictly greedy chunking by cascading
//   remainders to smaller buckets without upgrading to a larger bucket. On CPU,
//   this should be set to true because linear compute scaling favors cascading
//   to smaller runners to avoid padding waste.
//
// Output: A vector of std::pair<std::string, int>
//   std::string - the prefill runner signature to be used for the current
//   prefill call.
//   int - the prefill input length for the current prefill call.
absl::StatusOr<std::vector<std::pair<std::string, int>>>
GetOptimizedPrefillWorkGroups(
    const SortedPrefillSignatureMap& prefill_runner_set, int input_length,
    std::optional<int> max_prefill_sequence_length = std::nullopt,
    bool use_greedy_chunking = false);

// Initializes the attention mask tensor for prefill/decode.
// The mask is a 4D tensor with shape [batch=1, seq_len, 1, max_kv_len].
// is_f16 only applies to FLOAT mask data type.
absl::Status InitializeAttentionMask(::litert::TensorBuffer& mask, bool is_f16);

// Parameters extracted from ExecutorMetadata for attention mask generation.
struct AttentionMaskParams {
  proto::AttentionMaskType global_type = proto::ATTENTION_MASK_TYPE_CAUSAL;
  proto::AttentionMaskType local_type = proto::ATTENTION_MASK_TYPE_CAUSAL;
  std::optional<int> sliding_window_size = std::nullopt;
};

// Extracts attention mask parameters (global type, local type, sliding
// window size) from the optional ExecutorMetadata protobuf.
// If metadata is null or does not specify types, defaults to causal type.
AttentionMaskParams GetAttentionMaskParams(
    const proto::ExecutorMetadata* executor_metadata);

// The operational mode for filling the ring-buffer attention mask.
enum class RingBufferAttentionMaskMode {
  // Prefill mode: keys are split into past cache and fresh chunk.
  kPrefill = 0,
  // Single-step decode mode.
  kDecode,
  // Multi-step verifier mode.
  kVerify,
};

// Fills attention mask for a given range of timesteps.
// The mask is a 4D tensor with shape [batch=1, seq_len, 1, max_kv_len].
// mask - The attention mask tensor to be filled.
// start_timestep - The starting timestep to be filled at seq = 1.
// steps - The number of steps to fill (the number of sequences to be filled).
// attention_mask_type - The attention mask type.
// token_ids - The token ids of the full context. Required for
//             kVisionBidirectional attention mask type during prefill.
// sliding_window_size - The sliding window size.
// mode - Explicit operational mode for ring-buffer masks (prefill, decode, or
// verify).
absl::Status FillAttentionMask(
    ::litert::TensorBuffer& mask, int start_timestep, int steps,
    proto::AttentionMaskType attention_mask_type,
    std::optional<absl::Span<const int>> token_ids = std::nullopt,
    std::optional<int> sliding_window_size = std::nullopt,
    std::optional<RingBufferAttentionMaskMode> mode = std::nullopt);

// Fills the parameters used by single buffer cache update from
// start_index to start_index + update_length.
// Note that this parameter tensor is used by add_values_to_cache kernel and
// runtime_batched_matmul kernel.
absl::Status FillSingleBufferCacheParamTensor(
    ::litert::TensorBuffer& param_tensor, int start_index, int update_length);

// Builds the model resources from the model_path for compiled model only.
// Supports .task and .litertlm formats.
absl::StatusOr<std::unique_ptr<ModelResources>>
BuildLiteRtCompiledModelResources(const ModelAssets& model_assets,
                                  bool enable_file_backed_model_loading = false,
                                  bool enable_file_backed_for_aot_npu = false);

// Computes token embeddings using the given lookup managers.
absl::Status GenericComputeTokenEmbeddings(
    const TensorBuffer& input_tokens, absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings,
    EmbeddingLookupManager* embedding_lookup_manager,
    EmbeddingLookupManager* per_layer_embedding_lookup_manager);

// Set centralized CPU options (e.g. threads and default XNNPack flags).
absl::Status SetCpuOptions(litert::CpuOptions& cpu_options,
                           int num_threads = 4);

// Set centralized standard GPU options common across executors.
absl::Status SetCommonGpuOptions(
    const ExecutorSettingsBase& executor_settings,
    litert::GpuOptions& gpu_options,
    std::optional<ActivationDataType> fallback_activation_data_type =
        std::nullopt);

// Set the CPU weight cache options for XNNPACK.
// Args:
//   - weight_cache_file: An optional weight cache file path.
//   - scoped_cache_file: An optional ScopedFile pointer holding an open file
//     descriptor to the weight cache file.
//   - logging_prefix: A prefix string for logging information.
//   - cpu_options: The CpuOptions reference to apply the settings to.
absl::Status SetCpuCacheOptions(
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        weight_cache_file,
    absl::string_view logging_prefix, litert::CpuOptions& cpu_options);

// Set the GPU weight cache options for ML Drift.
// Args:
//   - weight_cache_file: An optional weight cache file path or file descriptor.
//   - program_cache_file: An optional ScopedFile pointer holding an open file
//     descriptor or file path to the program cache file.
//   - cache_key: A string that defines the unique cache identifier for the
//     model weight cache and program cache files.
//   - logging_prefix: A prefix string for logging information.
//   - cache_compiled_shaders_only: If true, only compiled shaders are cached.
//   - gpu_options: The GpuOptions reference to apply the settings to.
absl::Status SetGpuCacheOptions(
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        weight_cache_file,
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        program_cache_file,
    absl::string_view cache_key, absl::string_view logging_prefix,
    bool cache_compiled_shaders_only, litert::GpuOptions& gpu_options);

// A struct holding the GPU cache files and the cache key.
struct GpuModelCacheData {
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
      program_cache_file;
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
      weight_cache_file;
  std::string cache_key;
};

// Returns the program/weight cache files and the cache key for GPU execution,
// unless cache is disabled (e.g. settings.GetCacheDir() is ":nocache").
absl::StatusOr<GpuModelCacheData> GetGpuModelCacheData(
    const ExecutorSettingsBase& executor_settings,
    absl::string_view cache_name);

// Holds a model's external weights so they can be registered with a
// litert::Options instance before compilation. External weights live in a
// dedicated file section rather than being embedded in (and mmapped with) the
// model, so they are represented as a file handle plus the sections sliced from
// it.
struct ExternalWeightResources {
  // File that backs the external weight sections. Empty when the model has no
  // external weights, e.g. it uses embedded weights or its resource format does
  // not support external weight sections.
  std::optional<ScopedFile> scoped_file;
  // Maps each external buffer group name to its region (offset and length)
  // within scoped_file. Empty when scoped_file is empty.
  litert::Options::ScopedWeightSectionMap sections;
};

// Returns the external weights for model_type when present. An empty result
// means the model has embedded weights or its resource format does not support
// external weight sections.
absl::StatusOr<ExternalWeightResources> GetExternalWeightResources(
    ModelResources& resources, ModelType model_type);

// Adds model_type's external weight section to compilation_options when one is
// present.
absl::Status SetExternalWeightOptions(ModelResources& resources,
                                      ModelType model_type,
                                      litert::Options& compilation_options);

// Initializes the embedding lookup and per-layer embedding lookup managers from
// the given model resources and environment.
absl::Status InitializeEmbeddingLookups(
    ::litert::Environment& env, ModelResources& resources,
    std::unique_ptr<EmbeddingLookupManager>& embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager>& per_layer_embedding_lookup);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LITERT_COMPILED_MODEL_EXECUTOR_UTILS_H_
