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

// Automated benchmark driver for the LiteRT-LM embedding engine.
//
// Unlike embedding_litert_lm_main, which runs a single user-specified request,
// this binary discovers every signature the bundle actually contains and
// sweeps all of them:
//
//   1) Text:   one scenario per text encoder signature (128, 256, 512, ...),
//              capped by --max_benchmark_text_tokens, plus an overflow
//              scenario that exercises chunk_and_average.
//   2) Vision: one scenario per vision encoder signature (70, 140, 280, 560,
//              ...), capped by --max_benchmark_vision_tokens.
//   3) Audio:  a single fixed-duration clip.
//
// Each of the above can additionally be swept across GPU compute precisions
// via --gpu_precisions=fp16,fp32, which cross-products the scenario list: a
// scenario named "text_512" becomes "text_512@fp16" and "text_512@fp32".
//
// Design notes:
//
// * Every scenario gets its OWN engine, created from scratch and torn down
//   afterwards. Scenarios therefore never share compiled graphs or weight
//   caches: each one pays for, builds, and then re-uses its own cache during
//   its dedicated warmup iterations. This costs a model reload per scenario
//   but makes the per-scenario numbers independent and comparable. This
//   applies to precision variants too, so fp16 and fp32 never share a
//   compiled graph.
//
// * Vision and audio inputs are SYNTHETIC by default, and no data files are
//   needed to run a sweep.
//
//   For vision this is not merely a convenience. The engine can only
//   patchify a raw image when it can read the patch geometry out of model
//   metadata, which is unavailable in some builds. Rather than guess that
//   geometry, the synthetic path asks the vision encoder signature for the
//   shape of its own input tensors and fills tensors of exactly that shape.
//   That makes the benchmark portable without assuming anything about the
//   bundle. Latency is unaffected because these encoders are fixed-shape
//   dense graphs whose cost depends on tensor shapes, not tensor values.
//
//   Passing --image_path switches to a real image, which additionally
//   requires --vision_patch_width/height/pooling_kernel_size because the
//   geometry then genuinely cannot be inferred. Passing --audio_path switches
//   to a real clip, whose bytes go straight to the engine's own audio
//   preprocessor. Use the real-input paths when the embedding values matter;
//   the synthetic ones produce meaningless embeddings by construction.
//
// * Backends are selected independently per modality and may be mixed. A
//   --gpu_precisions variant only retypes the executors that are actually on
//   GPU, so in a mixed configuration the CPU and NPU arms stay identical
//   across the sweep and the comparison isolates one variable.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/model_signature_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/tokenizer/tokenizer.h"
#include "tflite/profiling/memory_info.h"  // from @litert

// ---------------------------------------------------------------------------
// Model / IO
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, model_path, "", "Path to the embedding .litertlm file.");
ABSL_FLAG(std::string, image_path, "",
          "Real image for the vision scenarios. Optional: if empty, synthetic "
          "tensors matching the vision encoder's declared input shapes are "
          "used instead. Supplying one also requires "
          "--vision_patch_width/height/pooling_kernel_size.");
ABSL_FLAG(std::string, audio_path, "",
          "Real audio clip for the audio scenario, in any format the engine's "
          "audio preprocessor accepts. Optional: if empty, a synthetic tone "
          "of --audio_duration_sec is used instead.");
ABSL_FLAG(std::string, output_csv, "",
          "If set, benchmark results are written to this CSV file.");
ABSL_FLAG(bool, append_csv, false,
          "Append to --output_csv instead of overwriting. The header is only "
          "written when the file is created.");

// ---------------------------------------------------------------------------
// Backends (independently selectable, may be mixed)
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, backend, "cpu",
          "Backend for the text encoder: cpu, gpu, or npu.");
ABSL_FLAG(std::string, vision_backend, "",
          "Backend for the vision encoder. Defaults to --backend.");
ABSL_FLAG(std::string, audio_backend, "",
          "Backend for the audio encoder. Defaults to --backend. Note that "
          "bundles commonly constrain the audio adapter to CPU.");
ABSL_FLAG(std::string, vision_adapter_backend, "cpu",
          "Backend for the vision adapter. Bundles commonly constrain this to "
          "CPU, which is why it defaults to cpu rather than to --backend.");

// ---------------------------------------------------------------------------
// Sweep scope
// ---------------------------------------------------------------------------
ABSL_FLAG(int, max_benchmark_text_tokens, 0,
          "Upper bound for the text sweep. Text encoder signatures larger "
          "than this are skipped. 0 means sweep every signature.");
ABSL_FLAG(int, max_benchmark_vision_tokens, 0,
          "Upper bound for the vision sweep. Vision encoder signatures larger "
          "than this are skipped. 0 means sweep every signature.");
ABSL_FLAG(bool, benchmark_text, true, "Run the text scenarios.");
ABSL_FLAG(bool, benchmark_vision, true, "Run the vision scenarios.");
ABSL_FLAG(bool, benchmark_audio, true, "Run the audio scenario.");
ABSL_FLAG(bool, benchmark_text_overflow, true,
          "Additionally run a prompt longer than the largest text signature, "
          "to exercise the chunk_and_average overflow path.");
ABSL_FLAG(std::string, scenario_filter, "",
          "Comma-separated list of scenario names to run (e.g. "
          "'text_128,vision_70'). Empty runs all. Useful for driving one "
          "scenario per process so that peak-RSS is not cumulative.");
ABSL_FLAG(bool, list_scenarios, false,
          "Print the scenario names this configuration would run, one per "
          "line, and exit without running anything.");

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------
ABSL_FLAG(int, num_warmup, 2, "Warmup iterations per scenario.");
ABSL_FLAG(int, num_iterations, 20, "Measured iterations per scenario.");
ABSL_FLAG(int, cooldown_ms, 0, "Sleep between measured iterations.");

// ---------------------------------------------------------------------------
// Image preprocessing
//
// These only apply to --image_path. Patchifying a real image requires knowing
// the geometry the encoder was trained on, and that is exactly what is missing
// from the model metadata in some builds. Rather than infer it from tensor
// shapes, which is guesswork dressed up as arithmetic, all three are required
// whenever --image_path is set. The synthetic path needs none of them.
// ---------------------------------------------------------------------------
ABSL_FLAG(int, vision_patch_width, 0,
          "Patch width used to patchify --image_path. Required with it.");
ABSL_FLAG(int, vision_patch_height, 0,
          "Patch height used to patchify --image_path. Required with it.");
ABSL_FLAG(int, vision_pooling_kernel_size, 0,
          "Pooling kernel size used to patchify --image_path. Required with "
          "it. The number of patches fed to the encoder is vision_tokens * "
          "pooling_kernel_size^2.");

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------
ABSL_FLAG(double, audio_duration_sec, 5.0,
          "Duration of the synthetic audio clip. Ignored when --audio_path is "
          "set, in which case the file is used at its natural length.");
ABSL_FLAG(int, audio_sample_rate_hz, 16000,
          "Sample rate of the synthetic audio clip. Must be the rate the "
          "audio encoder expects, since synthetic PCM frames bypass the "
          "resampling the file path would get.");
ABSL_FLAG(int, audio_text_signature, 0,
          "Text encoder sequence length to load for the audio scenario. 0 "
          "estimates it from --audio_duration_sec.");

// ---------------------------------------------------------------------------
// Execution knobs
// ---------------------------------------------------------------------------
ABSL_FLAG(bool, normalize, true, "L2-normalize the output embedding.");
ABSL_FLAG(bool, use_mmap, false, "Memory-map the model file.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Directory containing the LiteRT dispatch libraries (NPU).");
ABSL_FLAG(int, num_cpu_threads, 0,
          "CPU thread count. Applied to every modality running on CPU.");
ABSL_FLAG(std::string, activation_data_type, "",
          "Activation data type: float32, float16, int16, or int8.");
ABSL_FLAG(std::string, gpu_precisions, "",
          "Comma-separated list of GPU compute precisions to sweep, e.g. "
          "'fp16,fp32'. Every scenario is run once per entry, each in its own "
          "engine, and the scenario name is suffixed with '@<precision>'. "
          "Only the executors actually running on GPU are affected; CPU and "
          "NPU executors are left alone. Empty (the default) runs each "
          "scenario once and leaves the precision to --activation_data_type, "
          "or to the model metadata when that is also unset. Setting this "
          "overrides --activation_data_type for the GPU executors.");

namespace {

using ::litert::lm::ActivationDataType;
using ::litert::lm::Backend;
using ::litert::lm::BuildLiteRtCompiledModelResources;
using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::EmbeddingResponse;
using ::litert::lm::GetAvailableSignatures;
using ::litert::lm::InputAudio;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputOverflowStrategy;
using ::litert::lm::InputText;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelType;
using ::litert::lm::OwnedEnvironment;
using ::litert::lm::ScopedFile;
using ::litert::lm::SignatureInfo;
using ::litert::support::ImagePreprocessor;
using ::litert::support::ImagePreprocessParameter;
using ::litert::support::Tokenizer;

// ===========================================================================
// Diagnostics
// ===========================================================================

// Reports a benchmark-level problem that does not stop the run.
//
// This deliberately bypasses ABSL_LOG. The binary raises the minimum log level
// to ERROR so that the engine's very chatty INFO and VERBOSE output does not
// bury the results, which also suppresses ABSL_LOG(WARNING). These warnings
// change what the sweep actually measured -- a skipped scenario, an ignored
// flag -- so they have to survive that.
void Warn(absl::string_view message) {
  std::cerr << "WARNING: " << message << std::endl;
}

// ===========================================================================
// Statistics
// ===========================================================================

struct LatencyStats {
  int n = 0;
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
};

// Returns summary statistics over `values`. `stddev` is the sample standard
// deviation (Bessel-corrected), which is the right estimator here because the
// iterations are a sample of the device's behaviour rather than the whole
// population.
LatencyStats ComputeStats(std::vector<double> values) {
  LatencyStats stats;
  if (values.empty()) return stats;
  std::sort(values.begin(), values.end());
  stats.n = static_cast<int>(values.size());
  double sum = 0.0;
  for (double v : values) sum += v;
  stats.mean = sum / stats.n;
  if (stats.n > 1) {
    double sq = 0.0;
    for (double v : values) sq += (v - stats.mean) * (v - stats.mean);
    stats.stddev = std::sqrt(sq / (stats.n - 1));
  }
  stats.min = values.front();
  stats.max = values.back();
  stats.p50 = (stats.n % 2 == 1)
                  ? values[stats.n / 2]
                  : 0.5 * (values[stats.n / 2 - 1] + values[stats.n / 2]);
  const int p90_index =
      std::min(stats.n - 1, static_cast<int>(std::ceil(0.90 * stats.n)) - 1);
  stats.p90 = values[std::max(0, p90_index)];
  return stats;
}

// ===========================================================================
// Memory
// ===========================================================================

struct MemorySnapshot {
  // Process-wide high-water mark. This is monotonic for the lifetime of the
  // process, so when several scenarios run in one process the value reported
  // for a later scenario includes every earlier one. Run one scenario per
  // process (--scenario_filter) if you need an isolated figure.
  double peak_rss_mb = 0.0;
  // Heap currently in use. Unlike the high-water mark this goes back down when
  // an engine is destroyed, so differences between snapshots are meaningful.
  double heap_in_use_mb = 0.0;
  // Current resident set size, read from /proc. Unavailable off Linux.
  double current_rss_mb = 0.0;
};

double ReadCurrentRssMb() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  if (!status.is_open()) return 0.0;
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream iss(line.substr(6));
      double kb = 0.0;
      iss >> kb;
      return kb / 1024.0;
    }
  }
#endif
  return 0.0;
}

MemorySnapshot TakeMemorySnapshot() {
  MemorySnapshot snapshot;
  const auto usage = tflite::profiling::memory::GetMemoryUsage();
  if (tflite::profiling::memory::MemoryUsage::IsSupported()) {
    snapshot.peak_rss_mb = usage.mem_footprint_kb / 1024.0;
    snapshot.heap_in_use_mb =
        static_cast<double>(usage.in_use_allocated_bytes) / (1024.0 * 1024.0);
  }
  snapshot.current_rss_mb = ReadCurrentRssMb();
  return snapshot;
}

// ===========================================================================
// Scenarios
// ===========================================================================

enum class Modality { kText, kVision, kAudio };

absl::string_view ModalityName(Modality modality) {
  switch (modality) {
    case Modality::kText:
      return "text";
    case Modality::kVision:
      return "vision";
    case Modality::kAudio:
      return "audio";
  }
  return "unknown";
}

struct Scenario {
  std::string name;
  Modality modality = Modality::kText;
  // Text: the encoder sequence length. Vision: the visual token count.
  // Audio: unused.
  int capacity = 0;
  // Text only: the prompt, pre-generated to land in `capacity`.
  std::string prompt;
  // Text only: the token count of `prompt`, as measured by the bundle's own
  // tokenizer.
  int prompt_tokens = 0;
  // Text only: whether this scenario deliberately overflows the largest
  // signature and must use chunk_and_average.
  bool overflow = false;
  // The text encoder sequence length the engine should load. For vision and
  // audio scenarios this must be large enough to hold the soft tokens that the
  // encoder emits.
  int text_signature_to_load = 0;
  // Set when --gpu_precisions is in use. Applied only to the executors whose
  // backend is GPU, and overriding --activation_data_type for those. Unset
  // means "leave the precision to the usual resolution waterfall".
  std::optional<ActivationDataType> gpu_precision;
  // Human-readable form of `gpu_precision` ("fp16"/"fp32"), empty when unset.
  // Carried separately so the table and CSV can report it without having to
  // reverse the enum.
  std::string precision_label;
};

struct ScenarioResult {
  std::string name;
  Modality modality = Modality::kText;
  int capacity = 0;
  bool ok = false;
  std::string error;
  // Empty unless --gpu_precisions was used.
  std::string precision_label;

  // Observed from the response.
  int input_length = 0;
  int num_chunks = 0;
  int embedding_dim = 0;

  // Timings.
  LatencyStats e2e;
  double engine_init_ms = 0.0;
  double host_preprocess_ms = 0.0;
  // Named marks reported by the engine's own benchmark instrumentation.
  std::map<std::string, double> marks;

  // Memory.
  MemorySnapshot before;
  MemorySnapshot after;
};

// ===========================================================================
// Prompt generation
// ===========================================================================

// A deliberately mundane word list. Using ordinary lowercase words keeps the
// tokens-per-word ratio close to 1 and stable, which makes the search below
// converge quickly.
constexpr absl::string_view kFillerWords[] = {
    "the",       "quick",      "brown",   "fox",        "jumps",     "over",
    "lazy",      "dog",        "and",     "runs",       "through",   "forest",
    "exploring", "every",      "corner",  "with",       "curiosity", "speed",
    "wonder",    "grace",      "leaping", "across",     "crystal",   "clear",
    "streams",   "beneath",    "ancient", "trees",      "under",     "starlit",
    "skies",     "searching",  "new",     "adventures", "finding",   "hidden",
    "treasures", "whispering", "winds",   "sunlit",     "valleys",   "peaceful",
    "meadows",   "blooming",   "flowers", "singing",    "birds",     "morning",
};
constexpr int kNumFillerWords = sizeof(kFillerWords) / sizeof(kFillerWords[0]);

std::string MakePromptOfWords(int num_words) {
  std::string prompt;
  prompt.reserve(num_words * 7);
  for (int i = 0; i < num_words; ++i) {
    if (i > 0) prompt.push_back(' ');
    prompt.append(kFillerWords[i % kNumFillerWords].data(),
                  kFillerWords[i % kNumFillerWords].size());
  }
  return prompt;
}

// Builds a prompt whose tokenized length is as close to `target_tokens` as
// possible without exceeding it.
//
// Token count is non-decreasing in word count, so this binary searches for the
// largest word count that still fits. That is ~12 tokenizer calls, which is
// negligible next to a model load, and it avoids the hand-calibrated word
// counts that previous benchmark harnesses relied on (and that silently broke
// whenever the tokenizer changed).
absl::StatusOr<std::pair<std::string, int>> BuildPromptForTokenCount(
    Tokenizer& tokenizer, int target_tokens) {
  if (target_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("target_tokens must be positive, got ", target_tokens));
  }
  int low = 1;
  // Generous upper bound: even a pathological tokenizer will not emit fewer
  // than one token per four words for this vocabulary.
  int high = std::max(8, target_tokens * 4);

  std::string best_prompt = MakePromptOfWords(1);
  int best_tokens = 0;

  while (low <= high) {
    const int mid = low + (high - low) / 2;
    std::string candidate = MakePromptOfWords(mid);
    LITERT_ASSIGN_OR_RETURN(auto ids, tokenizer.TextToTokenIds(candidate));
    const int num_tokens = static_cast<int>(ids.size());
    if (num_tokens <= target_tokens) {
      best_prompt = std::move(candidate);
      best_tokens = num_tokens;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return std::make_pair(std::move(best_prompt), best_tokens);
}

// ===========================================================================
// Synthetic audio
// ===========================================================================

// A quiet 440 Hz tone, used when no --audio_path is supplied.
//
// This returns PCM frames rather than an encoded clip because InputAudio
// accepts a std::vector<float> directly. Real files are handed to the engine
// as bytes and decoded by its own audio preprocessor, so this binary needs no
// audio codec of its own in either case.
std::vector<float> SynthesizeTone(double duration_sec, int sample_rate_hz) {
  const size_t num_samples = static_cast<size_t>(duration_sec * sample_rate_hz);
  std::vector<float> samples(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    samples[i] = 0.25f * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) /
                                  sample_rate_hz);
  }
  return samples;
}

// ===========================================================================
// File IO
// ===========================================================================

absl::StatusOr<std::string> ReadFileBytes(absl::string_view path) {
  std::ifstream file(std::string(path), std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to open: ", path));
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<ModelAssets> CreateModelAssets(bool use_mmap,
                                              absl::string_view model_path) {
  if (use_mmap) {
    LITERT_ASSIGN_OR_RETURN(auto mapped, MemoryMappedFile::Create(model_path));
    std::shared_ptr<MemoryMappedFile> shared = std::move(mapped);
    return ModelAssets::Create(shared, model_path);
  }
  LITERT_ASSIGN_OR_RETURN(auto opened, ScopedFile::Open(model_path));
  auto shared = std::make_shared<ScopedFile>(std::move(opened));
  return ModelAssets::Create(shared, model_path);
}

// ===========================================================================
// Configuration resolved from flags
// ===========================================================================

// Geometry needed to patchify a real image. Only populated from flags, and
// only when --image_path is set; there are deliberately no defaults, because
// silently guessing the wrong geometry produces a plausible-looking benchmark
// of the wrong workload.
struct PatchGeometry {
  int patch_width = 0;
  int patch_height = 0;
  int pooling_kernel_size = 0;
};

// The shape and element type of one vision encoder input tensor, captured
// verbatim from the model's signature during discovery.
struct VisionTensorSpec {
  std::string name;
  std::vector<int32_t> dimensions;
  // The encoder's map-based Encode only accepts float32 and int32 inputs.
  bool is_float = true;
};

// Every input tensor of one vision encoder signature.
using VisionInputSpec = std::vector<VisionTensorSpec>;

struct RunConfig {
  std::string model_path;
  Backend text_backend = Backend::CPU;
  Backend vision_encoder_backend = Backend::CPU;
  Backend vision_adapter_backend = Backend::CPU;
  Backend audio_backend = Backend::CPU;
  bool use_mmap = false;
  bool enable_file_backed_model_loading = false;
  std::string dispatch_library_dir;
  int num_cpu_threads = 0;
  std::optional<ActivationDataType> activation_data_type;
  int num_warmup = 2;
  int num_iterations = 20;
  int cooldown_ms = 0;
  bool normalize = true;
  // Set only when a real --image_path is being patchified.
  std::optional<PatchGeometry> patch_geometry;
  // Keyed by visual token count, so a vision scenario can look up the input
  // shapes of the signature it is about to exercise. Only used by the
  // synthetic path.
  std::map<int, VisionInputSpec> vision_input_specs;
};

// Rejects the backends this binary has never been able to drive end to end.
// GPU_ARTISAN in particular passes AudioExecutorSettings::SetBackend but is
// unimplemented, so accepting it would turn a typo into a confusing failure
// deep inside engine initialization.
absl::StatusOr<Backend> ParseBackend(absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(const Backend backend,
                          ::litert::lm::GetBackendFromString(name));
  if (backend != Backend::CPU && backend != Backend::GPU &&
      backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend '", name,
                     "'. This benchmark supports cpu, gpu, and npu."));
  }
  return backend;
}

// Canonical short label for a precision, used in scenario names and in the
// CSV. Derived from the enum rather than echoing the user's spelling so that
// --gpu_precisions=fp16 and --gpu_precisions=float16 name the same scenario.
std::string PrecisionLabel(ActivationDataType type) {
  switch (type) {
    case ActivationDataType::FLOAT32:
      return "fp32";
    case ActivationDataType::FLOAT16:
      return "fp16";
    case ActivationDataType::INT16:
      return "int16";
    case ActivationDataType::INT8:
      return "int8";
  }
  return "unknown";
}

// ===========================================================================
// Synthetic vision input
// ===========================================================================

// Records the declared input tensors of one vision encoder signature.
//
// Nothing is inferred here. The encoder's map-based Encode looks up each
// tensor by name and selects its signature from the shape of the "images"
// tensor it is handed, so echoing the signature's own declaration back to it
// is self-consistent by construction. That is the whole reason the synthetic
// path needs no patch geometry: the shapes are not derived from the geometry,
// they *are* the geometry, already resolved by whoever packed the bundle.
absl::StatusOr<VisionInputSpec> CaptureVisionInputSpec(
    const ::litert::Model& model, size_t signature_index) {
  LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(signature_index));
  VisionInputSpec spec;
  for (const absl::string_view name : signature.InputNames()) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, signature.InputTensorType(name));
    const auto element_type = tensor_type.ElementType();
    if (element_type != ::litert::ElementType::Float32 &&
        element_type != ::litert::ElementType::Int32) {
      return absl::UnimplementedError(absl::StrCat(
          "Vision encoder input '", name,
          "' is neither float32 nor int32, which the encoder's Encode() does "
          "not accept. Supply a real --image_path for this bundle."));
    }
    const auto& dimensions = tensor_type.Layout().Dimensions();
    VisionTensorSpec tensor;
    tensor.name = std::string(name);
    tensor.dimensions.assign(dimensions.begin(), dimensions.end());
    tensor.is_float = element_type == ::litert::ElementType::Float32;
    spec.push_back(std::move(tensor));
  }
  if (spec.empty()) {
    return absl::NotFoundError(
        "The vision encoder signature declares no input tensors.");
  }
  return spec;
}

// Fills a tensor of `num_elements` floats with a deterministic, well-behaved
// pattern.
//
// The values are irrelevant to latency on these fixed-shape dense encoders,
// but they are not irrelevant to correctness of the *measurement*: NaNs,
// infinities and denormals can all trip slow paths on some CPU and GPU
// backends. A cheap LCG mapped into [0, 1) stays in the same numeric regime as
// the normalized pixel values a real image would produce.
std::vector<float> MakeDeterministicFloats(size_t num_elements) {
  std::vector<float> values(num_elements);
  uint32_t state = 0x9E3779B9u;
  for (size_t i = 0; i < num_elements; ++i) {
    state = state * 1664525u + 1013904223u;
    values[i] = static_cast<float>(state >> 8) / static_cast<float>(1 << 24);
  }
  return values;
}

// Fills a positions tensor shaped [..., num_patches, 2] with a raster scan.
//
// The real preprocessor writes (column, row) per patch over a
// patches_w x patches_h grid, and the encoder treats these as indices into a
// position embedding. Random values would risk indexing out of range, so this
// lays out a near-square grid, which both stays in range and marks every patch
// as occupied. Full occupancy is the right thing to benchmark: it is the
// worst case, and unlike a real image it does not vary with aspect ratio.
std::vector<int32_t> MakeRasterPositions(int num_patches) {
  const int grid_width =
      std::max(1, static_cast<int>(std::ceil(std::sqrt(num_patches))));
  std::vector<int32_t> positions(static_cast<size_t>(num_patches) * 2);
  for (int i = 0; i < num_patches; ++i) {
    positions[i * 2 + 0] = i % grid_width;  // column
    positions[i * 2 + 1] = i / grid_width;  // row
  }
  return positions;
}

// Builds a complete synthetic input for one vision encoder signature.
absl::StatusOr<InputImage> MakeSyntheticVisionInput(
    const VisionInputSpec& spec) {
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> tensor_map;
  for (const VisionTensorSpec& tensor : spec) {
    size_t num_elements = 1;
    for (const int32_t dimension : tensor.dimensions) {
      if (dimension <= 0) {
        return absl::UnimplementedError(absl::StrCat(
            "Vision encoder input '", tensor.name,
            "' has a dynamic or zero dimension, which cannot be synthesized. "
            "Supply a real --image_path for this bundle."));
      }
      num_elements *= static_cast<size_t>(dimension);
    }
    ::litert::Dimensions dimensions(tensor.dimensions.begin(),
                                    tensor.dimensions.end());

    if (tensor.is_float) {
      const std::vector<float> values = MakeDeterministicFloats(num_elements);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer, ::litert::lm::CopyToTensorBuffer<float>(
                           absl::MakeConstSpan(values), std::move(dimensions)));
      tensor_map.emplace(tensor.name, std::move(buffer));
    } else {
      // The only int32 input these encoders take is the per-patch position
      // grid, whose patch count is the second-to-last dimension.
      const int num_patches =
          tensor.dimensions.size() >= 2
              ? tensor.dimensions[tensor.dimensions.size() - 2]
              : static_cast<int>(num_elements / 2);
      std::vector<int32_t> values = MakeRasterPositions(num_patches);
      values.resize(num_elements, 0);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer, ::litert::lm::CopyToTensorBuffer<int32_t>(
                           absl::MakeConstSpan(values), std::move(dimensions)));
      tensor_map.emplace(tensor.name, std::move(buffer));
    }
  }
  return InputImage(std::move(tensor_map));
}

// ===========================================================================
// Scenario execution
// ===========================================================================

// Returns the smallest available signature that can hold `required_length`,
// falling back to the largest one when nothing is big enough.
int SmallestSignatureAtLeast(const std::vector<int>& sorted_lengths,
                             int required_length) {
  for (int length : sorted_lengths) {
    if (length >= required_length) return length;
  }
  return sorted_lengths.empty() ? 0 : sorted_lengths.back();
}

absl::Status ConfigureSettings(const RunConfig& config,
                               const Scenario& scenario,
                               EmbeddingEngineSettings& settings) {
  // Enabling the benchmark params is what makes the engine record its
  // per-stage marks; without this GetBenchmarkInfo() stays empty.
  settings.GetMutableBenchmarkParams();

  // Resolves the activation data type for one executor. A --gpu_precisions
  // variant only applies where that executor really is on GPU, because the
  // flag is about GPU calculation precision and silently retyping a CPU or NPU
  // executor would make the two sweep arms differ in more than the one
  // variable under test.
  const auto activation_for =
      [&config,
       &scenario](Backend backend) -> std::optional<ActivationDataType> {
    if (scenario.gpu_precision.has_value() && backend == Backend::GPU) {
      return scenario.gpu_precision;
    }
    return config.activation_data_type;
  };

  auto& main_settings = settings.GetMutableMainExecutorSettings();
  if (!config.dispatch_library_dir.empty()) {
    main_settings.SetLitertDispatchLibDir(config.dispatch_library_dir);
  }
  if (config.num_cpu_threads > 0 && config.text_backend == Backend::CPU) {
    main_settings.SetNumThreads(config.num_cpu_threads);
  }
  if (const auto activation = activation_for(config.text_backend);
      activation.has_value()) {
    main_settings.SetActivationDataType(*activation);
  }

  if (auto& vision_settings = settings.GetMutableVisionExecutorSettings();
      vision_settings.has_value()) {
    // CreateDefault already pins the adapter to CPU, but bundles differ in
    // what they allow, so make the choice explicit and overridable.
    ABSL_RETURN_IF_ERROR(
        vision_settings->SetAdapterBackend(config.vision_adapter_backend));
    if (!config.dispatch_library_dir.empty()) {
      vision_settings->SetLitertDispatchLibDir(config.dispatch_library_dir);
    }
    if (const auto activation = activation_for(config.vision_encoder_backend);
        activation.has_value()) {
      vision_settings->SetActivationDataType(*activation);
    }
  }

  if (auto& audio_settings = settings.GetMutableAudioExecutorSettings();
      audio_settings.has_value()) {
    if (!config.dispatch_library_dir.empty()) {
      audio_settings->SetLitertDispatchLibDir(config.dispatch_library_dir);
    }
    if (const auto activation = activation_for(config.audio_backend);
        activation.has_value()) {
      audio_settings->SetActivationDataType(*activation);
    }
  }

  // Pinning both bounds to the same value makes the engine load exactly one
  // text encoder signature, which is what keeps the per-scenario memory
  // figures attributable to that sequence length.
  if (scenario.text_signature_to_load > 0) {
    settings.SetMinInputLength(scenario.text_signature_to_load);
    settings.SetMaxInputLength(scenario.text_signature_to_load);
  }
  if (scenario.modality == Modality::kVision) {
    settings.SetVisionTokensPerImage(scenario.capacity);
  }
  return absl::OkStatus();
}

// The multimodal payloads, resolved once and reused by every scenario.
//
// Each modality is in exactly one of two states: a real file was supplied, or
// it was not and the input is synthetic. Keeping both in one struct lets
// RunScenario stay a two-argument function as the modes multiply.
struct BenchmarkInputs {
  // Raw bytes of --image_path. Empty means the vision scenarios synthesize
  // their input from the encoder signature instead.
  std::string image_bytes;
  // Raw bytes of --audio_path, passed to the engine's own preprocessor.
  // Empty means `audio_pcm` is used instead.
  std::string audio_bytes;
  // Synthetic mono PCM frames, used only when `audio_bytes` is empty.
  std::vector<float> audio_pcm;
};

// Patchifies a real `image_bytes` for a `vision_tokens`-token signature.
//
// Only used for --image_path. The engine can do this itself, but only when it
// can read the patch parameters out of model metadata, which is not possible
// in every build; doing it here also isolates the host-side cost from the
// encoder cost.
absl::StatusOr<InputImage> PatchifyImage(const PatchGeometry& geometry,
                                         absl::string_view image_bytes,
                                         int vision_tokens) {
  std::unique_ptr<ImagePreprocessor> preprocessor = ImagePreprocessor::Create();
  if (preprocessor == nullptr) {
    return absl::InternalError("Failed to create an image preprocessor.");
  }
  ImagePreprocessParameter parameter;
  // Deliberately leaves the target dimensions unset: that is what selects the
  // aspect-ratio-preserving resize inside the preprocessor, which is the
  // behaviour the packed encoders were calibrated against.
  parameter.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = geometry.patch_width,
      .patch_height = geometry.patch_height,
      .max_num_patches = vision_tokens * geometry.pooling_kernel_size *
                         geometry.pooling_kernel_size,
      .pooling_kernel_size = geometry.pooling_kernel_size,
  });
  InputImage raw_image{std::string(image_bytes)};
  return preprocessor->Preprocess(raw_image, parameter);
}

absl::Status RunScenarioOrDie(const RunConfig& config, const Scenario& scenario,
                              const BenchmarkInputs& inputs,
                              ScenarioResult& result) {
  LITERT_ASSIGN_OR_RETURN(
      auto model_assets, CreateModelAssets(config.use_mmap, config.model_path));
  LITERT_ASSIGN_OR_RETURN(
      auto resources,
      BuildLiteRtCompiledModelResources(
          model_assets, config.enable_file_backed_model_loading));
  LITERT_ASSIGN_OR_RETURN(auto tokenizer, resources->GetTokenizer());
  if (tokenizer == nullptr) {
    return absl::NotFoundError("Tokenizer not found in model resources.");
  }

  // Only the modality under test is configured, so a text scenario never pays
  // for compiling the vision or audio encoder.
  std::optional<Backend> vision_backend;
  std::optional<Backend> audio_backend;
  if (scenario.modality == Modality::kVision) {
    vision_backend = config.vision_encoder_backend;
  } else if (scenario.modality == Modality::kAudio) {
    audio_backend = config.audio_backend;
  }

  LITERT_ASSIGN_OR_RETURN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                             model_assets, config.text_backend,
                                             vision_backend, audio_backend));
  ABSL_RETURN_IF_ERROR(ConfigureSettings(config, scenario, settings));

  LITERT_ASSIGN_OR_RETURN(auto environment, ::litert::lm::CreateEnvironment(
                                                settings, resources.get()));
  auto owned_environment =
      std::make_unique<OwnedEnvironment>(std::move(environment));

  const absl::Time init_start = absl::Now();
  LITERT_ASSIGN_OR_RETURN(
      auto engine, EmbeddingEngineImpl::Create(
                       std::move(resources), std::move(owned_environment),
                       std::move(tokenizer), std::move(settings)));
  result.engine_init_ms = absl::ToDoubleMilliseconds(absl::Now() - init_start);

  std::vector<InputData> contents;
  switch (scenario.modality) {
    case Modality::kText:
      contents.emplace_back(InputText(scenario.prompt));
      break;
    case Modality::kVision: {
      const absl::Time preprocess_start = absl::Now();
      std::optional<InputImage> image;
      if (inputs.image_bytes.empty()) {
        const auto spec = config.vision_input_specs.find(scenario.capacity);
        if (spec == config.vision_input_specs.end()) {
          return absl::InternalError(
              absl::StrCat("No vision input spec was captured for ",
                           scenario.capacity, " tokens."));
        }
        LITERT_ASSIGN_OR_RETURN(image, MakeSyntheticVisionInput(spec->second));
      } else {
        if (!config.patch_geometry.has_value()) {
          return absl::InternalError(
              "An image was supplied but no patch geometry was resolved.");
        }
        LITERT_ASSIGN_OR_RETURN(
            image, PatchifyImage(*config.patch_geometry, inputs.image_bytes,
                                 scenario.capacity));
      }
      result.host_preprocess_ms =
          absl::ToDoubleMilliseconds(absl::Now() - preprocess_start);
      contents.emplace_back(*std::move(image));
      break;
    }
    case Modality::kAudio:
      // A real clip goes to the engine as bytes so that its own preprocessor
      // decodes it; a synthetic one is already PCM and skips that entirely.
      if (inputs.audio_bytes.empty()) {
        contents.emplace_back(InputAudio(inputs.audio_pcm));
      } else {
        contents.emplace_back(InputAudio(inputs.audio_bytes));
      }
      break;
  }

  EmbeddingOptions options;
  options.normalize = config.normalize;
  // Text scenarios are sized to fit, so an overflow there would be a bug worth
  // surfacing. The multimodal scenarios cannot know their soft-token count in
  // advance, so they degrade to chunking rather than failing; `num_chunks` in
  // the output makes that visible.
  options.input_overflow_strategy =
      (scenario.modality == Modality::kText && !scenario.overflow)
          ? InputOverflowStrategy::kError
          : InputOverflowStrategy::kChunkAndAverage;

  for (int i = 0; i < config.num_warmup; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto unused_response,
                            engine->ComputeEmbedding(contents, options));
    (void)unused_response;
  }

  std::vector<double> latencies_ms;
  latencies_ms.reserve(config.num_iterations);
  EmbeddingResponse last_response;
  for (int i = 0; i < config.num_iterations; ++i) {
    const absl::Time start = absl::Now();
    LITERT_ASSIGN_OR_RETURN(auto response,
                            engine->ComputeEmbedding(contents, options));
    latencies_ms.push_back(absl::ToDoubleMilliseconds(absl::Now() - start));
    last_response = std::move(response);
    if (config.cooldown_ms > 0) {
      absl::SleepFor(absl::Milliseconds(config.cooldown_ms));
    }
  }

  result.e2e = ComputeStats(std::move(latencies_ms));
  result.input_length = last_response.input_length;
  result.num_chunks = last_response.num_chunks;
  result.embedding_dim = static_cast<int>(last_response.embedding.size());

  if (auto benchmark_info = engine->GetBenchmarkInfo();
      benchmark_info.has_value()) {
    // Each mark is a start/stop pair per request, so what survives the loop is
    // the last iteration's duration for that stage rather than a total.
    for (const auto& [name, duration] : benchmark_info->GetMarkDurations()) {
      result.marks[name] = absl::ToDoubleMilliseconds(duration);
    }
    for (const auto& [name, duration] : benchmark_info->GetInitPhases()) {
      result.marks[name] = absl::ToDoubleMilliseconds(duration);
    }
  }
  return absl::OkStatus();
}

ScenarioResult RunScenario(const RunConfig& config, const Scenario& scenario,
                           const BenchmarkInputs& inputs) {
  ScenarioResult result;
  result.name = scenario.name;
  result.modality = scenario.modality;
  result.capacity = scenario.capacity;
  result.precision_label = scenario.precision_label;
  result.before = TakeMemorySnapshot();
  const absl::Status status =
      RunScenarioOrDie(config, scenario, inputs, result);
  result.after = TakeMemorySnapshot();
  result.ok = status.ok();
  if (!status.ok()) {
    result.error = status.ToString();
  }
  return result;
}

// ===========================================================================
// Reporting
// ===========================================================================

std::string FormatMarks(const std::map<std::string, double>& marks) {
  std::vector<std::string> parts;
  parts.reserve(marks.size());
  for (const auto& [name, value] : marks) {
    parts.push_back(absl::StrFormat("%s=%.3f", name, value));
  }
  return absl::StrJoin(parts, ";");
}

void PrintResultTable(const std::vector<ScenarioResult>& results) {
  // 24 wide because a precision-suffixed name such as
  // "text_overflow_1536@fp32" is 23 characters.
  std::cout << "\n"
            << absl::StrFormat(
                   "%-24s %-7s %8s %8s %10s %9s %9s %9s %10s %10s\n",
                   "scenario", "modality", "capacity", "tokens", "mean_ms",
                   "stddev", "p50_ms", "p90_ms", "init_ms", "peak_mb");
  std::cout << std::string(114, '-') << "\n";
  for (const ScenarioResult& result : results) {
    if (!result.ok) {
      std::cout << absl::StrFormat("%-24s %-7s %8d  FAILED: %s\n", result.name,
                                   ModalityName(result.modality),
                                   result.capacity, result.error);
      continue;
    }
    std::cout << absl::StrFormat(
        "%-24s %-7s %8d %8d %10.2f %9.2f %9.2f %9.2f %10.2f %10.1f\n",
        result.name, ModalityName(result.modality), result.capacity,
        result.input_length, result.e2e.mean, result.e2e.stddev, result.e2e.p50,
        result.e2e.p90, result.engine_init_ms, result.after.peak_rss_mb);
  }
  std::cout << std::endl;

  for (const ScenarioResult& result : results) {
    if (!result.ok || result.marks.empty()) continue;
    std::cout << result.name << " stage marks (last iteration):\n";
    for (const auto& [name, value] : result.marks) {
      std::cout << absl::StrFormat("    %-32s %10.3f ms\n", name, value);
    }
  }
}

std::string EscapeCsvField(absl::string_view field) {
  if (field.find_first_of(",\"\n") == absl::string_view::npos) {
    return std::string(field);
  }
  std::string escaped = "\"";
  for (const char c : field) {
    if (c == '"') escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

absl::Status WriteCsv(absl::string_view path, bool append,
                      const RunConfig& config, const BenchmarkInputs& inputs,
                      const std::vector<ScenarioResult>& results) {
  const bool write_header = !append || !std::ifstream(std::string(path)).good();
  std::ofstream out(std::string(path),
                    append ? (std::ios::out | std::ios::app) : std::ios::out);
  if (!out.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open CSV file for writing: ", path));
  }
  // The driver script cuts fixed column indices out of this file, so new
  // columns go on the end rather than anywhere more logical.
  if (write_header) {
    out << "timestamp,model,scenario,modality,gpu_precision,capacity,ok,"
           "text_backend,"
           "vision_backend,vision_adapter_backend,audio_backend,iterations,"
           "input_tokens,num_chunks,embedding_dim,mean_ms,stddev_ms,min_ms,"
           "p50_ms,p90_ms,max_ms,engine_init_ms,host_preprocess_ms,"
           "peak_rss_mb,rss_delta_mb,heap_delta_mb,marks,error,"
           "vision_input,audio_input\n";
  }
  const absl::string_view vision_input =
      inputs.image_bytes.empty() ? "synthetic" : "file";
  const absl::string_view audio_input =
      inputs.audio_bytes.empty() ? "synthetic" : "file";
  const std::string timestamp =
      absl::FormatTime(absl::RFC3339_sec, absl::Now(), absl::LocalTimeZone());
  for (const ScenarioResult& result : results) {
    out << absl::StrFormat(
        "%s,%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,"
        "%.4f,"
        "%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%s,%s,%s,%s\n",
        timestamp, EscapeCsvField(config.model_path),
        EscapeCsvField(result.name), ModalityName(result.modality),
        result.precision_label, result.capacity, result.ok ? 1 : 0,
        ::litert::lm::GetBackendString(config.text_backend),
        ::litert::lm::GetBackendString(config.vision_encoder_backend),
        ::litert::lm::GetBackendString(config.vision_adapter_backend),
        ::litert::lm::GetBackendString(config.audio_backend), result.e2e.n,
        result.input_length, result.num_chunks, result.embedding_dim,
        result.e2e.mean, result.e2e.stddev, result.e2e.min, result.e2e.p50,
        result.e2e.p90, result.e2e.max, result.engine_init_ms,
        result.host_preprocess_ms, result.after.peak_rss_mb,
        result.after.current_rss_mb - result.before.current_rss_mb,
        result.after.heap_in_use_mb - result.before.heap_in_use_mb,
        EscapeCsvField(FormatMarks(result.marks)), EscapeCsvField(result.error),
        vision_input, audio_input);
  }
  return absl::OkStatus();
}

// ===========================================================================
// Driver
// ===========================================================================

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);

  RunConfig config;
  config.model_path = absl::GetFlag(FLAGS_model_path);
  if (config.model_path.empty()) {
    return absl::InvalidArgumentError("--model_path is required.");
  }

  LITERT_ASSIGN_OR_RETURN(config.text_backend,
                          ParseBackend(absl::GetFlag(FLAGS_backend)));
  const std::string vision_backend_flag = absl::GetFlag(FLAGS_vision_backend);
  LITERT_ASSIGN_OR_RETURN(
      config.vision_encoder_backend,
      ParseBackend(vision_backend_flag.empty() ? absl::GetFlag(FLAGS_backend)
                                               : vision_backend_flag));
  const std::string audio_backend_flag = absl::GetFlag(FLAGS_audio_backend);
  LITERT_ASSIGN_OR_RETURN(
      config.audio_backend,
      ParseBackend(audio_backend_flag.empty() ? absl::GetFlag(FLAGS_backend)
                                              : audio_backend_flag));
  LITERT_ASSIGN_OR_RETURN(
      config.vision_adapter_backend,
      ParseBackend(absl::GetFlag(FLAGS_vision_adapter_backend)));

  config.num_warmup = absl::GetFlag(FLAGS_num_warmup);
  config.num_iterations = absl::GetFlag(FLAGS_num_iterations);
  if (config.num_iterations <= 0) {
    return absl::InvalidArgumentError("--num_iterations must be positive.");
  }
  config.cooldown_ms = absl::GetFlag(FLAGS_cooldown_ms);
  config.normalize = absl::GetFlag(FLAGS_normalize);
  config.num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  config.dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);

  if (const std::string activation_data_type =
          absl::GetFlag(FLAGS_activation_data_type);
      !activation_data_type.empty()) {
    LITERT_ASSIGN_OR_RETURN(
        config.activation_data_type,
        ::litert::lm::GetActivationDataTypeFromString(activation_data_type));
  }

  // NPU needs the weights file-backed, and that is incompatible with mapping
  // the bundle ourselves. Any modality on NPU is enough to force this, which
  // is what makes mixed CPU/NPU configurations work.
  config.enable_file_backed_model_loading =
      config.text_backend == Backend::NPU ||
      config.vision_encoder_backend == Backend::NPU ||
      config.audio_backend == Backend::NPU;
  config.use_mmap = absl::GetFlag(FLAGS_use_mmap);
  if (config.enable_file_backed_model_loading && config.use_mmap) {
    Warn(
        "Disabling --use_mmap because an NPU backend was requested and NPU "
        "requires file-backed loading.");
    config.use_mmap = false;
  }

  // -- Discovery ----------------------------------------------------------
  // A throwaway engine-less load, just to learn what the bundle contains and
  // to size the prompts. Everything here is released before the first
  // scenario so that it cannot contaminate the memory figures.
  std::vector<int> text_lengths;
  std::vector<int> vision_lengths;
  bool has_audio = false;
  std::map<int, std::pair<std::string, int>> prompts_by_length;
  std::map<int, VisionInputSpec> vision_input_specs;
  {
    LITERT_ASSIGN_OR_RETURN(
        auto model_assets,
        CreateModelAssets(config.use_mmap, config.model_path));
    LITERT_ASSIGN_OR_RETURN(
        auto resources,
        BuildLiteRtCompiledModelResources(
            model_assets, config.enable_file_backed_model_loading));

    LITERT_ASSIGN_OR_RETURN(
        const auto text_signatures,
        GetAvailableSignatures(*resources, ModelType::kTfLiteTextEncoder));
    for (const SignatureInfo& signature : text_signatures) {
      if (signature.length > 0) text_lengths.push_back(signature.length);
    }
    std::sort(text_lengths.begin(), text_lengths.end());

    if (auto vision_signatures =
            GetAvailableSignatures(*resources, ModelType::kTfLiteVisionEncoder);
        vision_signatures.ok()) {
      LITERT_ASSIGN_OR_RETURN(
          const ::litert::Model* vision_model,
          resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder));
      // GetAvailableSignatures walks the model's signatures in order, so index
      // i here is signature i there.
      for (size_t i = 0; i < vision_signatures->size(); ++i) {
        const int tokens = (*vision_signatures)[i].length;
        if (tokens <= 0) continue;
        vision_lengths.push_back(tokens);
        // A bundle that cannot be synthesized is not fatal at this point,
        // because the run may still be given a real --image_path. The failure
        // surfaces per scenario instead.
        if (auto spec = CaptureVisionInputSpec(*vision_model, i); spec.ok()) {
          vision_input_specs[tokens] = *std::move(spec);
        } else {
          Warn(absl::StrCat(
              "Cannot synthesize input for the ", tokens,
              "-token vision signature: ", spec.status().ToString()));
        }
      }
      std::sort(vision_lengths.begin(), vision_lengths.end());
    }

    has_audio =
        resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok() ||
        resources->GetTFLiteModel(ModelType::kTfLiteAudioFrontend).ok();

    LITERT_ASSIGN_OR_RETURN(auto tokenizer, resources->GetTokenizer());
    if (tokenizer == nullptr) {
      return absl::NotFoundError("Tokenizer not found in model resources.");
    }
    // The prompts have to be built here because the tokenizer is handed to
    // the engine, and each scenario builds a fresh one.
    for (const int length : text_lengths) {
      // Leaves room for the special tokens the engine inserts, so a scenario
      // named text_128 really does run on the 128-token signature.
      LITERT_ASSIGN_OR_RETURN(auto prompt,
                              BuildPromptForTokenCount(*tokenizer, length - 8));
      prompts_by_length[length] = std::move(prompt);
    }
    if (absl::GetFlag(FLAGS_benchmark_text_overflow) && !text_lengths.empty()) {
      const int largest = text_lengths.back();
      LITERT_ASSIGN_OR_RETURN(
          auto prompt,
          BuildPromptForTokenCount(*tokenizer, largest + largest / 2));
      prompts_by_length[largest + largest / 2] = std::move(prompt);
    }
  }

  if (text_lengths.empty()) {
    return absl::NotFoundError("The bundle has no text encoder signatures.");
  }

  config.vision_input_specs = std::move(vision_input_specs);

  // -- Inputs -------------------------------------------------------------
  BenchmarkInputs inputs;
  if (const std::string image_path = absl::GetFlag(FLAGS_image_path);
      !image_path.empty()) {
    // The geometry is only knowable from outside the bundle, so demand all of
    // it rather than half-guess. The synthetic path, which needs none of it,
    // is what makes this a reasonable thing to demand.
    const PatchGeometry geometry{
        .patch_width = absl::GetFlag(FLAGS_vision_patch_width),
        .patch_height = absl::GetFlag(FLAGS_vision_patch_height),
        .pooling_kernel_size = absl::GetFlag(FLAGS_vision_pooling_kernel_size),
    };
    if (geometry.patch_width <= 0 || geometry.patch_height <= 0 ||
        geometry.pooling_kernel_size <= 0) {
      return absl::InvalidArgumentError(
          "--image_path requires --vision_patch_width, --vision_patch_height "
          "and --vision_pooling_kernel_size, because the patch geometry cannot "
          "be read from the bundle in every build. Omit --image_path to "
          "benchmark with synthetic vision input instead, which needs none of "
          "them.");
    }
    config.patch_geometry = geometry;
    LITERT_ASSIGN_OR_RETURN(inputs.image_bytes, ReadFileBytes(image_path));
  } else {
    for (const int flag : {absl::GetFlag(FLAGS_vision_patch_width),
                           absl::GetFlag(FLAGS_vision_patch_height),
                           absl::GetFlag(FLAGS_vision_pooling_kernel_size)}) {
      if (flag > 0) {
        Warn(
            "Ignoring the --vision_patch_* flags: they only apply to "
            "--image_path, and the vision scenarios are using synthetic "
            "input.");
        break;
      }
    }
  }

  const double audio_duration_sec = absl::GetFlag(FLAGS_audio_duration_sec);
  if (const std::string audio_path = absl::GetFlag(FLAGS_audio_path);
      !audio_path.empty()) {
    // Handed over verbatim: the engine's audio preprocessor decodes and
    // downmixes far more formats than this binary has any business knowing
    // about. The clip is used at its natural length, so --audio_duration_sec
    // no longer describes the workload.
    LITERT_ASSIGN_OR_RETURN(inputs.audio_bytes, ReadFileBytes(audio_path));
  } else {
    inputs.audio_pcm = SynthesizeTone(
        audio_duration_sec, absl::GetFlag(FLAGS_audio_sample_rate_hz));
  }

  // -- Scenarios ----------------------------------------------------------
  std::vector<Scenario> scenarios;
  const int max_text_tokens = absl::GetFlag(FLAGS_max_benchmark_text_tokens);
  if (absl::GetFlag(FLAGS_benchmark_text)) {
    for (const int length : text_lengths) {
      if (max_text_tokens > 0 && length > max_text_tokens) continue;
      Scenario scenario;
      scenario.name = absl::StrCat("text_", length);
      scenario.modality = Modality::kText;
      scenario.capacity = length;
      scenario.prompt = prompts_by_length[length].first;
      scenario.prompt_tokens = prompts_by_length[length].second;
      scenario.text_signature_to_load = length;
      scenarios.push_back(std::move(scenario));
    }
    if (absl::GetFlag(FLAGS_benchmark_text_overflow)) {
      const int largest = text_lengths.back();
      const int target = largest + largest / 2;
      if (max_text_tokens <= 0 || largest <= max_text_tokens) {
        Scenario scenario;
        scenario.name = absl::StrCat("text_overflow_", target);
        scenario.modality = Modality::kText;
        scenario.capacity = target;
        scenario.prompt = prompts_by_length[target].first;
        scenario.prompt_tokens = prompts_by_length[target].second;
        scenario.overflow = true;
        scenario.text_signature_to_load = largest;
        scenarios.push_back(std::move(scenario));
      }
    }
  }

  const int max_vision_tokens =
      absl::GetFlag(FLAGS_max_benchmark_vision_tokens);
  if (absl::GetFlag(FLAGS_benchmark_vision) && !vision_lengths.empty()) {
    for (const int tokens : vision_lengths) {
      if (max_vision_tokens > 0 && tokens > max_vision_tokens) continue;
      if (inputs.image_bytes.empty() &&
          !config.vision_input_specs.contains(tokens)) {
        Warn(absl::StrCat("Skipping vision_", tokens,
                          ": its input could not be synthesized and no "
                          "--image_path was provided."));
        continue;
      }
      Scenario scenario;
      scenario.name = absl::StrCat("vision_", tokens);
      scenario.modality = Modality::kVision;
      scenario.capacity = tokens;
      // The soft tokens plus the surrounding special tokens all pass
      // through the text encoder, so it must be able to hold them.
      scenario.text_signature_to_load =
          SmallestSignatureAtLeast(text_lengths, tokens + 16);
      scenarios.push_back(std::move(scenario));
    }
  }

  if (absl::GetFlag(FLAGS_benchmark_audio) && has_audio) {
    Scenario scenario;
    // Only the synthetic clip has a duration this binary controls. A real file
    // is used as-is, so claiming a duration in the scenario name would
    // misdescribe the workload; the file's own length decides the token count.
    const bool synthetic_audio = inputs.audio_bytes.empty();
    scenario.name = synthetic_audio
                        ? absl::StrFormat("audio_%.0fs", audio_duration_sec)
                        : "audio_file";
    scenario.modality = Modality::kAudio;
    scenario.capacity =
        synthetic_audio ? static_cast<int>(audio_duration_sec * 1000) : 0;
    if (const int override_length = absl::GetFlag(FLAGS_audio_text_signature);
        override_length > 0) {
      scenario.text_signature_to_load = override_length;
    } else {
      // The streaming audio encoders in these bundles emit roughly 25 tokens
      // per second; the margin absorbs both that estimate's error and the
      // special tokens. For a real file --audio_duration_sec is no longer the
      // workload, but it is still the only available hint at how long the clip
      // is, so it keeps sizing the text signature. Use --audio_text_signature
      // when the file is much longer than that.
      const int estimated_tokens =
          static_cast<int>(audio_duration_sec * 25.0) + 32;
      scenario.text_signature_to_load =
          SmallestSignatureAtLeast(text_lengths, estimated_tokens);
    }
    scenarios.push_back(std::move(scenario));
  }

  // -- GPU precision sweep -------------------------------------------------
  //
  // Expands the scenario list into one entry per requested precision. This
  // happens before filtering and listing so that --list_scenarios reports the
  // expanded names and --scenario_filter can select a single
  // scenario/precision pair, which is what lets the driver script run each
  // combination in its own process.
  if (const std::string precisions_flag = absl::GetFlag(FLAGS_gpu_precisions);
      !precisions_flag.empty()) {
    const bool any_gpu = config.text_backend == Backend::GPU ||
                         config.vision_encoder_backend == Backend::GPU ||
                         config.vision_adapter_backend == Backend::GPU ||
                         config.audio_backend == Backend::GPU;
    if (!any_gpu) {
      // Without a GPU executor every arm of the sweep would be byte-for-byte
      // identical, so this is a configuration mistake rather than a no-op
      // worth silently honouring.
      return absl::InvalidArgumentError(
          "--gpu_precisions was set but no backend is gpu. It only affects "
          "executors running on GPU.");
    }

    std::vector<ActivationDataType> precisions;
    std::vector<std::string> labels;
    for (absl::string_view token :
         absl::StrSplit(precisions_flag, ',', absl::SkipWhitespace())) {
      const std::string trimmed =
          std::string(absl::StripAsciiWhitespace(token));
      if (trimmed.empty()) continue;
      LITERT_ASSIGN_OR_RETURN(
          const ActivationDataType type,
          ::litert::lm::GetActivationDataTypeFromString(trimmed));
      const std::string label = PrecisionLabel(type);
      // A duplicate would produce two identically named scenarios and two
      // identical CSV rows, which is only ever a typo.
      if (std::find(labels.begin(), labels.end(), label) != labels.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "--gpu_precisions lists '", label, "' more than once."));
      }
      precisions.push_back(type);
      labels.push_back(label);
    }
    if (precisions.empty()) {
      return absl::InvalidArgumentError(
          "--gpu_precisions was set but parsed to an empty list.");
    }

    if (absl::GetFlag(FLAGS_activation_data_type).empty()) {
      ABSL_LOG(INFO) << "Sweeping GPU precisions: "
                     << absl::StrJoin(labels, ", ");
    } else {
      Warn(
          "--gpu_precisions overrides --activation_data_type for the "
          "executors running on GPU.");
    }

    std::vector<Scenario> expanded;
    expanded.reserve(scenarios.size() * precisions.size());
    for (const Scenario& scenario : scenarios) {
      for (size_t i = 0; i < precisions.size(); ++i) {
        Scenario variant = scenario;
        variant.gpu_precision = precisions[i];
        variant.precision_label = labels[i];
        variant.name = absl::StrCat(scenario.name, "@", labels[i]);
        expanded.push_back(std::move(variant));
      }
    }
    scenarios = std::move(expanded);
  }

  if (const std::string filter = absl::GetFlag(FLAGS_scenario_filter);
      !filter.empty()) {
    const std::vector<std::string> wanted = absl::StrSplit(filter, ',');
    std::vector<Scenario> filtered;
    for (Scenario& scenario : scenarios) {
      if (std::find(wanted.begin(), wanted.end(), scenario.name) !=
          wanted.end()) {
        filtered.push_back(std::move(scenario));
      }
    }
    scenarios = std::move(filtered);
  }

  if (scenarios.empty()) {
    return absl::InvalidArgumentError(
        "No scenarios to run. Check --benchmark_* and --scenario_filter.");
  }

  if (absl::GetFlag(FLAGS_list_scenarios)) {
    for (const Scenario& scenario : scenarios) {
      std::cout << scenario.name << std::endl;
    }
    return absl::OkStatus();
  }

  // -- Run ----------------------------------------------------------------
  std::cout << "Model:            " << config.model_path << "\n"
            << "Text backend:     "
            << ::litert::lm::GetBackendString(config.text_backend) << "\n"
            << "Vision backend:   "
            << ::litert::lm::GetBackendString(config.vision_encoder_backend)
            << " (adapter: "
            << ::litert::lm::GetBackendString(config.vision_adapter_backend)
            << ")\n"
            << "Audio backend:    "
            << ::litert::lm::GetBackendString(config.audio_backend) << "\n"
            << "Text signatures:  " << absl::StrJoin(text_lengths, ", ") << "\n"
            << "Vision signatures:"
            << (vision_lengths.empty()
                    ? " none"
                    : " " + absl::StrJoin(vision_lengths, ", "))
            << "\n"
            << "Audio encoder:    " << (has_audio ? "present" : "absent")
            << "\n"
            << "Vision input:     "
            << (inputs.image_bytes.empty()
                    ? std::string("synthetic (from the encoder signature)")
                    : absl::StrCat("file ", absl::GetFlag(FLAGS_image_path),
                                   ", ", config.patch_geometry->patch_width,
                                   "x", config.patch_geometry->patch_height,
                                   " patches, "
                                   "pooling ",
                                   config.patch_geometry->pooling_kernel_size))
            << "\n"
            << "Audio input:      "
            << (inputs.audio_bytes.empty()
                    ? absl::StrCat("synthetic tone, ", audio_duration_sec, "s")
                    : absl::StrCat("file ", absl::GetFlag(FLAGS_audio_path)))
            << "\n"
            << "Iterations:       " << config.num_warmup << " warmup + "
            << config.num_iterations << " measured\n"
            << "Scenarios:        " << scenarios.size() << std::endl;

  std::vector<ScenarioResult> results;
  results.reserve(scenarios.size());
  for (const Scenario& scenario : scenarios) {
    std::cout << "\n[" << results.size() + 1 << "/" << scenarios.size() << "] "
              << scenario.name << " (text signature "
              << scenario.text_signature_to_load << ")" << std::endl;
    ScenarioResult result = RunScenario(config, scenario, inputs);
    if (result.ok) {
      std::cout << absl::StrFormat("    %.2f ms +/- %.2f (init %.0f ms)\n",
                                   result.e2e.mean, result.e2e.stddev,
                                   result.engine_init_ms);
    } else {
      // A single unsupported modality should not throw away the rest of the
      // sweep, so failures are recorded and reported instead of aborting.
      std::cout << "    FAILED: " << result.error << std::endl;
    }
    results.push_back(std::move(result));
  }

  PrintResultTable(results);

  if (const std::string output_csv = absl::GetFlag(FLAGS_output_csv);
      !output_csv.empty()) {
    ABSL_RETURN_IF_ERROR(WriteCsv(output_csv, absl::GetFlag(FLAGS_append_csv),
                                  config, inputs, results));
    std::cout << "Wrote " << results.size() << " rows to " << output_csv
              << std::endl;
  }

  for (const ScenarioResult& result : results) {
    if (!result.ok) {
      return absl::InternalError(
          absl::StrCat("At least one scenario failed, first: ", result.name));
    }
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  const absl::Status status = MainHelper(argc, argv);
  if (!status.ok()) {
    std::cerr << "Benchmark failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
