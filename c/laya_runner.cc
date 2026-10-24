// Implementation of c/laya_runner.h — see the header for the contract.
// API shape mirrors c/tflite_embed.cc (Environment + CompiledModel, CPU).
//
// Tensor identification: the contract (HOST_CONTRACT.md §C) requires mapping
// by the runtime's serving_default signature order. Signature order alone is
// not observable through this C++ surface, so each tensor is matched by its
// exact byte size — unique within its graph once the window N and dim are
// known from the main outputs (logits [1,N] = 4N bytes < pooled [1,768] =
// 4·768 bytes for every supported N).

#include "laya_runner.h"  // NOLINT

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert

#include <cstdlib>

namespace {

constexpr int kLayaError = 1;

constexpr size_t BytesOf(size_t floats) { return floats * sizeof(float); }

litert::TensorBuffer* FindBySize(std::vector<litert::TensorBuffer>& bufs,
                                 size_t bytes) {
  for (auto& buf : bufs) {
    auto size = buf.Size();
    if (size.HasValue() && *size == bytes) {
      return &buf;
    }
  }
  return nullptr;
}

}  // namespace

struct LiteRtLmLaya {
  std::optional<litert::Environment> env;
  std::unique_ptr<litert::CompiledModel> main_model;
  std::unique_ptr<litert::CompiledModel> act_model;
  // Ordered [embeds, mask, qtype] / [logits, pooled] / [pooled, feats] / [logits].
  std::vector<litert::TensorBuffer> main_inputs;
  std::vector<litert::TensorBuffer> main_outputs;
  std::vector<litert::TensorBuffer> act_inputs;
  std::vector<litert::TensorBuffer> act_outputs;
  size_t window = 0;  // N positions
  size_t dim = 0;     // embedding width
};

extern "C" {

LiteRtLmLaya* litert_lm_laya_create(const char* main_graph_path,
                                    const char* act_graph_path) {
  if (main_graph_path == nullptr || act_graph_path == nullptr) {
    ABSL_LOG(ERROR) << "laya_runner: null path argument";
    return nullptr;
  }
  auto laya = std::make_unique<LiteRtLmLaya>();

  const auto env_options = litert::EnvironmentOptions(
      absl::Span<const litert::EnvironmentOptions::Option>{});
  auto env = litert::Environment::Create(env_options);
  if (!env.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: environment create failed";
    return nullptr;
  }
  laya->env.emplace(std::move(*env));

  // CPU thread count: LITERT_LM_LAYA_THREADS env (0/unset = LiteRT default).
  auto compilation_options = litert::Options::Create();
  if (!compilation_options.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: options create failed";
    return nullptr;
  }
  compilation_options->SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  // CPU thread count: LITERT_LM_LAYA_THREADS env; default 6 (Apple Silicon
  // perf cores — measured 222 ms/question vs 1016 ms at the LiteRT default
  // of 1; 8 threads is not faster).
  int threads = 6;
  if (const char* threads_env = std::getenv("LITERT_LM_LAYA_THREADS")) {
    threads = std::atoi(threads_env);
  }
  if (threads > 0) {
    auto cpu = compilation_options->GetCpuOptions();
    if (cpu.HasValue() && cpu->SetNumThreads(threads).HasValue()) {
      ABSL_VLOG(1) << "laya_runner: cpu threads=" << threads;
    } else {
      ABSL_LOG(ERROR) << "laya_runner: SetNumThreads(" << threads
                      << ") failed; using default";
    }
  }

  auto main_compiled = litert::CompiledModel::Create(
      *laya->env, std::string(main_graph_path), *compilation_options);
  if (!main_compiled.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: main graph compile failed: "
                    << main_compiled.Error().Message();
    return nullptr;
  }
  laya->main_model =
      std::make_unique<litert::CompiledModel>(std::move(*main_compiled));

  auto act_compiled = litert::CompiledModel::Create(
      *laya->env, std::string(act_graph_path), *compilation_options);
  if (!act_compiled.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: act graph compile failed: "
                    << act_compiled.Error().Message();
    return nullptr;
  }
  laya->act_model =
      std::make_unique<litert::CompiledModel>(std::move(*act_compiled));

  auto main_in = laya->main_model->CreateInputBuffers();
  if (!main_in.HasValue() || main_in->size() != 3) {
    ABSL_LOG(ERROR) << "laya_runner: main input buffers failed (want 3)";
    return nullptr;
  }
  auto main_out = laya->main_model->CreateOutputBuffers();
  if (!main_out.HasValue() || main_out->size() != 2) {
    ABSL_LOG(ERROR) << "laya_runner: main output buffers failed (want 2)";
    return nullptr;
  }
  auto act_in = laya->act_model->CreateInputBuffers();
  if (!act_in.HasValue() || act_in->size() != 2) {
    ABSL_LOG(ERROR) << "laya_runner: act input buffers failed (want 2)";
    return nullptr;
  }
  auto act_out = laya->act_model->CreateOutputBuffers();
  if (!act_out.HasValue() || act_out->size() != 1) {
    ABSL_LOG(ERROR) << "laya_runner: act output buffers failed (want 1)";
    return nullptr;
  }
  laya->main_inputs = std::move(*main_in);
  laya->main_outputs = std::move(*main_out);
  laya->act_inputs = std::move(*act_in);
  laya->act_outputs = std::move(*act_out);

  // Window and dim from the main outputs: the smaller is token_logits [1,N],
  // the larger is pooled_cls [1,dim=768].
  auto out0 = laya->main_outputs[0].Size();
  auto out1 = laya->main_outputs[1].Size();
  if (!out0.HasValue() || !out1.HasValue() || *out0 == *out1 ||
      *out0 % sizeof(float) != 0 || *out1 % sizeof(float) != 0) {
    ABSL_LOG(ERROR) << "laya_runner: unexpected main output shapes";
    return nullptr;
  }
  const size_t logits_bytes = std::min(*out0, *out1);
  const size_t pooled_bytes = std::max(*out0, *out1);
  laya->window = logits_bytes / sizeof(float);
  laya->dim = pooled_bytes / sizeof(float);
  if (laya->dim != 768 || (laya->window != 256 && laya->window != 512) ||
      laya->dim <= laya->window) {
    ABSL_LOG(ERROR) << "laya_runner: shapes do not match the Laya contract "
                    << "(window=" << laya->window << " dim=" << laya->dim << ")";
    return nullptr;
  }
  if (pooled_bytes < logits_bytes) {
    std::swap(laya->main_outputs[0], laya->main_outputs[1]);
  }

  // Main inputs by exact size: embeds [1,N,768], mask [1,N], qtype [1,3].
  std::vector<litert::TensorBuffer> ordered;
  auto* embeds = FindBySize(laya->main_inputs,
                            BytesOf(laya->window * laya->dim));
  auto* mask = FindBySize(laya->main_inputs, BytesOf(laya->window));
  auto* qtype = FindBySize(laya->main_inputs, BytesOf(3));
  if (embeds == nullptr || mask == nullptr || qtype == nullptr ||
      embeds == mask || mask == qtype || embeds == qtype) {
    ABSL_LOG(ERROR) << "laya_runner: main input shapes do not match the "
                       "Laya contract";
    return nullptr;
  }
  ordered.push_back(std::move(*embeds));
  ordered.push_back(std::move(*mask));
  ordered.push_back(std::move(*qtype));
  laya->main_inputs = std::move(ordered);

  // Act inputs: pooled [1,dim] and feats [1,4]; output act_logits [1,2].
  auto* act_pooled = FindBySize(laya->act_inputs, BytesOf(laya->dim));
  auto* act_feats = FindBySize(laya->act_inputs, BytesOf(4));
  if (act_pooled == nullptr || act_feats == nullptr || act_pooled == act_feats) {
    ABSL_LOG(ERROR) << "laya_runner: act input shapes do not match the "
                       "Laya contract";
    return nullptr;
  }
  std::vector<litert::TensorBuffer> act_ordered;
  act_ordered.push_back(std::move(*act_pooled));
  act_ordered.push_back(std::move(*act_feats));
  laya->act_inputs = std::move(act_ordered);
  auto act_logits_size = laya->act_outputs[0].Size();
  if (!act_logits_size.HasValue() || *act_logits_size != BytesOf(2)) {
    ABSL_LOG(ERROR) << "laya_runner: act output shape does not match the "
                       "Laya contract";
    return nullptr;
  }

  ABSL_VLOG(1) << "laya_runner ready: window=" << laya->window
               << " dim=" << laya->dim;
  return laya.release();
}

int litert_lm_laya_window(const LiteRtLmLaya* laya) {
  return laya == nullptr ? 0 : static_cast<int>(laya->window);
}

int litert_lm_laya_embed_dim(const LiteRtLmLaya* laya) {
  return laya == nullptr ? 0 : static_cast<int>(laya->dim);
}

int litert_lm_laya_run_main(LiteRtLmLaya* laya, const float* inputs_embeds,
                            const float* attention_mask,
                            const float* qtype_onehot, float* token_logits_out,
                            float* pooled_out) {
  if (laya == nullptr || inputs_embeds == nullptr ||
      attention_mask == nullptr || qtype_onehot == nullptr ||
      token_logits_out == nullptr || pooled_out == nullptr) {
    return kLayaError;
  }

  auto embeds_written = laya->main_inputs[0].Write(
      absl::MakeConstSpan(inputs_embeds, laya->window * laya->dim));
  if (!embeds_written.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: embeds write failed";
    return kLayaError;
  }
  auto mask_written = laya->main_inputs[1].Write(
      absl::MakeConstSpan(attention_mask, laya->window));
  if (!mask_written.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: mask write failed";
    return kLayaError;
  }
  auto qtype_written =
      laya->main_inputs[2].Write(absl::MakeConstSpan(qtype_onehot, 3));
  if (!qtype_written.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: qtype write failed";
    return kLayaError;
  }

  auto run = laya->main_model->Run(absl::MakeSpan(laya->main_inputs),
                                   absl::MakeSpan(laya->main_outputs));
  if (!run.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: main run failed";
    return kLayaError;
  }

  auto logits_read = laya->main_outputs[0].Read(absl::MakeSpan(
      reinterpret_cast<uint8_t*>(token_logits_out), BytesOf(laya->window)));
  if (!logits_read.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: logits read failed";
    return kLayaError;
  }
  auto pooled_read = laya->main_outputs[1].Read(absl::MakeSpan(
      reinterpret_cast<uint8_t*>(pooled_out), BytesOf(laya->dim)));
  if (!pooled_read.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: pooled read failed";
    return kLayaError;
  }
  return 0;
}

int litert_lm_laya_run_act(LiteRtLmLaya* laya, const float* pooled,
                           const float* feats, float* act_logits_out) {
  if (laya == nullptr || pooled == nullptr || feats == nullptr ||
      act_logits_out == nullptr) {
    return kLayaError;
  }
  auto pooled_written =
      laya->act_inputs[0].Write(absl::MakeConstSpan(pooled, laya->dim));
  if (!pooled_written.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: act pooled write failed";
    return kLayaError;
  }
  auto feats_written = laya->act_inputs[1].Write(absl::MakeConstSpan(feats, 4));
  if (!feats_written.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: act feats write failed";
    return kLayaError;
  }
  auto run = laya->act_model->Run(absl::MakeSpan(laya->act_inputs),
                                  absl::MakeSpan(laya->act_outputs));
  if (!run.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: act run failed";
    return kLayaError;
  }
  auto read = laya->act_outputs[0].Read(absl::MakeSpan(
      reinterpret_cast<uint8_t*>(act_logits_out), BytesOf(2)));
  if (!read.HasValue()) {
    ABSL_LOG(ERROR) << "laya_runner: act read failed";
    return kLayaError;
  }
  return 0;
}

void litert_lm_laya_delete(LiteRtLmLaya* laya) { delete laya; }

}  // extern "C"
