// Implementation of c/tflite_embed.h — see the header for the contract.

#include "tflite_embed.h"  // NOLINT

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "support/tokenizer/sentencepiece_tokenizer.h"

namespace {

constexpr int kEmbedError = 1;

int SpecialIdOr(litert::support::SentencePieceTokenizer& tokenizer,
                const char* piece, int fallback) {
  auto id = tokenizer.TokenToId(piece);
  return id.ok() ? *id : fallback;
}

}  // namespace

struct LiteRtLmTfliteEmbed {
  std::unique_ptr<litert::support::SentencePieceTokenizer> tokenizer;
  std::optional<litert::Environment> env;
  std::unique_ptr<litert::CompiledModel> compiled_model;
  std::vector<litert::TensorBuffer> input_buffers;
  std::vector<litert::TensorBuffer> output_buffers;
  size_t seq_len = 0;   // in i32 elements
  size_t dim = 0;       // out f32 elements
  int bos_id = 2;       // Gemma "<bos>" fallback.
};

extern "C" {

LiteRtLmTfliteEmbed* litert_lm_tflite_embed_create(
    const char* tflite_path, const char* sentencepiece_path) {
  if (tflite_path == nullptr || sentencepiece_path == nullptr) {
    ABSL_LOG(ERROR) << "tflite_embed_create: null path argument";
    return nullptr;
  }
  auto embed = std::make_unique<LiteRtLmTfliteEmbed>();

  auto tokenizer =
      litert::support::SentencePieceTokenizer::CreateFromFile(
          sentencepiece_path);
  if (!tokenizer.ok()) {
    ABSL_LOG(ERROR) << "tflite_embed: sentencepiece load failed";
    return nullptr;
  }
  embed->tokenizer = std::move(*tokenizer);
  embed->bos_id = SpecialIdOr(*embed->tokenizer, "<bos>", 2);

  const auto env_options =
      litert::EnvironmentOptions(
          absl::Span<const litert::EnvironmentOptions::Option>{});
  auto env = litert::Environment::Create(env_options);
  if (!env.HasValue()) {
    ABSL_LOG(ERROR) << "tflite_embed: environment create failed";
    return nullptr;
  }
  embed->env.emplace(std::move(*env));

  auto compiled = litert::CompiledModel::Create(
      *embed->env, std::string(tflite_path), litert::HwAccelerators::kCpu);
  if (!compiled.HasValue()) {
    ABSL_LOG(ERROR) << "tflite_embed: compiled model create failed: "
                    << compiled.Error().Message();
    return nullptr;
  }
  embed->compiled_model =
      std::make_unique<litert::CompiledModel>(std::move(*compiled));

  auto inputs = embed->compiled_model->CreateInputBuffers();
  if (!inputs.HasValue() || inputs->empty()) {
    ABSL_LOG(ERROR) << "tflite_embed: input buffers failed";
    return nullptr;
  }
  embed->input_buffers = std::move(*inputs);

  auto outputs = embed->compiled_model->CreateOutputBuffers();
  if (!outputs.HasValue() || outputs->empty()) {
    ABSL_LOG(ERROR) << "tflite_embed: output buffers failed";
    return nullptr;
  }
  embed->output_buffers = std::move(*outputs);

  auto out_size = embed->output_buffers[0].Size();
  if (!out_size.HasValue() || *out_size % sizeof(float) != 0 || *out_size == 0) {
    ABSL_LOG(ERROR) << "tflite_embed: unexpected output buffer size";
    return nullptr;
  }
  embed->dim = *out_size / sizeof(float);

  auto in_size = embed->input_buffers[0].Size();
  if (!in_size.HasValue() || *in_size % sizeof(int) != 0 ||
      *in_size < 2 * sizeof(int)) {
    ABSL_LOG(ERROR) << "tflite_embed: unexpected input buffer size";
    return nullptr;
  }
  embed->seq_len = *in_size / sizeof(int);

  ABSL_VLOG(1) << "tflite_embed ready: seq=" << embed->seq_len
               << " dim=" << embed->dim;
  return embed.release();
}

int litert_lm_tflite_embed_dim(const LiteRtLmTfliteEmbed* embed) {
  return embed == nullptr ? 0 : static_cast<int>(embed->dim);
}

int litert_lm_tflite_embed_text(LiteRtLmTfliteEmbed* embed, const char* text,
                                float* out) {
  if (embed == nullptr || text == nullptr || out == nullptr) {
    return kEmbedError;
  }

  auto ids = embed->tokenizer->TextToTokenIds(text);
  if (!ids.ok()) {
    ABSL_LOG(ERROR) << "tflite_embed: tokenize failed";
    return kEmbedError;
  }

  // BOS + tokens (truncated), right-padded with 0 (<pad>) to the graph's
  // static length.
  std::vector<int> batch(embed->seq_len, 0);
  const size_t budget = embed->seq_len - 1;
  batch[0] = embed->bos_id;
  for (size_t i = 0; i < ids->size() && i < budget; ++i) {
    batch[i + 1] = (*ids)[i];
  }

  auto written = embed->input_buffers[0].Write(
      absl::MakeConstSpan(batch.data(), batch.size()));
  if (!written.HasValue()) {
    ABSL_LOG(ERROR) << "tflite_embed: input write failed";
    return kEmbedError;
  }

  auto run = embed->compiled_model->Run(
      absl::MakeSpan(embed->input_buffers),
      absl::MakeSpan(embed->output_buffers));
  if (!run.HasValue()) {
    ABSL_LOG(ERROR) << "tflite_embed: run failed";
    return kEmbedError;
  }

  auto read = embed->output_buffers[0].Read(absl::MakeSpan(
      reinterpret_cast<uint8_t*>(out), embed->dim * sizeof(float)));
  if (!read.HasValue()) {
    ABSL_LOG(ERROR) << "tflite_embed: output read failed";
    return kEmbedError;
  }
  return 0;
}

void litert_lm_tflite_embed_delete(LiteRtLmTfliteEmbed* embed) {
  delete embed;
}

}  // extern "C"
