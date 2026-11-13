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

#include "omni/any2any/lfm2_audio/lfm2_audio_stage.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/lfm2_audio/lfm2_audio_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::any2any {
namespace {

constexpr int kSupportedNfft = 512;
constexpr int kNumFreqBins = kSupportedNfft / 2 + 1;  // 257
constexpr float kPerFeatureNormEps = 1e-5f;

// 512-point radix-2 real-to-complex FFT returning the first 257 bins
// (`0..N/2`), matching `torch.fft.rfft(..., n=512)`.
void RealFft512(const std::array<float, kSupportedNfft>& input,
                std::array<std::complex<float>, kNumFreqBins>& output) {
  std::array<std::complex<double>, kSupportedNfft> buf{};
  for (int i = 0; i < kSupportedNfft; ++i) {
    buf[i] = std::complex<double>(static_cast<double>(input[i]), 0.0);
  }

  for (int i = 1, j = 0; i < kSupportedNfft; ++i) {
    int bit = kSupportedNfft >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(buf[i], buf[j]);
    }
  }

  const double pi = std::acos(-1.0);
  for (int len = 2; len <= kSupportedNfft; len <<= 1) {
    const double ang = -2.0 * pi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    const int half = len >> 1;
    for (int i = 0; i < kSupportedNfft; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int j = 0; j < half; ++j) {
        const std::complex<double> u = buf[i + j];
        const std::complex<double> v = buf[i + j + half] * w;
        buf[i + j] = u + v;
        buf[i + j + half] = u - v;
        w *= wlen;
      }
    }
  }

  for (int k = 0; k < kNumFreqBins; ++k) {
    output[k] = std::complex<float>(static_cast<float>(buf[k].real()),
                                    static_cast<float>(buf[k].imag()));
  }
}

double HzToSlaneyMel(double hz) {
  constexpr double kFSp = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFSp;
  const double logstep = std::log(6.4) / 27.0;
  if (hz >= kMinLogHz) {
    return kMinLogMel + std::log(hz / kMinLogHz) / logstep;
  }
  return hz / kFSp;
}

double SlaneyMelToHz(double mel) {
  constexpr double kFSp = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFSp;
  const double logstep = std::log(6.4) / 27.0;
  if (mel >= kMinLogMel) {
    return kMinLogHz * std::exp(logstep * (mel - kMinLogMel));
  }
  return kFSp * mel;
}

// Computes a Slaney-normalized mel filterbank matrix of shape
// `[n_mels, n_fft / 2 + 1]`, matching `librosa.filters.mel(..., norm="slaney")`
// used by NeMo `FilterbankFeatures`.
std::vector<float> ComputeSlaneyMelFilterbank(int n_mels, int n_fft,
                                              float sample_rate, float f_min,
                                              float f_max) {
  const int n_freqs = n_fft / 2 + 1;
  std::vector<float> weights(static_cast<size_t>(n_mels) * n_freqs, 0.0f);

  std::vector<double> fft_freqs(n_freqs, 0.0);
  for (int k = 0; k < n_freqs; ++k) {
    fft_freqs[k] =
        static_cast<double>(k) * static_cast<double>(sample_rate) / n_fft;
  }

  const double mel_min = HzToSlaneyMel(f_min);
  const double mel_max = HzToSlaneyMel(f_max);
  std::vector<double> f_pts(n_mels + 2, 0.0);
  for (int i = 0; i < n_mels + 2; ++i) {
    const double mel =
        mel_min + (mel_max - mel_min) * static_cast<double>(i) / (n_mels + 1);
    f_pts[i] = SlaneyMelToHz(mel);
  }

  for (int m = 0; m < n_mels; ++m) {
    const double f_left = f_pts[m];
    const double f_center = f_pts[m + 1];
    const double f_right = f_pts[m + 2];
    const double enorm = 2.0 / std::max(1e-12, f_right - f_left);
    float* row = weights.data() + static_cast<size_t>(m) * n_freqs;
    for (int k = 0; k < n_freqs; ++k) {
      const double lower =
          (fft_freqs[k] - f_left) / std::max(1e-12, f_center - f_left);
      const double upper =
          (f_right - fft_freqs[k]) / std::max(1e-12, f_right - f_center);
      const double w = std::max(0.0, std::min(lower, upper));
      row[k] = static_cast<float>(w * enorm);
    }
  }
  return weights;
}

// Builds a symmetric Hann window (`periodic=False`) of length `win_length`
// centered inside a buffer of length `n_fft`, matching `torch.hann_window(400,
// periodic=False)` inside `torch.stft(..., n_fft=512, win_length=400)`.
std::array<float, kSupportedNfft> BuildCenteredSymmetricHannWindow(
    int win_length) {
  std::array<float, kSupportedNfft> window{};
  window.fill(0.0f);
  const int clamped_win = std::clamp(win_length, 1, kSupportedNfft);
  const int pad_left = (kSupportedNfft - clamped_win) / 2;
  if (clamped_win == 1) {
    window[pad_left] = 1.0f;
    return window;
  }
  const double two_pi = 2.0 * std::acos(-1.0);
  const double denom = static_cast<double>(clamped_win - 1);
  for (int i = 0; i < clamped_win; ++i) {
    window[pad_left + i] =
        static_cast<float>(0.5 - 0.5 * std::cos(two_pi * i / denom));
  }
  return window;
}

absl::Status ValidateTokenIdsInVocab(absl::Span<const int> token_ids,
                                     int vocab_size,
                                     absl::string_view caller_name) {
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        caller_name, ": llm_vocab_size must be positive, got ", vocab_size));
  }
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int id = token_ids[i];
    if (id < 0 || id >= vocab_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          caller_name, ": token_id at index ", i, " (", id,
          ") is out of valid vocabulary range [0, ", vocab_size, ")."));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> EmbedTokenSequenceWithRunner(
    absl::Span<const int> token_ids, int vocab_size, int hidden_dim,
    LiteRtRunner* absl_nullable runner,
    std::vector<TensorBuffer>& input_buffers,
    std::vector<TensorBuffer>& output_buffers, absl::string_view caller_name) {
  ABSL_RETURN_IF_ERROR(
      ValidateTokenIdsInVocab(token_ids, vocab_size, caller_name));
  if (token_ids.empty()) {
    return std::vector<float>();
  }
  if (runner == nullptr || input_buffers.empty() || output_buffers.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat(caller_name, ": text_embedder_runner must not be null."));
  }

  std::vector<float> out(token_ids.size() * hidden_dim, 0.0f);
  std::vector<float> step_emb(hidden_dim, 0.0f);
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int32_t tok = static_cast<int32_t>(token_ids[i]);
    LITERT_RETURN_IF_ERROR(
        input_buffers[0].Write<int32_t>(absl::MakeConstSpan(&tok, 1)));
    ABSL_RETURN_IF_ERROR(
        runner->Run(/*signature_name=*/"", input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(
        output_buffers[0].Read<float>(absl::MakeSpan(step_emb)));
    std::copy(step_emb.begin(), step_emb.end(), out.begin() + i * hidden_dim);
  }
  return out;
}

absl::StatusOr<std::vector<float>> ReadTensorBufferAsFloatVector(
    TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (tensor_type.ElementType() == ElementType::Float32) {
    std::vector<float> out(num_elements);
    LITERT_RETURN_IF_ERROR(tensor_buffer.Read<float>(absl::MakeSpan(out)));
    return out;
  }
  return absl::InvalidArgumentError(
      "Expected Float32 TensorBuffer from LiteRtLmRunner::Decode.");
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithText(
    const TensorBuffer& text_tokens) {
  LITERT_ASSIGN_OR_RETURN(TensorBuffer text_tokens_dup,
                          text_tokens.Duplicate());
  return lm::ExecutorInputs(lm::ExecutorTextData(std::move(text_tokens_dup)),
                            /*vision_data=*/std::nullopt,
                            /*audio_data=*/std::nullopt);
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithAudio(
    const TensorBuffer& audio_embeddings) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          audio_embeddings.TensorType());
  const auto& dims = tensor_type.Layout().Dimensions();
  if (dims.size() != 3 || dims[0] != 1) {
    return absl::InvalidArgumentError(
        "Audio embeddings tensor must have shape [1, seq_len, hidden_dim].");
  }
  const int seq_len = dims[1];
  std::vector<int32_t> placeholder_tokens(seq_len,
                                          lm::ExecutorAudioData::kSpecialToken);
  LITERT_ASSIGN_OR_RETURN(TensorBuffer text_tokens,
                          support::CreateTensorBuffer<int32_t>({1, seq_len}));
  LITERT_RETURN_IF_ERROR(
      text_tokens.Write<int32_t>(absl::MakeConstSpan(placeholder_tokens)));
  LITERT_ASSIGN_OR_RETURN(TensorBuffer audio_dup, audio_embeddings.Duplicate());
  return lm::ExecutorInputs(
      lm::ExecutorTextData(std::move(text_tokens)),
      /*vision_data=*/std::nullopt,
      lm::ExecutorAudioData(std::move(audio_dup),
                            /*per_layer_embeddings=*/std::nullopt));
}

struct DecodeStepOutput {
  std::vector<float> logits;
  std::vector<float> hidden_state;
};

absl::StatusOr<DecodeStepOutput> ExtractDecodeStepOutput(
    TensorBuffer& decode_buf, int vocab_size, int hidden_dim,
    bool require_hidden_state) {
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  if (raw_out.empty()) {
    return absl::InternalError("LLM Decode returned an empty TensorBuffer.");
  }
  DecodeStepOutput step_out;
  const int raw_len = static_cast<int>(raw_out.size());
  if (raw_len == vocab_size + hidden_dim) {
    step_out.hidden_state.assign(raw_out.begin() + vocab_size, raw_out.end());
    raw_out.resize(vocab_size);
  } else if (raw_len == vocab_size) {
    if (require_hidden_state) {
      return absl::FailedPreconditionError(
          "LFM2 Decode output contains only vocabulary logits without "
          "last_hidden_state required for Depthformer audio generation.");
    }
    step_out.hidden_state.assign(hidden_dim, 0.0f);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("LFM2 Decode output size (", raw_len,
                     ") does not match llm_vocab_size (", vocab_size,
                     ") or llm_vocab_size + llm_hidden_size (",
                     vocab_size + hidden_dim, ")."));
  }
  step_out.logits = std::move(raw_out);
  return step_out;
}

int SampleFromLogits(absl::Span<const float> logits, float temperature,
                     int top_k, float top_p, bool do_sample,
                     std::mt19937_64& rng) {
  if (logits.empty()) {
    return 0;
  }
  if (!do_sample || temperature <= 0.0f) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const double temp = std::max<double>(temperature, 1e-6);
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i]) || logits[i] <= -1e8f) {
      continue;
    }
    const double p = std::exp((logits[i] - max_logit) / temp);
    candidates.push_back({p, static_cast<int>(i)});
  }
  if (candidates.empty()) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const int k =
      (top_k > 0) ? std::min<int>(top_k, candidates.size()) : candidates.size();
  std::partial_sort(
      candidates.begin(), candidates.begin() + k, candidates.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });
  candidates.resize(k);

  if (top_p > 0.0f && top_p < 1.0f && candidates.size() > 1) {
    double total_p = 0.0;
    for (const auto& c : candidates) {
      total_p += c.first;
    }
    if (total_p > 0.0) {
      double cumulative = 0.0;
      size_t keep = 0;
      for (; keep < candidates.size(); ++keep) {
        cumulative += candidates[keep].first / total_p;
        if (cumulative >= top_p) {
          ++keep;
          break;
        }
      }
      candidates.resize(std::max<size_t>(1, keep));
    }
  }

  double sum_p = 0.0;
  for (const auto& c : candidates) {
    sum_p += c.first;
  }
  if (sum_p <= 0.0) {
    return candidates.front().second;
  }
  std::uniform_real_distribution<double> dist(0.0, sum_p);
  double r = dist(rng);
  double acc = 0.0;
  for (const auto& c : candidates) {
    acc += c.first;
    if (r <= acc) {
      return c.second;
    }
  }
  return candidates.back().second;
}

}  // namespace

absl::StatusOr<std::vector<float>> ExtractNemoLogMelSpectrogram(
    absl::Span<const float> pcm_16k, const Lfm2AudioModelConfig& config,
    int* absl_nonnull out_valid_frames) {
  if (config.audio_mel_bins <= 0 || config.audio_hop_length <= 0 ||
      config.audio_win_length <= 0 ||
      config.audio_win_length > kSupportedNfft ||
      config.audio_n_fft != kSupportedNfft ||
      config.audio_input_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "Invalid Lfm2AudioModelConfig parameters for NeMo log-mel extraction.");
  }

  // 1. Pre-emphasis filter: y[0] = x[0], y[t] = x[t] - preemph * x[t - 1].
  std::vector<float> preemph_pcm(pcm_16k.size(), 0.0f);
  if (!pcm_16k.empty()) {
    preemph_pcm[0] = pcm_16k[0];
    for (size_t i = 1; i < pcm_16k.size(); ++i) {
      preemph_pcm[i] = pcm_16k[i] - config.audio_preemph * pcm_16k[i - 1];
    }
  }

  // 2. Center zero-padding (`n_fft // 2 = 256` on both sides).
  // Per NeMo `FilterbankFeatures.forward` (`processor.py:489-501`):
  // `seq_len = get_seq_len(N) = floor(N / hop_length)` frames are normalized
  // over `0 .. seq_len - 1`, while `torch.stft(..., center=True)` produces
  // `floor(N / hop_length) + 1` frames and `FilterbankFeatures` masks frame
  // `seq_len` (`index >= seq_len`) to `0.0f`.
  const int pad = kSupportedNfft / 2;
  const int norm_frames =
      preemph_pcm.empty()
          ? 1
          : std::max<int>(1, static_cast<int>(preemph_pcm.size()) /
                                 config.audio_hop_length);
  const int total_stft_frames =
      (preemph_pcm.size() >= static_cast<size_t>(config.audio_hop_length))
          ? (norm_frames + 1)
          : norm_frames;
  *out_valid_frames = total_stft_frames;

  const size_t min_padded_size =
      static_cast<size_t>(norm_frames - 1) * config.audio_hop_length +
      kSupportedNfft;
  std::vector<float> padded(
      std::max<size_t>(min_padded_size, preemph_pcm.size() + 2 * pad), 0.0f);
  if (!preemph_pcm.empty()) {
    std::copy(preemph_pcm.begin(), preemph_pcm.end(), padded.begin() + pad);
  }

  // 3. Windowed STFT power spectrum + Slaney mel filterbank + log(mel + 2^-24).
  const std::array<float, kSupportedNfft> hann_window =
      BuildCenteredSymmetricHannWindow(config.audio_win_length);
  const std::vector<float> mel_basis = ComputeSlaneyMelFilterbank(
      config.audio_mel_bins, kSupportedNfft,
      static_cast<float>(config.audio_input_sample_rate_hz), 0.0f,
      static_cast<float>(config.audio_input_sample_rate_hz) * 0.5f);

  const int n_mels = config.audio_mel_bins;
  const float log_guard = std::max(config.audio_log_zero_guard, 1e-30f);
  std::vector<float> log_mel(static_cast<size_t>(total_stft_frames) * n_mels,
                             0.0f);

  std::array<float, kSupportedNfft> windowed_frame{};
  std::array<std::complex<float>, kNumFreqBins> spectrum{};
  std::array<float, kNumFreqBins> power_spec{};

  for (int t = 0; t < norm_frames; ++t) {
    const size_t start = static_cast<size_t>(t) * config.audio_hop_length;
    for (int i = 0; i < kSupportedNfft; ++i) {
      windowed_frame[i] = padded[start + i] * hann_window[i];
    }
    RealFft512(windowed_frame, spectrum);
    for (int k = 0; k < kNumFreqBins; ++k) {
      const float re = spectrum[k].real();
      const float im = spectrum[k].imag();
      power_spec[k] = re * re + im * im;
    }
    float* frame_out = log_mel.data() + static_cast<size_t>(t) * n_mels;
    for (int m = 0; m < n_mels; ++m) {
      const float* filter =
          mel_basis.data() + static_cast<size_t>(m) * kNumFreqBins;
      float mel_energy = 0.0f;
      for (int k = 0; k < kNumFreqBins; ++k) {
        mel_energy += filter[k] * power_spec[k];
      }
      frame_out[m] = std::log(mel_energy + log_guard);
    }
  }

  // 4. NeMo `normalize = "per_feature"` over valid frames `0 .. norm_frames -
  // 1` with Bessel sample std (`denom = norm_frames - 1` when `norm_frames >
  // 1`) and `+ 1e-5f` added to `std`. Any trailing STFT frame (`t ==
  // norm_frames`) remains `0.0f` as masked by `FilterbankFeatures`.
  if (norm_frames <= 1) {
    std::fill(log_mel.begin(), log_mel.end(), 0.0f);
    return log_mel;
  }

  for (int m = 0; m < n_mels; ++m) {
    double sum = 0.0;
    for (int t = 0; t < norm_frames; ++t) {
      sum += log_mel[static_cast<size_t>(t) * n_mels + m];
    }
    const double mean = sum / static_cast<double>(norm_frames);
    double sq_sum = 0.0;
    for (int t = 0; t < norm_frames; ++t) {
      const double diff = log_mel[static_cast<size_t>(t) * n_mels + m] - mean;
      sq_sum += diff * diff;
    }
    const double std_val =
        std::sqrt(sq_sum / static_cast<double>(norm_frames - 1));
    const float inv_denom =
        static_cast<float>(1.0 / (std_val + kPerFeatureNormEps));
    for (int t = 0; t < norm_frames; ++t) {
      float& val = log_mel[static_cast<size_t>(t) * n_mels + m];
      val = static_cast<float>((val - mean) * inv_denom);
    }
  }

  return log_mel;
}

absl::StatusOr<std::vector<int>> TokenizeLfm2AudioTextWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text,
    const Lfm2AudioModelConfig& config) {
  if (text.empty()) {
    return std::vector<int>();
  }

  struct SpecialEntry {
    absl::string_view literal;
    int fallback_id;
  };
  const std::array<SpecialEntry, 6> specials = {{
      {lfm2_audio::kPadToken, config.pad_token_id},
      {lfm2_audio::kBosToken, config.bos_token_id},
      {lfm2_audio::kImStartToken, config.im_start_token_id},
      {lfm2_audio::kImEndToken, config.im_end_token_id},
      {lfm2_audio::kAudioStartToken, config.audio_start_token_id},
      {lfm2_audio::kTextEndToken, config.text_end_token_id},
  }};

  std::vector<int> token_ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best_pos = absl::string_view::npos;
    size_t best_idx = 0;
    for (size_t i = 0; i < specials.size(); ++i) {
      const size_t found = text.find(specials[i].literal, pos);
      if (found != absl::string_view::npos &&
          (best_pos == absl::string_view::npos || found < best_pos)) {
        best_pos = found;
        best_idx = i;
      }
    }

    if (best_pos == absl::string_view::npos) {
      absl::string_view rem = text.substr(pos);
      ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                            tokenizer.TextToTokenIds(rem));
      token_ids.insert(token_ids.end(), ids.begin(), ids.end());
      break;
    }

    if (best_pos > pos) {
      absl::string_view prefix = text.substr(pos, best_pos - pos);
      ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                            tokenizer.TextToTokenIds(prefix));
      token_ids.insert(token_ids.end(), ids.begin(), ids.end());
    }

    const auto& sp = specials[best_idx];
    auto tok_id = tokenizer.TokenToId(sp.literal);
    token_ids.push_back(tok_id.ok() ? *tok_id : sp.fallback_id);
    pos = best_pos + sp.literal.size();
  }

  return token_ids;
}

// =============================================================================
// Lfm2AudioInputFanOutStage
// =============================================================================

Lfm2AudioInputFanOutStage::Lfm2AudioInputFanOutStage(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_audio_encoder)
    : FanOutStage<OmniSession::Input>(upstream),
      text_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, TextInput>>())),
      audio_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, AudioInput>>())),
      audio_input_metadata_queue_(AddOutputQueue(
          std::make_unique<SequencedVariantQueue<OmniSession::Input,
                                                 AudioInputMetadata>>())),
      end_of_turn_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>>())),
      has_audio_encoder_(has_audio_encoder) {}

Lfm2AudioInputFanOutStage::~Lfm2AudioInputFanOutStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioInputFanOutStage>>
Lfm2AudioInputFanOutStage::Create(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_audio_encoder) {
  return absl::WrapUnique(
      new Lfm2AudioInputFanOutStage(upstream, has_audio_encoder));
}

absl::Status Lfm2AudioInputFanOutStage::RouteInput(OmniSession::Input input) {
  if (std::holds_alternative<EndOfInput>(input)) {
    if (has_open_turn_) {
      has_open_turn_ = false;
      const uint64_t seqnum = GetNextSeqnum();
      return end_of_turn_queue_->PushBack(seqnum, EndOfInputTurn{});
    }
    return absl::OkStatus();
  }
  if (std::holds_alternative<ImageInput>(input) ||
      std::holds_alternative<ImageGenInputMetadata>(input)) {
    return absl::InvalidArgumentError(
        "LiquidAI/LFM2.5-Audio-1.5B does not support visual (image) inputs.");
  }
  if (std::holds_alternative<AudioInputMetadata>(input)) {
    if (!has_audio_encoder_) {
      return absl::FailedPreconditionError(
          "AudioInputMetadata is not supported because the model bundle has no "
          "audio_encoder section.");
    }
    return audio_input_metadata_queue_->PushBack(/*seqnum=*/0,
                                                 std::move(input));
  }
  if (std::holds_alternative<AudioInput>(input) && !has_audio_encoder_) {
    return absl::FailedPreconditionError(
        "AudioInput is not supported because the model bundle has no "
        "audio_encoder section.");
  }
  const uint64_t seqnum = GetNextSeqnum();
  if (std::holds_alternative<TextInput>(input)) {
    has_open_turn_ = true;
    return text_queue_->PushBack(seqnum, std::move(input));
  }
  if (std::holds_alternative<AudioInput>(input)) {
    has_open_turn_ = true;
    return audio_queue_->PushBack(seqnum, std::move(input));
  }
  if (std::holds_alternative<EndOfInputTurn>(input)) {
    has_open_turn_ = false;
    return end_of_turn_queue_->PushBack(seqnum, std::move(input));
  }
  return absl::InvalidArgumentError(
      "Unsupported input variant in Lfm2AudioInputFanOutStage.");
}

void Lfm2AudioInputFanOutStage::ResetInternal() {
  FanOutStage<OmniSession::Input>::ResetInternal();
  has_open_turn_ = false;
}

// =============================================================================
// Lfm2AudioStage
// =============================================================================

Lfm2AudioStage::Lfm2AudioStage(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    int chunk_frames, bool feature_major_input,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    std::vector<TensorBuffer> audio_enc_input_buffers,
    std::vector<TensorBuffer> audio_enc_output_buffers,
    const Lfm2AudioModelConfig& config)
    : audio_queue_(*audio_queue),
      audio_input_metadata_queue_(*audio_input_metadata_queue),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      mel_bins_(config_.audio_mel_bins),
      chunk_frames_(chunk_frames),
      feature_major_input_(feature_major_input),
      input_sample_rate_hz_(config_.audio_input_sample_rate_hz),
      audio_encoder_runner_(std::move(audio_encoder_runner)),
      audio_enc_input_buffers_(std::move(audio_enc_input_buffers)),
      audio_enc_output_buffers_(std::move(audio_enc_output_buffers)) {}

Lfm2AudioStage::~Lfm2AudioStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioStage>> Lfm2AudioStage::Create(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.audio_mel_bins <= 0 ||
      config.audio_input_sample_rate_hz <= 0 ||
      config.audio_subsampling_factor <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage config dimensions and sample rate must be positive.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto audio_enc_input_buffers,
      audio_encoder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      auto audio_enc_output_buffers,
      audio_encoder_runner->CreateOutputBuffers(/*signature_name=*/""));
  if (audio_enc_input_buffers.empty() || audio_enc_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage audio_encoder_runner must have at least 1 input and 1 "
        "output buffer.");
  }

  LITERT_ASSIGN_OR_RETURN(auto in_type,
                          audio_enc_input_buffers[0].TensorType());
  const auto& in_dims = in_type.Layout().Dimensions();
  LITERT_ASSIGN_OR_RETURN(size_t in_elements, in_type.Layout().NumElements());
  if (in_elements == 0 || in_elements % config.audio_mel_bins != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Lfm2AudioStage audio_encoder_runner input size (", in_elements,
        ") is not divisible by audio_mel_bins (", config.audio_mel_bins, ")."));
  }

  // `Lfm2ConformerEncoderModule` takes `[1, n_mels (128), T_mel]`
  // (`feature_major_input = true`). Also support `[1, T_mel, n_mels]` if a test
  // runner uses time-major layout.
  bool feature_major_input = true;
  if (in_dims.size() == 3) {
    if (in_dims[1] == config.audio_mel_bins &&
        in_dims[2] != config.audio_mel_bins) {
      feature_major_input = true;
    } else if (in_dims[2] == config.audio_mel_bins &&
               in_dims[1] != config.audio_mel_bins) {
      feature_major_input = false;
    }
  }
  const int chunk_frames =
      static_cast<int>(in_elements / config.audio_mel_bins);

  LITERT_ASSIGN_OR_RETURN(auto out_type,
                          audio_enc_output_buffers[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t out_elements, out_type.Layout().NumElements());
  if (out_elements == 0 || out_elements % config.llm_hidden_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Lfm2AudioStage audio_encoder_runner output size (",
                     out_elements, ") is not divisible by llm_hidden_size (",
                     config.llm_hidden_size, ")."));
  }

  return absl::WrapUnique(
      new Lfm2AudioStage(audio_queue, audio_input_metadata_queue, chunk_frames,
                         feature_major_input, std::move(audio_encoder_runner),
                         std::move(audio_enc_input_buffers),
                         std::move(audio_enc_output_buffers), config));
}

bool Lfm2AudioStage::NeedScheduleInternal() const {
  return !audio_queue_.Empty() || !audio_input_metadata_queue_.Empty();
}

absl::Status Lfm2AudioStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("No audio items ready to schedule.");
  }

  while (!audio_input_metadata_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<AudioInputMetadata> meta,
                          audio_input_metadata_queue_.PopFront());
    if (meta.item.sample_rate_hz <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "AudioInputMetadata.sample_rate_hz must be positive, got ",
          meta.item.sample_rate_hz));
    }
    input_sample_rate_hz_ = meta.item.sample_rate_hz;
  }

  while (!audio_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<AudioInput> seq_audio,
                          audio_queue_.PopFront());
    ABSL_ASSIGN_OR_RETURN(Lfm2AudioEmbeddingSegment seg,
                          ProcessAudio(seq_audio.item));
    PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
        .seqnum = seq_audio.seqnum,
        .item = std::move(seg),
    });
  }

  return absl::OkStatus();
}

void Lfm2AudioStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  resampler_.Reset();
  input_sample_rate_hz_ = config_.audio_input_sample_rate_hz;
}

absl::StatusOr<Lfm2AudioEmbeddingSegment> Lfm2AudioStage::ProcessAudio(
    const AudioInput& audio) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> resampled,
      resampler_.Resample(audio.pcm_samples, input_sample_rate_hz_,
                          config_.audio_input_sample_rate_hz));

  int valid_frames = 0;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> log_mel_time_major,
      ExtractNemoLogMelSpectrogram(resampled, config_, &valid_frames));

  LITERT_ASSIGN_OR_RETURN(auto out_type,
                          audio_enc_output_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t out_elements, out_type.Layout().NumElements());
  const int max_chunk_tokens = static_cast<int>(out_elements / hidden_dim_);

  Lfm2AudioEmbeddingSegment seg;
  std::vector<float> chunk_mel(static_cast<size_t>(chunk_frames_) * mel_bins_,
                               0.0f);
  std::vector<float> chunk_out(out_elements, 0.0f);

  const int total_frames = std::max(1, valid_frames);
  for (int start_frame = 0; start_frame < total_frames;
       start_frame += chunk_frames_) {
    const int actual_frames =
        std::min(chunk_frames_, total_frames - start_frame);
    std::fill(chunk_mel.begin(), chunk_mel.end(), 0.0f);

    if (feature_major_input_) {
      // Transpose `[actual_frames, mel_bins_]` -> `[mel_bins_, chunk_frames_]`.
      for (int t = 0; t < actual_frames; ++t) {
        const float* src = log_mel_time_major.data() +
                           static_cast<size_t>(start_frame + t) * mel_bins_;
        for (int m = 0; m < mel_bins_; ++m) {
          chunk_mel[static_cast<size_t>(m) * chunk_frames_ + t] = src[m];
        }
      }
    } else {
      const size_t copy_floats = static_cast<size_t>(actual_frames) * mel_bins_;
      std::copy_n(log_mel_time_major.data() +
                      static_cast<size_t>(start_frame) * mel_bins_,
                  copy_floats, chunk_mel.data());
    }

    LITERT_RETURN_IF_ERROR(audio_enc_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(chunk_mel)));
    if (audio_enc_input_buffers_.size() >= 2) {
      const int32_t len_val = static_cast<int32_t>(actual_frames);
      LITERT_RETURN_IF_ERROR(audio_enc_input_buffers_[1].Write<int32_t>(
          absl::MakeConstSpan(&len_val, 1)));
    }

    ABSL_RETURN_IF_ERROR(audio_encoder_runner_->Run(/*signature_name=*/"",
                                                    audio_enc_input_buffers_,
                                                    audio_enc_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        audio_enc_output_buffers_[0].Read<float>(absl::MakeSpan(chunk_out)));

    int valid_tokens = (actual_frames + config_.audio_subsampling_factor - 1) /
                       config_.audio_subsampling_factor;
    if (audio_enc_output_buffers_.size() >= 2) {
      int32_t reported_tokens = 0;
      if (audio_enc_output_buffers_[1].Read<int32_t>(
              absl::MakeSpan(&reported_tokens, 1))) {
        if (reported_tokens > 0) {
          valid_tokens = reported_tokens;
        }
      }
    }
    valid_tokens = std::clamp(valid_tokens, 1, max_chunk_tokens);

    seg.embeddings.insert(
        seg.embeddings.end(), chunk_out.begin(),
        chunk_out.begin() + static_cast<size_t>(valid_tokens) * hidden_dim_);
    seg.token_ids.insert(seg.token_ids.end(), valid_tokens,
                         config_.audio_embed_placeholder_token_id);
    seg.num_tokens += valid_tokens;
  }

  return seg;
}

// =============================================================================
// Lfm2AudioTextStage
// =============================================================================

Lfm2AudioTextStage::Lfm2AudioTextStage(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    std::vector<TensorBuffer> embed_input_buffers,
    std::vector<TensorBuffer> embed_output_buffers,
    const Lfm2AudioModelConfig& config)
    : text_queue_(*text_queue),
      end_of_turn_queue_(*end_of_turn_queue),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      embed_input_buffers_(std::move(embed_input_buffers)),
      embed_output_buffers_(std::move(embed_output_buffers)) {}

Lfm2AudioTextStage::~Lfm2AudioTextStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioTextStage>> Lfm2AudioTextStage::Create(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioTextStage config.llm_hidden_size and config.llm_vocab_size "
        "must be positive.");
  }

  std::vector<TensorBuffer> embed_inputs;
  std::vector<TensorBuffer> embed_outputs;
  if (text_embedder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        embed_inputs,
        text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        embed_outputs,
        text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (embed_inputs.size() != 1 || embed_outputs.size() != 1) {
      return absl::InvalidArgumentError(
          "Lfm2AudioTextStage text_embedder_runner must have 1 input and 1 "
          "output buffer.");
    }
    LITERT_ASSIGN_OR_RETURN(auto out_type, embed_outputs[0].TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t out_elements,
                            out_type.Layout().NumElements());
    if (out_elements != static_cast<size_t>(config.llm_hidden_size)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Lfm2AudioTextStage text_embedder_runner output size (", out_elements,
          ") does not match llm_hidden_size (", config.llm_hidden_size, ")."));
    }
  }

  return absl::WrapUnique(new Lfm2AudioTextStage(
      text_queue, end_of_turn_queue, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(embed_inputs),
      std::move(embed_outputs), config));
}

bool Lfm2AudioTextStage::NeedScheduleInternal() const {
  return !text_queue_.Empty() || !end_of_turn_queue_.Empty();
}

absl::Status Lfm2AudioTextStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError(
        "No text or end-of-turn items ready to schedule.");
  }

  while (!text_queue_.Empty() || !end_of_turn_queue_.Empty()) {
    bool take_text = false;
    if (!text_queue_.Empty() && !end_of_turn_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* t_peek,
                            text_queue_.PeekFront());
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInputTurn>* e_peek,
                            end_of_turn_queue_.PeekFront());
      take_text = (t_peek->seqnum < e_peek->seqnum);
    } else {
      take_text = !text_queue_.Empty();
    }

    if (take_text) {
      ABSL_ASSIGN_OR_RETURN(Sequenced<TextInput> seq_text,
                            text_queue_.PopFront());
      Lfm2AudioEmbeddingSegment seg;
      if (!seq_text.item.text.empty()) {
        ABSL_ASSIGN_OR_RETURN(seg.token_ids,
                              TokenizeLfm2AudioTextWithSpecialTokens(
                                  *tokenizer_, seq_text.item.text, config_));
        seg.num_tokens = static_cast<int>(seg.token_ids.size());
        ABSL_ASSIGN_OR_RETURN(seg.embeddings, EmbedTokens(seg.token_ids));
      }
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seq_text.seqnum,
          .item = std::move(seg),
      });
    } else {
      ABSL_ASSIGN_OR_RETURN(Sequenced<EndOfInputTurn> seq_eot,
                            end_of_turn_queue_.PopFront());
      Lfm2AudioEmbeddingSegment seg;
      seg.is_end_of_turn = true;
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seq_eot.seqnum,
          .item = std::move(seg),
      });
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Lfm2AudioTextStage::EmbedTokens(
    absl::Span<const int> token_ids) {
  return EmbedTokenSequenceWithRunner(
      token_ids, config_.llm_vocab_size, hidden_dim_,
      text_embedder_runner_.get(), embed_input_buffers_, embed_output_buffers_,
      "Lfm2AudioTextStage");
}

// =============================================================================
// Lfm2AudioFrontendStage
// =============================================================================

Lfm2AudioFrontendStage::Lfm2AudioFrontendStage(
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    std::vector<TensorBuffer> embed_input_buffers,
    std::vector<TensorBuffer> embed_output_buffers,
    const Lfm2AudioModelConfig& config)
    : text_stage_(*text_stage),
      audio_stage_(audio_stage),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      embed_input_buffers_(std::move(embed_input_buffers)),
      embed_output_buffers_(std::move(embed_output_buffers)) {}

Lfm2AudioFrontendStage::~Lfm2AudioFrontendStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioFrontendStage>>
Lfm2AudioFrontendStage::Create(
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendStage config.llm_hidden_size and "
        "config.llm_vocab_size must be positive.");
  }

  std::vector<TensorBuffer> embed_inputs;
  std::vector<TensorBuffer> embed_outputs;
  if (text_embedder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        embed_inputs,
        text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        embed_outputs,
        text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (embed_inputs.size() != 1 || embed_outputs.size() != 1) {
      return absl::InvalidArgumentError(
          "Lfm2AudioFrontendStage text_embedder_runner must have 1 input and 1 "
          "output buffer.");
    }
    LITERT_ASSIGN_OR_RETURN(auto out_type, embed_outputs[0].TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t out_elements,
                            out_type.Layout().NumElements());
    if (out_elements != static_cast<size_t>(config.llm_hidden_size)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Lfm2AudioFrontendStage text_embedder_runner output size (",
          out_elements, ") does not match llm_hidden_size (",
          config.llm_hidden_size, ")."));
    }
  }

  return absl::WrapUnique(new Lfm2AudioFrontendStage(
      text_stage, audio_stage, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(embed_inputs),
      std::move(embed_outputs), config));
}

bool Lfm2AudioFrontendStage::NeedScheduleInternal() const {
  auto matches_next =
      [this](const SingleThreadedStageWithDeque<
             Sequenced<Lfm2AudioEmbeddingSegment>>* stage) -> bool {
    if (stage == nullptr || !stage->HasOutput()) {
      return false;
    }
    auto peeked = stage->PeekOutput();
    return peeked.ok() && (*peeked)->seqnum == next_expected_seqnum_;
  };
  return matches_next(&text_stage_) || matches_next(audio_stage_);
}

absl::Status Lfm2AudioFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  bool matched_any = false;
  while (true) {
    SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>*
        ready_stage = nullptr;
    for (auto* candidate : {&text_stage_, audio_stage_}) {
      if (candidate == nullptr || !candidate->HasOutput()) {
        continue;
      }
      auto peeked = candidate->PeekOutput();
      if (peeked.ok() && (*peeked)->seqnum == next_expected_seqnum_) {
        ready_stage = candidate;
        break;
      }
    }

    if (ready_stage == nullptr) {
      break;
    }

    matched_any = true;
    ABSL_ASSIGN_OR_RETURN(Sequenced<Lfm2AudioEmbeddingSegment> seq_seg,
                          ready_stage->GetOutput());
    ++next_expected_seqnum_;

    if (seq_seg.item.is_end_of_turn) {
      ABSL_RETURN_IF_ERROR(FinalizeTurn());
    } else if (seq_seg.item.num_tokens > 0) {
      ABSL_RETURN_IF_ERROR(AppendSegment(seq_seg.item));
    }
  }

  if (!matched_any) {
    return absl::NotFoundError(
        "Lfm2AudioFrontendStage has no ready sequenced items.");
  }
  return absl::OkStatus();
}

void Lfm2AudioFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  next_expected_seqnum_ = 0;
  is_first_turn_ = true;
  resolved_mode_ = Lfm2AudioGenerationMode::kInterleaved;
  turn_segments_.clear();
}

absl::Status Lfm2AudioFrontendStage::AppendSegment(
    const Lfm2AudioEmbeddingSegment& seg) {
  if (seg.embeddings.size() !=
      static_cast<size_t>(seg.num_tokens) * hidden_dim_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Lfm2AudioEmbeddingSegment embeddings size (", seg.embeddings.size(),
        ") does not match num_tokens * llm_hidden_size (",
        seg.num_tokens * hidden_dim_, ")."));
  }
  turn_segments_.push_back(seg);
  return absl::OkStatus();
}

absl::StatusOr<Lfm2AudioEmbeddingSegment>
Lfm2AudioFrontendStage::EmbedControlText(absl::string_view text) {
  Lfm2AudioEmbeddingSegment seg;
  ABSL_ASSIGN_OR_RETURN(seg.token_ids, TokenizeLfm2AudioTextWithSpecialTokens(
                                           *tokenizer_, text, config_));
  seg.num_tokens = static_cast<int>(seg.token_ids.size());
  ABSL_ASSIGN_OR_RETURN(seg.embeddings,
                        EmbedTokenSequenceWithRunner(
                            seg.token_ids, config_.llm_vocab_size, hidden_dim_,
                            text_embedder_runner_.get(), embed_input_buffers_,
                            embed_output_buffers_, "Lfm2AudioFrontendStage"));
  return seg;
}

absl::Status Lfm2AudioFrontendStage::FinalizeTurn() {
  std::vector<Lfm2AudioEmbeddingSegment> assembled;
  if (is_first_turn_) {
    std::string effective_system_prompt = config_.system_prompt;
    if (effective_system_prompt ==
        lfm2_audio::kDefaultInterleavedSystemPrompt) {
      if (!config_.generate_audio) {
        effective_system_prompt =
            std::string(lfm2_audio::kDefaultSequentialTextSystemPrompt);
      } else if (config_.generation_mode ==
                 Lfm2AudioGenerationMode::kSequential) {
        effective_system_prompt =
            std::string(lfm2_audio::kDefaultTtsSystemPrompt);
      }
    }

    absl::string_view voice_instruction;
    if (absl::EqualsIgnoreCase(config_.voice_name, "us_male") ||
        absl::EqualsIgnoreCase(config_.voice_name, "US male")) {
      voice_instruction = "Use the US male voice.";
    } else if (absl::EqualsIgnoreCase(config_.voice_name, "us_female") ||
               absl::EqualsIgnoreCase(config_.voice_name, "US female")) {
      voice_instruction = "Use the US female voice.";
    } else if (absl::EqualsIgnoreCase(config_.voice_name, "uk_male") ||
               absl::EqualsIgnoreCase(config_.voice_name, "UK male")) {
      voice_instruction = "Use the UK male voice.";
    } else if (absl::EqualsIgnoreCase(config_.voice_name, "uk_female") ||
               absl::EqualsIgnoreCase(config_.voice_name, "UK female")) {
      voice_instruction = "Use the UK female voice.";
    }
    if (config_.generate_audio && !voice_instruction.empty() &&
        !effective_system_prompt.empty() &&
        !absl::StrContainsIgnoreCase(effective_system_prompt, "voice")) {
      absl::StrAppend(&effective_system_prompt, " ", voice_instruction);
    }

    if (!config_.generate_audio) {
      resolved_mode_ = Lfm2AudioGenerationMode::kSequential;
    } else if (config_.generation_mode != Lfm2AudioGenerationMode::kAuto) {
      resolved_mode_ = config_.generation_mode;
    } else if (absl::StrContainsIgnoreCase(effective_system_prompt,
                                           "interleaved")) {
      resolved_mode_ = Lfm2AudioGenerationMode::kInterleaved;
    } else {
      resolved_mode_ = Lfm2AudioGenerationMode::kSequential;
    }

    std::string prefix = std::string(lfm2_audio::kBosToken);
    if (!effective_system_prompt.empty()) {
      absl::StrAppend(&prefix, lfm2_audio::kImStartToken, "system\n",
                      effective_system_prompt, lfm2_audio::kImEndToken, "\n");
    }
    ABSL_ASSIGN_OR_RETURN(Lfm2AudioEmbeddingSegment sys_seg,
                          EmbedControlText(prefix));
    assembled.push_back(std::move(sys_seg));
    is_first_turn_ = false;
  }

  ABSL_ASSIGN_OR_RETURN(
      Lfm2AudioEmbeddingSegment user_start,
      EmbedControlText(absl::StrCat(lfm2_audio::kImStartToken, "user\n")));
  assembled.push_back(std::move(user_start));

  for (auto& seg : turn_segments_) {
    assembled.push_back(std::move(seg));
  }
  turn_segments_.clear();

  ABSL_ASSIGN_OR_RETURN(
      Lfm2AudioEmbeddingSegment user_end,
      EmbedControlText(absl::StrCat(lfm2_audio::kImEndToken, "\n",
                                    lfm2_audio::kImStartToken, "assistant\n")));
  assembled.push_back(std::move(user_end));

  Lfm2AudioFrontendOutput out;
  out.generate_audio = config_.generate_audio;
  out.generation_mode = config_.generate_audio
                            ? resolved_mode_
                            : Lfm2AudioGenerationMode::kSequential;
  for (const auto& seg : assembled) {
    out.prompt_embeddings.insert(out.prompt_embeddings.end(),
                                 seg.embeddings.begin(), seg.embeddings.end());
    out.prompt_token_ids.insert(out.prompt_token_ids.end(),
                                seg.token_ids.begin(), seg.token_ids.end());
    out.prompt_len += seg.num_tokens;
  }

  PushOutput(std::move(out));
  return absl::OkStatus();
}

// =============================================================================
// Lfm2AudioGenerationStage
// =============================================================================

Lfm2AudioGenerationStage::Lfm2AudioGenerationStage(
    Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::vector<TensorBuffer> proj_input_buffers,
    std::vector<TensorBuffer> proj_output_buffers,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
    std::vector<TensorBuffer> dec_input_buffers,
    std::vector<TensorBuffer> dec_output_buffers,
    std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
    std::vector<TensorBuffer> voc_input_buffers,
    std::vector<TensorBuffer> voc_output_buffers, int voc_chunk_frames,
    int voc_codebooks_dim_index, const Lfm2AudioModelConfig& config)
    : frontend_(*frontend),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      llm_runner_(std::move(llm_runner)),
      tts_projector_runner_(std::move(tts_projector_runner)),
      proj_input_buffers_(std::move(proj_input_buffers)),
      proj_output_buffers_(std::move(proj_output_buffers)),
      tts_decoder_runner_(std::move(tts_decoder_runner)),
      dec_input_buffers_(std::move(dec_input_buffers)),
      dec_output_buffers_(std::move(dec_output_buffers)),
      audio_vocoder_runner_(std::move(audio_vocoder_runner)),
      voc_input_buffers_(std::move(voc_input_buffers)),
      voc_output_buffers_(std::move(voc_output_buffers)),
      voc_chunk_frames_(voc_chunk_frames),
      voc_codebooks_dim_index_(voc_codebooks_dim_index),
      rng_(config_.seed) {}

Lfm2AudioGenerationStage::~Lfm2AudioGenerationStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioGenerationStage>>
Lfm2AudioGenerationStage::Create(
    Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0 ||
      config.tts_hidden_size <= 0 || config.tts_vocab_size <= 1 ||
      config.tts_num_codebooks <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioGenerationStage config dimensions must be positive (and "
        "tts_vocab_size >= 2).");
  }

  std::vector<TensorBuffer> proj_inputs;
  std::vector<TensorBuffer> proj_outputs;
  if (tts_projector_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(proj_inputs, tts_projector_runner->CreateInputBuffers(
                                           /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        proj_outputs,
        tts_projector_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (proj_inputs.size() != 2 || proj_outputs.size() != 2) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage tts_projector_runner must have 2 input "
          "buffers (llm_hidden, audio_codes) and 2 output buffers "
          "(depthformer_in, audio_emb).");
    }
  }

  std::vector<TensorBuffer> dec_inputs;
  std::vector<TensorBuffer> dec_outputs;
  if (tts_decoder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(dec_inputs, tts_decoder_runner->CreateInputBuffers(
                                          /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(dec_outputs, tts_decoder_runner->CreateOutputBuffers(
                                           /*signature_name=*/""));
    if (dec_inputs.size() != 4 || dec_outputs.size() != 2) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage tts_decoder_runner must have 4 input "
          "buffers (cond_emb, prev_audio_code, step_idx, kv_cache) and 2 "
          "output buffers (logits, new_kv_cache).");
    }
  }

  std::vector<TensorBuffer> voc_inputs;
  std::vector<TensorBuffer> voc_outputs;
  int voc_chunk_frames = std::max(1, config.vocoder_chunk_frames);
  int voc_codebooks_dim_index = 1;
  if (audio_vocoder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(voc_inputs, audio_vocoder_runner->CreateInputBuffers(
                                          /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        voc_outputs,
        audio_vocoder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (voc_inputs.empty() || voc_outputs.empty()) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage audio_vocoder_runner must have at least 1 "
          "input and 1 output buffer.");
    }
    LITERT_ASSIGN_OR_RETURN(auto voc_in_type, voc_inputs[0].TensorType());
    const auto& dims = voc_in_type.Layout().Dimensions();
    if (dims.size() == 3) {
      if (dims[1] == config.tts_num_codebooks) {
        voc_codebooks_dim_index = 1;
        voc_chunk_frames = dims[2];
      } else if (dims[2] == config.tts_num_codebooks) {
        voc_codebooks_dim_index = 2;
        voc_chunk_frames = dims[1];
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "audio_vocoder_runner 3D input shape [", dims[0], ", ", dims[1],
            ", ", dims[2], "] does not match tts_num_codebooks (",
            config.tts_num_codebooks, ")."));
      }
    } else if (dims.size() == 2) {
      voc_codebooks_dim_index = 0;
      voc_chunk_frames = dims[1];
    } else {
      return absl::InvalidArgumentError(
          "audio_vocoder_runner input must have rank 2 or 3.");
    }
  }

  auto stage = absl::WrapUnique(new Lfm2AudioGenerationStage(
      frontend, std::move(tokenizer), std::move(llm_runner),
      std::move(tts_projector_runner), std::move(proj_inputs),
      std::move(proj_outputs), std::move(tts_decoder_runner),
      std::move(dec_inputs), std::move(dec_outputs),
      std::move(audio_vocoder_runner), std::move(voc_inputs),
      std::move(voc_outputs), voc_chunk_frames, voc_codebooks_dim_index,
      config));

  int im_end_id = config.im_end_token_id;
  if (auto tok_id = stage->tokenizer_->TokenToId(lfm2_audio::kImEndToken);
      tok_id.ok()) {
    im_end_id = *tok_id;
  }
  stage->turn_end_token_ids_.push_back(im_end_id);
  if (auto nl_ids = stage->tokenizer_->TextToTokenIds("\n");
      nl_ids.ok() && !nl_ids->empty()) {
    stage->turn_end_token_ids_.insert(stage->turn_end_token_ids_.end(),
                                      nl_ids->begin(), nl_ids->end());
  }

  return stage;
}

bool Lfm2AudioGenerationStage::NeedScheduleInternal() const {
  return frontend_.HasOutput();
}

absl::Status Lfm2AudioGenerationStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_ASSIGN_OR_RETURN(Lfm2AudioFrontendOutput prompt, frontend_.GetOutput());

  if (prompt.prompt_len <= 0 || prompt.prompt_embeddings.empty()) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendOutput has empty prompt_embeddings.");
  }
  if (prompt.prompt_embeddings.size() !=
      static_cast<size_t>(prompt.prompt_len) * hidden_dim_) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendOutput prompt_embeddings size mismatch.");
  }

  const int turn_end_len = static_cast<int>(turn_end_token_ids_.size());
  if (config_.max_kv_len > 0 &&
      kv_tokens_used_ + prompt.prompt_len + 1 + turn_end_len >
          config_.max_kv_len) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Prompt length (", prompt.prompt_len,
        ") plus existing session context (", kv_tokens_used_,
        ") and turn suffix (", turn_end_len,
        ") exceeds configured max_kv_len (", config_.max_kv_len, ")."));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto prefill_buf,
      support::CreateTensorBuffer<float>({1, prompt.prompt_len, hidden_dim_}));
  LITERT_RETURN_IF_ERROR(
      prefill_buf.Write<float>(absl::MakeConstSpan(prompt.prompt_embeddings)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_inputs,
                        CreateExecutorInputsWithAudio(prefill_buf));
  ABSL_RETURN_IF_ERROR(llm_runner_->Prefill(prefill_inputs));
  kv_tokens_used_ += prompt.prompt_len;

  enum class Modality { kText, kAudioOut };
  Modality modality = Modality::kText;
  const bool can_generate_audio = prompt.generate_audio &&
                                  tts_projector_runner_ != nullptr &&
                                  tts_decoder_runner_ != nullptr;
  Lfm2AudioGenerationMode effective_mode = prompt.generation_mode;
  if (config_.generation_mode != Lfm2AudioGenerationMode::kAuto) {
    effective_mode = config_.generation_mode;
  } else if (effective_mode == Lfm2AudioGenerationMode::kAuto) {
    effective_mode =
        absl::StrContainsIgnoreCase(config_.system_prompt, "interleaved")
            ? Lfm2AudioGenerationMode::kInterleaved
            : Lfm2AudioGenerationMode::kSequential;
  }
  resolved_session_mode_ = effective_mode;
  const bool interleaved_mode =
      can_generate_audio &&
      (effective_mode == Lfm2AudioGenerationMode::kInterleaved) &&
      config_.interleaved_n_text > 0 && config_.interleaved_n_audio > 0;
  int modality_left = std::max(1, config_.interleaved_n_text);
  bool text_done = false;
  bool audio_done = false;

  int32_t current_token = -1;
  bool has_pending_audio_emb = false;
  std::vector<float> pending_audio_emb(hidden_dim_, 0.0f);

  LITERT_ASSIGN_OR_RETURN(auto token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));
  LITERT_ASSIGN_OR_RETURN(
      auto audio_emb_buf,
      support::CreateTensorBuffer<float>({1, 1, hidden_dim_}));

  std::vector<int> generated_text_token_ids;
  std::vector<std::vector<int32_t>> generated_audio_frames;
  int max_steps = std::max(1, config_.max_new_tokens);
  if (config_.max_kv_len > 0) {
    const int available_decode_steps =
        std::max(1, config_.max_kv_len - kv_tokens_used_ - 1 - turn_end_len);
    if (available_decode_steps < max_steps) {
      ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage capping max_new_tokens "
                           "from "
                        << max_steps << " to " << available_decode_steps
                        << " to stay within max_kv_len (" << config_.max_kv_len
                        << ").";
      max_steps = available_decode_steps;
    }
  }
  const int max_audio_frames = std::max(1, config_.max_tts_tokens);
  bool stopped_on_eos = false;

  for (int step = 0; step < max_steps; ++step) {
    --modality_left;
    lm::ExecutorInputs decode_inputs;
    if (has_pending_audio_emb) {
      LITERT_RETURN_IF_ERROR(
          audio_emb_buf.Write<float>(absl::MakeConstSpan(pending_audio_emb)));
      ABSL_ASSIGN_OR_RETURN(decode_inputs,
                            CreateExecutorInputsWithAudio(audio_emb_buf));
    } else {
      LITERT_RETURN_IF_ERROR(
          token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
      ABSL_ASSIGN_OR_RETURN(decode_inputs,
                            CreateExecutorInputsWithText(token_buf));
    }

    ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                          llm_runner_->Decode(decode_inputs));
    ++kv_tokens_used_;
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_out,
        ExtractDecodeStepOutput(decode_buf, config_.llm_vocab_size, hidden_dim_,
                                /*require_hidden_state=*/can_generate_audio));

    if (modality == Modality::kText) {
      const int token_id =
          SampleFromLogits(step_out.logits, config_.temperature, config_.top_k,
                           config_.top_p, config_.do_sample, rng_);
      if (token_id == config_.im_end_token_id ||
          (!can_generate_audio && (token_id == config_.text_end_token_id ||
                                   token_id == config_.audio_start_token_id))) {
        stopped_on_eos = (token_id == config_.im_end_token_id);
        has_pending_audio_emb = false;
        current_token = token_id;
        break;
      }
      // Runtime guard (not in the Python reference loop): once both
      // `<|text_end|>` (`text_done`) and `EOAudio` (`audio_done`) have been
      // emitted, end the turn immediately if the next text step does not emit
      // `<|im_end|>`, preventing post-`EOAudio` modality re-entry or trailing
      // tokens.
      if (text_done && audio_done) {
        has_pending_audio_emb = false;
        current_token = token_id;
        break;
      }
      if (token_id == config_.text_end_token_id) {
        text_done = true;
      }
      if (can_generate_audio && !audio_done &&
          (token_id == config_.audio_start_token_id ||
           (interleaved_mode && (modality_left <= 0 || text_done)))) {
        if (!text_done && token_id != config_.text_end_token_id &&
            token_id != config_.pad_token_id &&
            token_id != config_.bos_token_id &&
            token_id != config_.im_start_token_id &&
            token_id != config_.audio_start_token_id) {
          generated_text_token_ids.push_back(token_id);
        }
        modality = Modality::kAudioOut;
        modality_left = std::max(1, config_.interleaved_n_audio);
        current_token = token_id;
        has_pending_audio_emb = false;
        continue;
      }
      if (!text_done && token_id != config_.text_end_token_id &&
          token_id != config_.pad_token_id &&
          token_id != config_.bos_token_id &&
          token_id != config_.im_start_token_id &&
          token_id != config_.audio_start_token_id) {
        generated_text_token_ids.push_back(token_id);
      }
      current_token = token_id;
      has_pending_audio_emb = false;
    } else {
      const bool force_eo_audio =
          static_cast<int>(generated_audio_frames.size()) >= max_audio_frames;
      if (force_eo_audio) {
        ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage reached max_tts_tokens ("
                          << max_audio_frames
                          << " audio frames); forcing EOAudio.";
      }
      ABSL_ASSIGN_OR_RETURN(
          AudioFrameResult frame_res,
          GenerateAudioFrame(step_out.hidden_state, force_eo_audio));
      if (frame_res.is_eo_audio) {
        modality = Modality::kText;
        modality_left = std::max(1, config_.interleaved_n_text);
        if (text_done || !interleaved_mode) {
          audio_done = true;
        }
      } else {
        generated_audio_frames.push_back(std::move(frame_res.codes));
        if (interleaved_mode && modality_left <= 0 && !text_done) {
          modality = Modality::kText;
          modality_left = std::max(1, config_.interleaved_n_text);
        }
      }
      pending_audio_emb = std::move(frame_res.audio_embedding);
      has_pending_audio_emb = true;
    }
  }

  // Commit the final token/embedding and `<|im_end|>\n` into the LFM2 KV cache.
  if (!stopped_on_eos) {
    if (has_pending_audio_emb) {
      LITERT_RETURN_IF_ERROR(
          audio_emb_buf.Write<float>(absl::MakeConstSpan(pending_audio_emb)));
      ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                            CreateExecutorInputsWithAudio(audio_emb_buf));
      ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
      ++kv_tokens_used_;
    } else if (current_token >= 0) {
      LITERT_RETURN_IF_ERROR(
          token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
      ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                            CreateExecutorInputsWithText(token_buf));
      ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
      ++kv_tokens_used_;
    }
  }
  for (int tok : turn_end_token_ids_) {
    const int32_t tok32 = static_cast<int32_t>(tok);
    LITERT_RETURN_IF_ERROR(
        token_buf.Write<int32_t>(absl::MakeConstSpan(&tok32, 1)));
    ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                          CreateExecutorInputsWithText(token_buf));
    ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
    ++kv_tokens_used_;
  }

  if (!generated_text_token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string text,
                          tokenizer_->TokenIdsToText(generated_text_token_ids));
    if (!text.empty()) {
      PushOutput(Output(TextOutput{.confirmed_text = std::move(text)}));
    }
  }

  if (!generated_audio_frames.empty() && audio_vocoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::vector<float> waveform,
                          SynthesizeWaveform(generated_audio_frames));
    if (!waveform.empty()) {
      PushOutput(Output(AudioOutput{
          .pcm_samples = std::move(waveform),
          .sample_rate_hz = config_.audio_output_sample_rate_hz,
      }));
    }
  }

  PushOutput(Output(EndOfOutputTurn()));
  return absl::OkStatus();
}

void Lfm2AudioGenerationStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  resolved_session_mode_ = Lfm2AudioGenerationMode::kAuto;
  kv_tokens_used_ = 0;
  rng_.seed(config_.seed);
  (void)llm_runner_->Reset();
}

absl::StatusOr<Lfm2AudioGenerationStage::AudioFrameResult>
Lfm2AudioGenerationStage::GenerateAudioFrame(
    absl::Span<const float> llm_hidden_state, bool force_eo_audio) {
  const int codebooks = config_.tts_num_codebooks;
  const int df_dim = config_.tts_hidden_size;
  const int df_vocab = config_.tts_vocab_size;
  const int eo_audio_code = df_vocab - 1;

  // 1. Run `tts_projector_runner_` (`depth_linear`) on `llm_hidden_state` to
  // get `depthformer_in` of shape `[1, codebooks, df_dim]`.
  std::vector<int32_t> frame_codes(codebooks, 0);
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(llm_hidden_state));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(frame_codes)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run(
      /*signature_name=*/"", proj_input_buffers_, proj_output_buffers_));

  std::vector<float> depthformer_in(static_cast<size_t>(codebooks) * df_dim,
                                    0.0f);
  LITERT_RETURN_IF_ERROR(
      proj_output_buffers_[0].Read<float>(absl::MakeSpan(depthformer_in)));

  // 2. Run 8-codebook RQ-Transformer Depthformer (`tts_decoder_runner_`).
  LITERT_ASSIGN_OR_RETURN(auto kv_type, dec_input_buffers_[3].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t kv_elements, kv_type.Layout().NumElements());
  std::vector<float> kv_cache(kv_elements, 0.0f);
  std::vector<float> logits(df_vocab, 0.0f);

  AudioFrameResult result;
  result.codes.assign(codebooks, 0);
  int32_t prev_code = 0;

  if (force_eo_audio) {
    result.codes.assign(codebooks, eo_audio_code);
    result.is_eo_audio = true;
  } else {
    for (int k = 0; k < codebooks; ++k) {
      absl::Span<const float> cond_k(
          depthformer_in.data() + static_cast<size_t>(k) * df_dim, df_dim);
      const int32_t step_idx = static_cast<int32_t>(k);

      LITERT_RETURN_IF_ERROR(dec_input_buffers_[0].Write<float>(cond_k));
      LITERT_RETURN_IF_ERROR(dec_input_buffers_[1].Write<int32_t>(
          absl::MakeConstSpan(&prev_code, 1)));
      LITERT_RETURN_IF_ERROR(dec_input_buffers_[2].Write<int32_t>(
          absl::MakeConstSpan(&step_idx, 1)));
      LITERT_RETURN_IF_ERROR(
          dec_input_buffers_[3].Write<float>(absl::MakeConstSpan(kv_cache)));

      ABSL_RETURN_IF_ERROR(tts_decoder_runner_->Run(
          /*signature_name=*/"", dec_input_buffers_, dec_output_buffers_));
      LITERT_RETURN_IF_ERROR(
          dec_output_buffers_[0].Read<float>(absl::MakeSpan(logits)));
      LITERT_RETURN_IF_ERROR(
          dec_output_buffers_[1].Read<float>(absl::MakeSpan(kv_cache)));

      if (k > 0 && eo_audio_code >= 0 &&
          static_cast<size_t>(eo_audio_code) < logits.size()) {
        // Runtime safeguard: only codebook 0 may emit `EOAudio` (`2048`),
        // because `LFM2AudioDetokenizer` only embeds codec tokens in `[0,
        // 2047]`. Mask `logits[eo_audio_code] = -inf` on codebooks 1..7 so a
        // non-zero codebook never samples `EOAudio`.
        logits[eo_audio_code] = -std::numeric_limits<float>::infinity();
      }

      const int sampled_code = SampleFromLogits(
          logits, config_.tts_temperature, config_.tts_top_k, config_.tts_top_p,
          config_.tts_temperature > 0.0f, rng_);
      if (k == 0 && sampled_code == eo_audio_code) {
        result.codes.assign(codebooks, eo_audio_code);
        result.is_eo_audio = true;
        break;
      }
      result.codes[k] = static_cast<int32_t>(sampled_code);
      prev_code = static_cast<int32_t>(sampled_code);
    }
  }

  // 3. Embed `result.codes` via `tts_projector_runner_`
  // (`audio_embedding(codes + codebook_offsets).sum(0)`) to feed back into the
  // LFM2 backbone on the next step.
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(llm_hidden_state));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(result.codes)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run(
      /*signature_name=*/"", proj_input_buffers_, proj_output_buffers_));

  result.audio_embedding.assign(hidden_dim_, 0.0f);
  LITERT_RETURN_IF_ERROR(proj_output_buffers_[1].Read<float>(
      absl::MakeSpan(result.audio_embedding)));
  return result;
}

absl::StatusOr<std::vector<float>> Lfm2AudioGenerationStage::SynthesizeWaveform(
    const std::vector<std::vector<int32_t>>& audio_frames) {
  if (audio_frames.empty() || audio_vocoder_runner_ == nullptr) {
    return std::vector<float>();
  }

  const int codebooks = config_.tts_num_codebooks;
  const int detok_vocab = config_.tts_vocab_size - 1;
  for (size_t f = 0; f < audio_frames.size(); ++f) {
    if (static_cast<int>(audio_frames[f].size()) != codebooks) {
      return absl::InvalidArgumentError(
          "Audio frame codebook count mismatch in SynthesizeWaveform.");
    }
    for (int k = 0; k < codebooks; ++k) {
      const int32_t code = audio_frames[f][k];
      if (code < 0 || code >= detok_vocab) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Audio code (", code, ") at frame ", f, ", codebook ", k,
            " is out of detokenizer vocabulary range [0, ", detok_vocab, ")."));
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto voc_in_type, voc_input_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t in_elements,
                          voc_in_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(auto voc_out_type,
                          voc_output_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t out_elements,
                          voc_out_type.Layout().NumElements());

  const int chunk_frames = std::max(1, voc_chunk_frames_);
  const size_t samples_per_frame =
      (out_elements % static_cast<size_t>(chunk_frames) == 0)
          ? (out_elements / static_cast<size_t>(chunk_frames))
          : static_cast<size_t>(config_.detokenizer_upsample_samples);

  std::vector<int32_t> chunk_in(in_elements, -1);
  std::vector<float> chunk_out(out_elements, 0.0f);
  std::vector<float> waveform;
  waveform.reserve(audio_frames.size() * samples_per_frame);

  auto run_vocoder_window = [&](int win_start, int valid) -> absl::Status {
    // Fill with `-1` so `Lfm2AudioDetokenizerModule` masks out any right-padded
    // slots `t >= valid` in `ISTFT`.
    std::fill(chunk_in.begin(), chunk_in.end(), -1);
    if (voc_codebooks_dim_index_ == 1) {
      // `[1, codebooks, chunk_frames]` (`LFM2AudioDetokenizer` layout).
      for (int t = 0; t < valid; ++t) {
        for (int k = 0; k < codebooks; ++k) {
          chunk_in[static_cast<size_t>(k) * chunk_frames + t] =
              audio_frames[win_start + t][k];
        }
      }
    } else if (voc_codebooks_dim_index_ == 2) {
      // `[1, chunk_frames, codebooks]`.
      for (int t = 0; t < valid; ++t) {
        for (int k = 0; k < codebooks; ++k) {
          chunk_in[static_cast<size_t>(t) * codebooks + k] =
              audio_frames[win_start + t][k];
        }
      }
    } else {
      // Rank-2 `[1, chunk_frames]` fallback.
      for (int t = 0; t < valid; ++t) {
        chunk_in[t] = audio_frames[win_start + t][0];
      }
    }

    LITERT_RETURN_IF_ERROR(
        voc_input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(chunk_in)));
    ABSL_RETURN_IF_ERROR(audio_vocoder_runner_->Run(
        /*signature_name=*/"", voc_input_buffers_, voc_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        voc_output_buffers_[0].Read<float>(absl::MakeSpan(chunk_out)));
    return absl::OkStatus();
  };

  const int total_frames = static_cast<int>(audio_frames.size());
  if (total_frames <= chunk_frames) {
    ABSL_RETURN_IF_ERROR(run_vocoder_window(/*win_start=*/0, total_frames));
    const size_t valid_samples = std::min(
        out_elements, static_cast<size_t>(total_frames) * samples_per_frame);
    waveform.insert(waveform.end(), chunk_out.begin(),
                    chunk_out.begin() + valid_samples);
    return waveform;
  }

  // Overlap-discard chunked detokenization:
  // `LFM2AudioDetokenizer` is causal with a receptive field of 17 code frames
  // in `DetokLfm2Model` plus 1 code frame on each side in `ISTFT` (`kLeftCtx =
  // 18`, `kRightCtx = 1`). When `chunk_frames >= 20`, we warm up each chunk
  // with >=18 left context frames and hold back 1 right context frame so
  // chunked output is identical to one-shot detokenization without boundary
  // splices.
  const int min_full_ctx_chunk_frames =
      lfm2_audio::kDetokenizerLeftContextFrames +
      lfm2_audio::kDetokenizerRightContextFrames + 1;
  const int left_ctx = (chunk_frames >= min_full_ctx_chunk_frames)
                           ? lfm2_audio::kDetokenizerLeftContextFrames
                           : ((chunk_frames >= 6) ? (chunk_frames / 2 - 1) : 0);
  const int right_ctx =
      (left_ctx > 0) ? lfm2_audio::kDetokenizerRightContextFrames : 0;
  if (chunk_frames < min_full_ctx_chunk_frames) {
    ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage vocoder chunk_frames ("
                      << chunk_frames
                      << ") is smaller than the full causal receptive field ("
                      << min_full_ctx_chunk_frames
                      << " frames); falling back to left_ctx=" << left_ctx
                      << ", right_ctx=" << right_ctx
                      << ", which may introduce chunk-boundary artifacts.";
  }

  if (left_ctx == 0) {
    for (int start = 0; start < total_frames; start += chunk_frames) {
      const int valid = std::min(chunk_frames, total_frames - start);
      ABSL_RETURN_IF_ERROR(run_vocoder_window(start, valid));
      const size_t valid_samples = std::min(
          out_elements, static_cast<size_t>(valid) * samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin(),
                      chunk_out.begin() + valid_samples);
    }
    return waveform;
  }

  // First chunk `[0 .. chunk_frames - 1]`: emit `[0 .. chunk_frames -
  // right_ctx)`.
  ABSL_RETURN_IF_ERROR(run_vocoder_window(/*win_start=*/0, chunk_frames));
  int emit_cursor = chunk_frames - right_ctx;
  const size_t first_samples = std::min(
      out_elements, static_cast<size_t>(emit_cursor) * samples_per_frame);
  waveform.insert(waveform.end(), chunk_out.begin(),
                  chunk_out.begin() + first_samples);

  while (emit_cursor < total_frames) {
    const int remaining = total_frames - emit_cursor;
    if (remaining <= chunk_frames - left_ctx) {
      // Final chunk: anchor window at `total_frames - chunk_frames` so it is
      // 100% full (`chunk_frames` valid frames, zero right-padding) with
      // `left_discard = chunk_frames - remaining >= left_ctx`.
      const int win_start = total_frames - chunk_frames;
      const int left_discard = emit_cursor - win_start;
      ABSL_RETURN_IF_ERROR(run_vocoder_window(win_start, chunk_frames));
      const size_t start_sample = std::min(
          out_elements, static_cast<size_t>(left_discard) * samples_per_frame);
      const size_t end_sample = std::min(
          out_elements, static_cast<size_t>(chunk_frames) * samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin() + start_sample,
                      chunk_out.begin() + end_sample);
      emit_cursor = total_frames;
    } else {
      // Middle chunk: start at `emit_cursor - left_ctx`, discard `left_ctx`
      // left warmup frames and `right_ctx` right lookahead frame.
      const int win_start = emit_cursor - left_ctx;
      ABSL_RETURN_IF_ERROR(run_vocoder_window(win_start, chunk_frames));
      const size_t start_sample = std::min(
          out_elements, static_cast<size_t>(left_ctx) * samples_per_frame);
      const size_t end_sample =
          std::min(out_elements, static_cast<size_t>(chunk_frames - right_ctx) *
                                     samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin() + start_sample,
                      chunk_out.begin() + end_sample);
      emit_cursor += chunk_frames - left_ctx - right_ctx;
    }
  }

  return waveform;
}

}  // namespace litert::omni::any2any
