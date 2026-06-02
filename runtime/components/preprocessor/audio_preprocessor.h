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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_H_

#include <array>
#include <ostream>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

// Configuration for audio preprocessing.
class AudioPreprocessorConfig {
 public:
  // The padding type of for FFT bins.
  enum FftPaddingType {
    // Right padding. The resulted FFT frame will be right padding with zeros or
    // truncated to the given FFT frame length.
    kRight = 0,
    // Center padding. The results FFT frame will be left and right padding with
    // zeros with same amount, or truncated with same amount on left and right,
    // to the given FFT frame length.
    kCenter = 1
  };

  // Creates Google's Universal Speech Model (USM) audio preprocessing
  // configuration.
  static AudioPreprocessorConfig CreateDefaultUsmConfig() {
    return AudioPreprocessorConfig(
        /* sample_rate_hz= */ 16000,
        /* num_channels= */ 1,
        /* frame_length= */ 512,
        /* hop_length= */ 160,
        /* fft_length = */ 1024,
        /* input_scale = */ 32768,
        /* pre_emphasis_factor = */ 0.97,
        /* num_mel_bins= */ 128,
        /* mel_low_hz= */ 125.0,
        /* mel_high_hz= */ 7500.0,
        /* mel_floor= */ 1e-6,
        /* normalize_mel= */ true,
        /* add_floor_to_mel_before_log= */ false,
        /* semicausal_padding= */ false, /* non_zero_hanning= */ true,
        /* periodic_hanning= */ true,
        /* fft_padding_type= */ FftPaddingType::kRight,
        /* skip_mel_spectrogram_extraction= */ false,
        /* buffer_last_frame= */ false);
  }

  static AudioPreprocessorConfig Create(
      int sample_rate_hz, int num_channels, int frame_length, int hop_length,
      int fft_length, float input_scale, float pre_emphasis_factor,
      int num_mel_bins, float mel_low_hz, float mel_high_hz, float mel_floor,
      bool normalize_mel, bool add_floor_to_mel_before_log,
      bool semicausal_padding, bool non_zero_hanning, bool periodic_hanning,
      FftPaddingType fft_padding_type,
      bool skip_mel_spectrogram_extraction = false,
      bool buffer_last_frame = false) {
    return AudioPreprocessorConfig(
        sample_rate_hz, num_channels, frame_length, hop_length, fft_length,
        input_scale, pre_emphasis_factor, num_mel_bins, mel_low_hz, mel_high_hz,
        mel_floor, normalize_mel, add_floor_to_mel_before_log,
        semicausal_padding, non_zero_hanning, periodic_hanning,
        fft_padding_type, skip_mel_spectrogram_extraction, buffer_last_frame);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const FftPaddingType& padding_type) {
    switch (padding_type) {
      case FftPaddingType::kRight:
        os << "right";
        break;
      case FftPaddingType::kCenter:
        os << "center";
        break;
      default:
        os << "unknown";
        break;
    }
    return os;
  }

  // Allows logging of the config.
  friend std::ostream& operator<<(std::ostream& os,
                                  const AudioPreprocessorConfig& config) {
    os << "AudioPreprocessorConfig {\n";
    os << "  sample_rate_hz: " << config.GetSampleRateHz() << "\n";
    os << "  num_channels: " << config.GetNumChannels() << "\n";
    os << "  input_scale: " << config.GetInputScale() << "\n";
    os << "  pre_emphasis_factor: " << config.GetPreEmphasisFactor() << "\n";
    os << "  fft_length: " << config.GetFftLength() << "\n";
    os << "  fft_bins: " << config.GetFftBins() << "\n";
    os << "  frame_length: " << config.GetFrameLength() << "\n";
    os << "  hop_length: " << config.GetHopLength() << "\n";
    os << "  num_mel_bins: " << config.GetNumMelBins() << "\n";
    os << "  mel_low_hz: " << config.GetMelLowHz() << "\n";
    os << "  mel_high_hz: " << config.GetMelHighHz() << "\n";
    os << "  mel_floor: " << config.GetMelFloor() << "\n";
    os << "  normalize_mel: " << config.GetNormalizeMel() << "\n";
    os << "  add_floor_to_mel_before_log: "
       << config.GetAddFloorToMelBeforeLog() << "\n";
    os << "  semicausal_padding: " << config.GetSemicausalPadding() << "\n";
    os << "  non_zero_hanning: " << config.GetNonZeroHanning() << "\n";
    os << "  periodic_hanning: " << config.GetPeriodicHanning() << "\n";
    os << "  fft_padding_type: " << config.GetFftPaddingType() << "\n";
    os << "  skip_mel_spectrogram_extraction: "
       << config.SkipMelSpectrogramExtraction() << "\n";
    os << "  buffer_last_frame: " << config.BufferLastFrame() << "\n";
    os << "}";
    return os;
  }

  // Getter APIs.
  // The sample rate while loading the audio. The audio should be resampled to
  // the configured sample rate.
  int GetSampleRateHz() const { return sample_rate_hz_; }
  // The number of audio channels the preprocessor expect from the audio
  // content.
  int GetNumChannels() const { return num_channels_; }
  // The scale applied to the audio PCM frames before processing to
  // spectrogram.
  float GetInputScale() const { return input_scale_; }
  // The pre-emphasis factor applied to the audio before processing to
  // spectrogram.
  float GetPreEmphasisFactor() const { return pre_emphasis_factor_; }
  // The FFT length used for processing the audio.
  int GetFftLength() const { return fft_length_; }
  // The number of FFT bins used for real-sequence Fourier transform (RFFT) and
  // Mel spectrogram processing. It is derived from the FFT length as FFT
  // length / 2 + 1.
  int GetFftBins() const { return fft_bins_; }
  // The frame length used for for each frame of Short-Time Fourier Transform
  // (STFT).
  int GetFrameLength() const { return frame_length_; }
  // The hop length used for in sliding window of Short-Time Fourier Transform
  // (STFT).
  int GetHopLength() const { return hop_length_; }
  // The number of Mel bins used for Mel spectrogram processing.
  int GetNumMelBins() const { return num_mel_bins_; }
  // The lower bound of the Mel frequency range.
  float GetMelLowHz() const { return mel_low_hz_; }
  // The upper bound of the Mel frequency range.
  float GetMelHighHz() const { return mel_high_hz_; }
  // The floor value of the Mel spectrogram.
  float GetMelFloor() const { return mel_floor_; }
  // Whether to normalize the Mel spectrogram with precalculated mean and std
  // dev.
  bool GetNormalizeMel() const { return normalize_mel_; }
  // Whether to add the floor value to the Mel spectrogram before taking the
  // logarithm.
  bool GetAddFloorToMelBeforeLog() const {
    return add_floor_to_mel_before_log_;
  }
  // Whether to use semicausal padding for the audio frames.
  bool GetSemicausalPadding() const { return semicausal_padding_; }
  // Whether to use non-zero Hanning window for FFT.
  bool GetNonZeroHanning() const { return non_zero_hanning_; }
  // Whether to use the periodic Hanning window for FFT.
  bool GetPeriodicHanning() const { return periodic_hanning_; }
  // The padding type used for FFT.
  FftPaddingType GetFftPaddingType() const { return fft_padding_type_; }
  // Whether to skip the Mel spectrogram extraction.
  bool SkipMelSpectrogramExtraction() const {
    return skip_mel_spectrogram_extraction_;
  }
  // Whether to buffer the last pcm frame instead of padding. Should be true for
  // streaming input, and false for non-streaming audio input.
  bool BufferLastFrame() const { return buffer_last_frame_; }

  // Setter APIs.
  void SetSampleRateHz(int sample_rate_hz) { sample_rate_hz_ = sample_rate_hz; }
  void SetNumChannels(int num_channels) { num_channels_ = num_channels; }
  void SetInputScale(float input_scale) { input_scale_ = input_scale; }
  void SetPreEmphasisFactor(float pre_emphasis_factor) {
    pre_emphasis_factor_ = pre_emphasis_factor;
  }
  // The FFT length must be even for real FFT optimization. The FFT bins will be
  // derived from the FFT length as FFT length / 2 + 1.
  void SetFftLength(int fft_length) {
    fft_length_ = fft_length;
    fft_bins_ = fft_length / 2 + 1;
  }
  void SetFrameLength(int frame_length) { frame_length_ = frame_length; }
  void SetHopLength(int hop_length) { hop_length_ = hop_length; }
  void SetNumMelBins(int num_mel_bins) { num_mel_bins_ = num_mel_bins; }
  void SetMelLowHz(float mel_low_hz) { mel_low_hz_ = mel_low_hz; }
  void SetMelHighHz(float mel_high_hz) { mel_high_hz_ = mel_high_hz; }
  void SetMelFloor(float mel_floor) { mel_floor_ = mel_floor; }
  void SetNormalizeMel(bool normalize_mel) { normalize_mel_ = normalize_mel; }
  void SetAddFloorToMelBeforeLog(bool add_floor_to_mel_before_log) {
    add_floor_to_mel_before_log_ = add_floor_to_mel_before_log;
  }
  void SetSemicausalPadding(bool semicausal_padding) {
    semicausal_padding_ = semicausal_padding;
  }
  void SetNonZeroHanning(bool non_zero_hanning) {
    non_zero_hanning_ = non_zero_hanning;
  }
  void SetPeriodicHanning(bool periodic_hanning) {
    periodic_hanning_ = periodic_hanning;
  }
  void SetFftPaddingType(FftPaddingType fft_padding_type) {
    fft_padding_type_ = fft_padding_type;
  }
  void SetSkipMelSpectrogramExtraction(bool skip_mel_spectrogram_extraction) {
    skip_mel_spectrogram_extraction_ = skip_mel_spectrogram_extraction;
  }
  void SetBufferLastFrame(bool buffer_last_frame) {
    buffer_last_frame_ = buffer_last_frame;
  }

  // The Mel Spectrogram means used for Universal Speech Model (USM) during
  // preprocessing.
  static constexpr std::array<float, 128> kUsmMelMean{
      6.398797734146062,  6.5292966718485665, 6.636971307272159,
      6.73283598251503,   6.83729192594687,   6.955722303271236,
      7.102944890730766,  7.114182036087843,  7.1506544101153,
      7.174958993259514,  7.1890256978077804, 7.196835788986042,
      7.211737590554171,  7.365040287042535,  7.350661707754529,
      7.34752702412618,   7.370936184320344,  7.552167274579683,
      7.4736985912567455, 7.461733145619613,  7.655010083032587,
      7.537023586741711,  7.59332033698754,   7.678828995158089,
      7.573545549481997,  7.721706263812856,  7.548489195294597,
      7.647480899467908,  7.546350507038094,  7.552359044394656,
      7.60142267532906,   7.510803537242497,  7.547512749381739,
      7.5734628575808145, 7.516065818981327,  7.544310572169082,
      7.556128732606547,  7.578428971230521,  7.565946473157099,
      7.565821431053628,  7.582146705201401,  7.5917054493764775,
      7.59647680034444,   7.612909043144701,  7.642191074647679,
      7.682020208604412,  7.669657702288002,  7.636762908696176,
      7.645613169792156,  7.687786852309006,  7.733375349074729,
      7.705414197270183,  7.773851002316419,  7.767855696186511,
      7.804625030416079,  7.8095583241565505, 7.845300151068656,
      7.832030482713495,  7.876477438621265,  7.886595835981996,
      7.907747879286325,  7.926010325946424,  7.927971987569718,
      7.94765994925662,   7.9609369675109205, 7.977485334083968,
      7.995276449058029,  8.020093867153456,  8.026893789702653,
      8.036394113138993,  8.072079269745391,  8.072009510709744,
      8.15832987882215,   8.169035932109242,  8.201262910500471,
      8.203176911295596,  8.237251381186532,  8.265968214462914,
      8.278791003594298,  8.279921657260331,  8.303751782080207,
      8.323985266369666,  8.358499418073363,  8.368121771923692,
      8.392162333974197,  8.40529917133684,   8.421934604788884,
      8.43307981480797,   8.416437732709245,  8.380481381138022,
      8.313028108945332,  8.172698101608145,  7.987087868524417,
      7.775018865353218,  7.587469885918491,  7.485680948258058,
      7.425561455270659,  7.426161453764725,  7.500171657170674,
      7.473711809407939,  7.497915553109761,  7.555291079941853,
      7.5404297094497155, 7.554637855844384,  7.5536294881940025,
      7.597411437015373,  7.620857310821611,  7.622024042245356,
      7.643684482318661,  7.651806604022742,  7.647768200868812,
      7.619968160658521,  7.663675433728041,  7.770133777809638,
      7.775737195054957,  7.756637821283381,  7.7958903182806445,
      7.824714343764584,  7.8699194044250325, 7.857690367947652,
      7.854133456399421,  7.83057312917979,   7.780062155284722,
      7.687571300835443,  7.626255596158039,  7.475138444832542,
      7.31241576045514,   7.162930372619685,
  };

  // The Mel Spectrogram standard deviations constants used for Universal Speech
  // Model (USM) during preprocessing.
  static constexpr std::array<float, 128> kUsmMelStdDev{
      1.6785894541269812, 1.6687138672328043, 1.6906522689607268,
      1.7375192957945016, 1.7755335232132188, 1.7945350399969586,
      1.8160038735261768, 1.8455822079478754, 1.854889301328728,
      1.8544058257314018, 1.8531530795826658, 1.8568193392072,
      1.8568580559801775, 1.8403822120311448, 1.8311156303932052,
      1.8381223837390877, 1.8582757939740133, 1.8751353033960765,
      1.8940031697532662, 1.9045566324594227, 1.9114104933328382,
      1.9234409916967738, 1.932244372950416,  1.9354540832886058,
      1.9196173248258872, 1.8884371698304272, 1.8666212011400265,
      1.851852265212217,  1.8466309429379515, 1.8370433682382064,
      1.8312948374209728, 1.8233918348681029, 1.8162900339615862,
      1.813554336166136,  1.7988012203002604, 1.7783664628243725,
      1.762995373099593,  1.754638830337111,  1.7562192553046327,
      1.7570134298011308, 1.748103676233597,  1.7420266564237143,
      1.7433799765791382, 1.7405273444710188, 1.7681605535143332,
      1.7928765468247894, 1.7832784911754684, 1.7556019331853459,
      1.734978397119943,  1.7251193027145706, 1.711577677561937,
      1.7077475454470532, 1.702793505675667,  1.7087228728780646,
      1.7055479598955696, 1.7048659481569446, 1.7136985315687527,
      1.7003759527643025, 1.7038510617369829, 1.712407460050622,
      1.7195395708962748, 1.715985369102956,  1.7047382463157097,
      1.6858892841332958, 1.6803980138770978, 1.6883086163746897,
      1.678822586089551,  1.6704169259147215, 1.6824154866833487,
      1.7002006169486261, 1.7095077608591729, 1.7127719919531275,
      1.7007540237588394, 1.7007030789334565, 1.7006801726721705,
      1.7084333739135957, 1.7080081837410785, 1.7088852843730529,
      1.7058124003569382, 1.7104967128913229, 1.7017088898161998,
      1.6946290530635235, 1.6886895951157692, 1.6913609136330663,
      1.6802034976166595, 1.6778644057956866, 1.6844856225324205,
      1.6919889285341483, 1.6918548241011255, 1.6771215766236411,
      1.6753742459089904, 1.6732896439517075, 1.665104739745144,
      1.682512689327978,  1.7001049276791989, 1.71496232533367,
      1.751371703351037,  1.7589949482516734, 1.7274831977280356,
      1.7428303906628124, 1.7427952258580872, 1.7072930970436015,
      1.72696991469254,   1.7128335116767701, 1.7266508365456639,
      1.699287147275948,  1.6860698274507981, 1.6862991003373358,
      1.683393071329867,  1.687619365543026,  1.7100825041856975,
      1.7407356256589301, 1.7218710733945026, 1.6776658140019411,
      1.6864518015922916, 1.7273244787326472, 1.6992470398169233,
      1.6800806970795965, 1.6579370965601807, 1.6647055065206582,
      1.65766768806214,   1.6294301234765352, 1.5918612004781831,
      1.5335441292387613, 1.3949765253217616, 1.2628815962896491,
      1.1053653031914006, 0.9263256925938697,
  };

 private:
  explicit AudioPreprocessorConfig(
      // Audio decoding parameters.
      int sample_rate_hz, int num_channels,
      // FFT parameters.
      int frame_length, int hop_length, int fft_length, float input_scale,
      float pre_emphasis_factor,
      // Mel spectrogram parameters.
      int num_mel_bins, float mel_low_hz, float mel_high_hz, float mel_floor,
      bool normalize_mel, bool add_floor_to_mel_before_log,
      bool semicausal_padding, bool non_zero_hanning, bool periodic_hanning,
      FftPaddingType fft_padding_type, bool skip_mel_spectrogram_extraction,
      bool buffer_last_frame)
      : sample_rate_hz_(sample_rate_hz),
        num_channels_(num_channels),
        fft_length_(fft_length),
        fft_bins_(fft_length / 2 + 1),
        frame_length_(frame_length),
        hop_length_(hop_length),
        num_mel_bins_(num_mel_bins),
        mel_low_hz_(mel_low_hz),
        mel_high_hz_(mel_high_hz),
        mel_floor_(mel_floor),
        input_scale_(input_scale),
        pre_emphasis_factor_(pre_emphasis_factor),
        normalize_mel_(normalize_mel),
        add_floor_to_mel_before_log_(add_floor_to_mel_before_log),
        semicausal_padding_(semicausal_padding),
        non_zero_hanning_(non_zero_hanning),
        periodic_hanning_(periodic_hanning),
        fft_padding_type_(fft_padding_type),
        skip_mel_spectrogram_extraction_(skip_mel_spectrogram_extraction),
        buffer_last_frame_(buffer_last_frame) {}
  int sample_rate_hz_;
  int num_channels_;
  int fft_length_;
  int fft_bins_;
  int frame_length_;
  int hop_length_;
  int num_mel_bins_;
  float mel_low_hz_;
  float mel_high_hz_;
  float mel_floor_;
  float input_scale_;
  float pre_emphasis_factor_;
  bool normalize_mel_;
  bool add_floor_to_mel_before_log_;
  bool semicausal_padding_;
  bool non_zero_hanning_;
  bool periodic_hanning_;
  FftPaddingType fft_padding_type_;
  bool skip_mel_spectrogram_extraction_;
  bool buffer_last_frame_;
};

// Interface for audio preprocessing.
class AudioPreprocessor {
 public:
  virtual ~AudioPreprocessor() = default;

  // Preprocesses the undecoded audio bytes and returns the preprocessed audio.
  virtual absl::StatusOr<InputAudio> Preprocess(
      const InputAudio& input_audio) = 0;

  // Reset the audio preprocessor to the initial state.
  virtual void Reset() = 0;
};

std::ostream& operator<<(
    std::ostream& os,
    const AudioPreprocessorConfig::FftPaddingType& padding_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_H_
