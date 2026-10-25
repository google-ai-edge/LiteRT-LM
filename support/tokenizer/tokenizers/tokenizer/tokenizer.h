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
//
// C++ port of `src/tokenizer/mod.rs` (lines 221-end: `InputSequence`,
// `EncodeInput`, `TokenizerBuilder`, `Tokenizer`, `TokenizerImpl`,
// `DecodeStream`, `step_decode_stream`) from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: Rust has a generic `TokenizerImpl<M, N, PT, PP, D>` and a
// `Tokenizer` newtype around
// `TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper,
// PostProcessorWrapper, DecoderWrapper>` (with `Deref`). Only that
// instantiation is ever used by LiteRT-LM, so the C++ `Tokenizer` class is the
// concrete implementation over the wrapper types; the methods of both Rust
// types are merged into it. `train`, `train_from_files` and `from_pretrained`
// are not ported (training / network access is out of scope).

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKENIZER_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/decoders.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/processors/processors.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/padding.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/truncation.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// `pub enum InputSequence<'s> { Raw(Cow<'s, str>), PreTokenized(Cow<'s, [&'s
// str]>), PreTokenizedOwned(Cow<'s, [String]>), PreTokenizedCow(Cow<'s,
// [Cow<'s, str>]>) }`
//
// PORT NOTE: the four Rust variants only differ in ownership; here a raw
// sequence is either a borrowed `absl::string_view` or an owned `std::string`,
// and a pre-tokenized sequence is either a vector of views or a vector of
// strings. Borrowed inputs must outlive the `InputSequence`.
class InputSequence {
 public:
  // `impl From<&str>` / `impl From<Cow<str>>`
  InputSequence(absl::string_view input)  // NOLINT(google-explicit-constructor)
      : value_(input) {}
  InputSequence(const char* input)  // NOLINT(google-explicit-constructor)
      : value_(absl::string_view(input)) {}
  // `impl From<String>`
  InputSequence(std::string input)  // NOLINT(google-explicit-constructor)
      : value_(std::move(input)) {}
  // `impl From<&[&str]>` / `impl From<Vec<&str>>`
  InputSequence(  // NOLINT(google-explicit-constructor)
      std::vector<absl::string_view> input)
      : value_(std::move(input)) {}
  InputSequence(  // NOLINT(google-explicit-constructor)
      std::initializer_list<absl::string_view> input)
      : value_(std::vector<absl::string_view>(input)) {}
  InputSequence(  // NOLINT(google-explicit-constructor)
      absl::Span<const absl::string_view> input)
      : value_(std::vector<absl::string_view>(input.begin(), input.end())) {}
  // `impl From<&[String]>` / `impl From<Vec<String>>`
  InputSequence(  // NOLINT(google-explicit-constructor)
      std::vector<std::string> input)
      : value_(std::move(input)) {}
  InputSequence(  // NOLINT(google-explicit-constructor)
      absl::Span<const std::string> input)
      : value_(std::vector<std::string>(input.begin(), input.end())) {}

  // Whether this is one of the `PreTokenized*` variants.
  bool IsPreTokenized() const {
    return !std::holds_alternative<absl::string_view>(value_) &&
           !std::holds_alternative<std::string>(value_);
  }

  // The content of a `Raw` sequence. Must not be called on a pre-tokenized
  // sequence.
  absl::string_view Raw() const;

  // The sub-sequences of a `PreTokenized*` sequence (empty for `Raw`).
  std::vector<absl::string_view> PreTokenized() const;

 private:
  std::variant<absl::string_view, std::string, std::vector<absl::string_view>,
               std::vector<std::string>>
      value_;
};

// `pub enum EncodeInput<'s> { Single(InputSequence<'s>),
// Dual(InputSequence<'s>, InputSequence<'s>) }`
class EncodeInput {
 public:
  // `impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s>`
  //
  // Accepts anything an `InputSequence` is constructible from (`const char*`,
  // `absl::string_view`, `std::string`, vectors of either, ...).
  template <typename I, typename = std::enable_if_t<
                            std::is_constructible_v<InputSequence, I&&> &&
                            !std::is_same_v<std::decay_t<I>, EncodeInput>>>
  EncodeInput(I&& input)  // NOLINT(google-explicit-constructor)
      : sequence_(std::forward<I>(input)) {}
  EncodeInput(InputSequence input)  // NOLINT(google-explicit-constructor)
      : sequence_(std::move(input)) {}

  // `impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>`
  EncodeInput(InputSequence first, InputSequence second)
      : sequence_(std::move(first)), pair_(std::move(second)) {}
  template <typename I1, typename I2>
  EncodeInput(std::pair<I1, I2> input)  // NOLINT(google-explicit-constructor)
      : sequence_(std::move(input.first)), pair_(std::move(input.second)) {}

  // `EncodeInput::Single(s)` / `EncodeInput::Dual(s, p)`
  static EncodeInput Single(InputSequence sequence) {
    return EncodeInput(std::move(sequence));
  }
  static EncodeInput Dual(InputSequence first, InputSequence second) {
    return EncodeInput(std::move(first), std::move(second));
  }

  bool IsDual() const { return pair_.has_value(); }
  const InputSequence& sequence() const { return sequence_; }
  const std::optional<InputSequence>& pair() const { return pair_; }

 private:
  InputSequence sequence_;
  std::optional<InputSequence> pair_;
};

class Tokenizer;

// Builder for Tokenizer structs.
//
// `build()` fails if the `model` is missing.
//
// `pub struct TokenizerBuilder<M, N, PT, PP, D>`
class TokenizerBuilder {
 public:
  // Get an empty TokenizerBuilder.
  TokenizerBuilder() = default;

  // Convert the TokenizerBuilder to a Tokenizer.
  //
  // Conversion fails if the `model` is missing.
  //
  // PORT NOTE: consumes the builder's contents (Rust `build(self)`).
  absl::StatusOr<Tokenizer> Build();

  // Set the model.
  TokenizerBuilder& WithModel(models::ModelWrapper model) {
    model_ = std::move(model);
    return *this;
  }

  // Set the normalizer.
  TokenizerBuilder& WithNormalizer(
      std::optional<normalizers::NormalizerWrapper> normalizer) {
    normalizer_ = std::move(normalizer);
    return *this;
  }

  // Set the pre-tokenizer.
  TokenizerBuilder& WithPreTokenizer(
      std::optional<pre_tokenizers::PreTokenizerWrapper> pretokenizer) {
    pre_tokenizer_ = std::move(pretokenizer);
    return *this;
  }

  // Set the post-processor.
  TokenizerBuilder& WithPostProcessor(
      std::optional<processors::PostProcessorWrapper> post_processor) {
    post_processor_ = std::move(post_processor);
    return *this;
  }

  // Set the decoder.
  TokenizerBuilder& WithDecoder(
      std::optional<decoders::DecoderWrapper> decoder) {
    decoder_ = std::move(decoder);
    return *this;
  }

  // Set the added vocabulary.
  TokenizerBuilder& WithAddedVocabulary(AddedVocabulary added_vocabulary) {
    added_vocabulary_ = std::move(added_vocabulary);
    return *this;
  }

  // Set the truncation parameters.
  TokenizerBuilder& WithTruncation(std::optional<TruncationParams> trunc) {
    truncation_ = std::move(trunc);
    return *this;
  }

  // Set the padding parameters.
  TokenizerBuilder& WithPadding(std::optional<PaddingParams> padding) {
    padding_ = std::move(padding);
    return *this;
  }

 private:
  std::optional<models::ModelWrapper> model_;
  std::optional<normalizers::NormalizerWrapper> normalizer_;
  std::optional<pre_tokenizers::PreTokenizerWrapper> pre_tokenizer_;
  std::optional<processors::PostProcessorWrapper> post_processor_;
  std::optional<decoders::DecoderWrapper> decoder_;

  AddedVocabulary added_vocabulary_;

  std::optional<TruncationParams> truncation_;
  std::optional<PaddingParams> padding_;
};

class DecodeStream;

// A `Tokenizer` is capable of encoding/decoding any text.
//
// `pub struct TokenizerImpl<M, N, PT, PP, D>` + `pub struct Tokenizer(...)`
class Tokenizer {
 public:
  // Instantiate a new Tokenizer, with the given Model
  //
  // `TokenizerImpl::new(model)` / `Tokenizer::new(model)`
  explicit Tokenizer(models::ModelWrapper model);

  Tokenizer(const Tokenizer&) = default;
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(const Tokenizer&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;

  // `impl std::str::FromStr for Tokenizer`: deserialize from a `tokenizer.json`
  // string.
  static absl::StatusOr<Tokenizer> FromStr(absl::string_view s);
  // Instantiate a new Tokenizer from the given file
  static absl::StatusOr<Tokenizer> FromFile(absl::string_view file);
  // Instantiate a new Tokenizer from bytes
  static absl::StatusOr<Tokenizer> FromBytes(absl::Span<const uint8_t> bytes);

  // `impl Deserialize for TokenizerImpl` (serialization.cc).
  static absl::StatusOr<Tokenizer> FromJson(const Json& json);
  // `impl Serialize for TokenizerImpl` (serialization.cc).
  Json ToJson() const;

  // Serialize the current tokenizer as a String
  absl::StatusOr<std::string> ToString(bool pretty) const;

  // Save the current tokenizer at the given path
  absl::Status Save(absl::string_view path, bool pretty) const;

  // Set the normalizer.
  //
  // # Performance note
  //
  // When added tokens with `normalized = true` are already present, this
  // method re-normalizes all of them and rebuilds the matching trie. For
  // tokenizers with many added tokens this may be slow. Prefer setting the
  // normalizer before adding tokens when constructing a tokenizer
  // programmatically.
  absl::Status WithNormalizer(
      std::optional<normalizers::NormalizerWrapper> normalizer);
  // Get the normalizer
  const normalizers::NormalizerWrapper* GetNormalizer() const {
    return normalizer_.has_value() ? &*normalizer_ : nullptr;
  }

  // Set the pre tokenizer
  Tokenizer& WithPreTokenizer(
      std::optional<pre_tokenizers::PreTokenizerWrapper> pre_tokenizer) {
    pre_tokenizer_ = std::move(pre_tokenizer);
    return *this;
  }
  // Get the pre tokenizer
  const pre_tokenizers::PreTokenizerWrapper* GetPreTokenizer() const {
    return pre_tokenizer_.has_value() ? &*pre_tokenizer_ : nullptr;
  }

  // Set the post processor
  Tokenizer& WithPostProcessor(
      std::optional<processors::PostProcessorWrapper> post_processor) {
    post_processor_ = std::move(post_processor);
    return *this;
  }
  // Get the post processor
  const processors::PostProcessorWrapper* GetPostProcessor() const {
    return post_processor_.has_value() ? &*post_processor_ : nullptr;
  }

  // Set the decoder
  Tokenizer& WithDecoder(std::optional<decoders::DecoderWrapper> decoder) {
    decoder_ = std::move(decoder);
    return *this;
  }
  // Get the decoder
  const decoders::DecoderWrapper* GetDecoder() const {
    return decoder_.has_value() ? &*decoder_ : nullptr;
  }

  // Set the model
  Tokenizer& WithModel(models::ModelWrapper model) {
    model_ = std::move(model);
    return *this;
  }
  // Get the model
  const models::ModelWrapper& GetModel() const { return model_; }

  // Set the added vocabulary.
  Tokenizer& WithAddedVocabulary(AddedVocabulary added_vocabulary) {
    added_vocabulary_ = std::move(added_vocabulary);
    return *this;
  }
  // Get the added vocabulary
  const AddedVocabulary& GetAddedVocabulary() const {
    return added_vocabulary_;
  }

  // Set the truncation parameters
  //
  // Fails if `stride` is too high relative to `max_length` and
  // `post_processor.added_tokens()`
  absl::Status WithTruncation(std::optional<TruncationParams> trunc);
  // Get the currently set truncation parameters
  const TruncationParams* GetTruncation() const {
    return truncation_.has_value() ? &*truncation_ : nullptr;
  }
  // Get a mutable reference to the currently set truncation parameters
  TruncationParams* GetTruncationMut() {
    return truncation_.has_value() ? &*truncation_ : nullptr;
  }

  // Set the padding parameters
  Tokenizer& WithPadding(std::optional<PaddingParams> padding) {
    padding_ = std::move(padding);
    return *this;
  }
  // Get the currently set padding parameters
  const PaddingParams* GetPadding() const {
    return padding_.has_value() ? &*padding_ : nullptr;
  }
  // Get a mutable reference to the currently set padding parameters
  PaddingParams* GetPaddingMut() {
    return padding_.has_value() ? &*padding_ : nullptr;
  }

  // Get the vocabulary as a plain HashMap for bindings compatibility
  absl::flat_hash_map<std::string, uint32_t> GetVocab(
      bool with_added_tokens) const;

  // Get the added tokens decoder
  absl::flat_hash_map<uint32_t, AddedToken> GetAddedTokensDecoder() const {
    return added_vocabulary_.GetAddedTokensDecoder();
  }

  // Get the size of the vocabulary
  size_t GetVocabSize(bool with_added_tokens) const;

  // Converts a token in the corresponding id.
  std::optional<uint32_t> TokenToId(absl::string_view token) const {
    return added_vocabulary_.TokenToId(token, model_);
  }

  // Converts an id to the corresponding token.
  std::optional<std::string> IdToToken(uint32_t id) const;

  // set the added vocab's splitting scheme
  void SetEncodeSpecialTokens(bool value) {
    added_vocabulary_.SetEncodeSpecialTokens(value);
  }

  // Get added token value
  bool GetEncodeSpecialTokens() const {
    return added_vocabulary_.GetEncodeSpecialTokens();
  }

  // Encode the given input. This method accepts both single sequences, as well
  // as pair sequences. Also, a sequence can be a string, or already
  // pre-tokenized input directly: Contrarily to `encode`, it does not compute
  // offsets
  //
  //   // Sequences:
  //   tokenizer.EncodeFast("Single sequence", false);
  //   tokenizer.EncodeFast({"Sequence A", "Sequence B"}, false);
  //
  //   // Pre-tokenized sequences:
  //   tokenizer.EncodeFast(InputSequence({"Single", "sequence"}), false);
  //   tokenizer.EncodeFast({InputSequence({"Sequence", "A"}),
  //                         InputSequence({"Sequence", "B"})}, false);
  //
  //   // or even both types together:
  //   tokenizer.EncodeFast({"A complete sequence",
  //                         InputSequence({"And", "a", "tokenized"})}, false);
  absl::StatusOr<Encoding> EncodeFast(const EncodeInput& input,
                                      bool add_special_tokens) const;

  // Encode the given input. This method accepts both single sequences, as well
  // as pair sequences. Also, a sequence can be a string, or already
  // pre-tokenized input directly (see `EncodeFast` for examples).
  absl::StatusOr<Encoding> Encode(const EncodeInput& input,
                                  bool add_special_tokens) const;

  // Encode the given input, using offsets relative to chars instead of bytes.
  // This method accepts both single sequences, as well as pair sequences.
  // Also, a sequence can be a string, or already pre-tokenized input directly
  // (see `EncodeFast` for examples).
  absl::StatusOr<Encoding> EncodeCharOffsets(const EncodeInput& input,
                                             bool add_special_tokens) const;

  // Decode the given ids, back to a String
  absl::StatusOr<std::string> Decode(absl::Span<const uint32_t> ids,
                                     bool skip_special_tokens) const;

  // Decode the given ids, back to a String
  // See [`DecodeStream`]
  //
  // The returned stream keeps a pointer to this tokenizer, which must outlive
  // it.
  ::tokenizers::DecodeStream DecodeStream(bool skip_special_tokens) const;

  // Register the given tokens as special tokens. This is especially useful for
  // removing these special tokens while decoding
  absl::StatusOr<size_t> AddSpecialTokens(std::vector<AddedToken> tokens) {
    return added_vocabulary_.AddSpecialTokens(std::move(tokens), model_,
                                              GetNormalizer());
  }

  // Add the given tokens to the added vocabulary
  absl::StatusOr<size_t> AddTokens(std::vector<AddedToken> tokens) {
    return added_vocabulary_.AddTokens(std::move(tokens), model_,
                                       GetNormalizer());
  }

  // Post processing logic, handling the case where there is no PostProcessor
  // set
  absl::StatusOr<Encoding> PostProcess(Encoding encoding,
                                       std::optional<Encoding> pair_encoding,
                                       bool add_special_tokens) const;

  // Encode all the sentences in parallel, using multiple threads
  //
  // PORT NOTE: serial (`utils/parallelism.rs` is not ported).
  absl::StatusOr<std::vector<Encoding>> EncodeBatch(
      const std::vector<EncodeInput>& inputs, bool add_special_tokens) const;

  // Encode all the sentences in parallel, using multiple threads.
  // The offsets on each `Encoding` will be relative to chars instead of bytes.
  absl::StatusOr<std::vector<Encoding>> EncodeBatchCharOffsets(
      const std::vector<EncodeInput>& inputs, bool add_special_tokens) const;

  // Encode all the sentences in parallel, using multiple threads
  absl::StatusOr<std::vector<Encoding>> EncodeBatchFast(
      const std::vector<EncodeInput>& inputs, bool add_special_tokens) const;

  // Decode all sentences in parallel
  absl::StatusOr<std::vector<std::string>> DecodeBatch(
      absl::Span<const std::vector<uint32_t>> sentences,
      bool skip_special_tokens) const;

 private:
  friend class TokenizerBuilder;

  // Encode a single sequence
  absl::StatusOr<Encoding> EncodeSingleSequence(const InputSequence& sequence,
                                                uint32_t type_id,
                                                OffsetType offsets_type) const;

  // Tokenization logic, makes the bridge between the pre-tokenization phase and
  // the real tokenization phase, and converting offsets back to the original
  // referential.
  absl::StatusOr<Encoding> DoTokenize(PreTokenizedString pretokenized,
                                      uint32_t type_id,
                                      std::optional<uint32_t> word_idx,
                                      OffsetType offsets_type) const;

  // Normalization logic, go through all normalizers
  absl::StatusOr<NormalizedString> DoNormalize(
      NormalizedString normalized) const;

  // PreTokenization logic, handling the case where there is no PreTokenizer
  // set
  absl::StatusOr<PreTokenizedString> DoPreTokenize(
      PreTokenizedString pretokenized) const;

  size_t GetNAddedTokens(bool is_pair) const;

  // Tokenizer parts
  std::optional<normalizers::NormalizerWrapper> normalizer_;
  std::optional<pre_tokenizers::PreTokenizerWrapper> pre_tokenizer_;
  models::ModelWrapper model_;
  std::optional<processors::PostProcessorWrapper> post_processor_;
  std::optional<decoders::DecoderWrapper> decoder_;

  // Added Vocabulary capabilities
  AddedVocabulary added_vocabulary_;

  // General processing parameters
  std::optional<TruncationParams> truncation_;
  std::optional<PaddingParams> padding_;
};

// DecodeStream will keep the state necessary to produce individual chunks of
// strings given an input stream of token_ids.
//
// This is necessary because decoding in general cannot achieve that since
// strings depend on surrounding ids to provide a valid string. Typically
// stripping extra spaces
//
// Example:
//
//   ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
//   Tokenizer::FromFile("roberta.json"));
//
//   DecodeStream decode_stream = tokenizer.DecodeStream(false);
//   EXPECT_EQ(decode_stream.Step(713).value(), "This");
//   EXPECT_EQ(decode_stream.Step(16).value(), " is");
//   EXPECT_EQ(decode_stream.Step(41).value(), " an");
//   EXPECT_EQ(decode_stream.Step(1246).value(), " example");
//
// Returning `None` means the given id is not enough to produce a chunk.
// This typically happens with `byte_fallback` options where some tokens do
// not represent valid utf-8, and only follow-up token_ids will help produce
// a valid chunk.
//
// To see how [`DecodeStream`] is necessary, consider a BPE vocabulary
// `{"▁This": 0}` with a `Metaspace` decoder: `tokenizer.Decode({0, 0}, false)`
// is "This This" (the decoder removes the extra initial space), while decoding
// one token at a time would produce "ThisThis" (`tokenizer.Decode({0}, false)`
// is "This"). Using a stream fixes it by keeping the necessary state:
// `Step(0)` yields "This", then `Step(0)` yields " This".
//
// (The doc-tests of the Rust original are ported to tests/stream_test.cc.)
class DecodeStream {
 public:
  // See [`DecodeStream`]
  absl::StatusOr<std::optional<std::string>> Step(uint32_t id);

 private:
  friend class Tokenizer;

  DecodeStream(const Tokenizer* tokenizer, bool skip_special_tokens)
      : tokenizer_(tokenizer),
        skip_special_tokens_(skip_special_tokens),
        prefix_(""),
        prefix_index_(0) {}

  // A reference to the tokenizer
  const Tokenizer* tokenizer_;
  // Regular decode option that is kept throughout.
  bool skip_special_tokens_;
  // A temporary buffer of the necessary token_ids needed
  // to produce valid string chunks.
  // This typically contains 3 parts:
  //  - read
  //  - prefix
  //  - rest
  //
  // Read is the bit necessary to surround the prefix
  // so decoding the whole ids produces a valid prefix.
  // Prefix is the previously produced string, kept around to trim off of
  // the next valid chunk
  std::vector<uint32_t> ids_;
  // The previously returned chunk that needs to be discarded from the
  // decoding of the current ids to produce the next chunk
  std::string prefix_;
  // The index within the ids corresponding to the prefix so we can drain
  // correctly
  size_t prefix_index_;
};

// `#[derive(thiserror::Error, Debug)] pub enum DecodeStreamError`
struct DecodeStreamError {
  // `#[error("Invalid prefix encountered while decoding stream. Token ID:
  // {token_id}, Expected prefix: '{expected_prefix}', Actual string:
  // '{actual_string}'")] InvalidPrefix { .. }`
  static absl::Status InvalidPrefix(uint32_t token_id,
                                    absl::string_view expected_prefix,
                                    absl::string_view actual_string);
};

// Internal function exposed only to bypass python limitations
absl::StatusOr<std::optional<std::string>> StepDecodeStream(
    const Tokenizer& tokenizer, std::vector<uint32_t> token_ids,
    bool skip_special_tokens, std::vector<uint32_t>& ids, std::string& prefix,
    size_t& prefix_index);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_TOKENIZER_H_
