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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_FAKE_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_FAKE_CONSTRAINT_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"

namespace litert::lm {

// Constrains the model to produce a fixed sequence of token ids.
class FakeConstraint : public Constraint {
 public:
  // Represents an index into the `token_ids_` vector.
  class FakeState : public Constraint::State {
   public:
    explicit FakeState(int index) : index_(index) {}
    int index() const { return index_; }

   private:
    const int index_;
  };

  // `token_ids` is the sequence of tokens IDs the model will be constrained to
  // produce.
  //
  // Note these are token IDs, not the string values of the tokens themselves.
  //
  // The caller is responsible for ensuring that this sequence is valid -
  // specifically, that the IDs are part of the vocabulary of the model this
  // constraint is used for, and that the last token is a stop token.
  explicit FakeConstraint(std::vector<int> token_ids, int vocabulary_size)
      : token_ids_(std::move(token_ids)), vocabulary_size_(vocabulary_size) {}
  virtual ~FakeConstraint() = default;

  std::unique_ptr<State> Start() const override;
  bool IsEnded(const State& state) const override;

  int GetVocabularySize() const override { return vocabulary_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override;

 private:
  std::vector<int> token_ids_;
  const int vocabulary_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_FAKE_CONSTRAINT_H_
