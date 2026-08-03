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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_BITMAP_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_BITMAP_H_

namespace litert::lm {

// The bitmap of vocabulary to indicate the allowed tokens.
class Bitmap {
 public:
  // Returns true if the `index`th token is allowed.
  virtual bool Get(int index) const = 0;

  virtual ~Bitmap() = default;
};

// A bitmap implementation that allows all tokens.
class AllAllowedBitmap : public Bitmap {
 public:
  bool Get(int index) const override { return true; }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_BITMAP_H_
