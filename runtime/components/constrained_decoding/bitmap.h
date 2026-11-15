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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_BITMAP_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_BITMAP_H_

#include "absl/base/attributes.h"  // from @com_google_absl

namespace litert::lm {

// Deprecated: Use LogitMask / BitmapLogitMask instead.
// The bitmap of vocabulary to indicate the allowed tokens.
class ABSL_DEPRECATED("Use LogitMask / BitmapLogitMask instead.") Bitmap {
 public:
  // Returns true if the `index`th token is allowed.
  virtual bool Get(int index) const = 0;

  virtual ~Bitmap() = default;
};

// Deprecated: Use BitmapLogitMask::CreateAllAllowed instead.
// A bitmap implementation that allows all tokens.
class ABSL_DEPRECATED("Use BitmapLogitMask::CreateAllAllowed instead.")
    AllAllowedBitmap : public Bitmap {
 public:
  bool Get(int index) const override { return true; }
};

// Deprecated: Use BitmapLogitMask::CreateSingleAllowedToken instead.
// A bitmap implementation that allows only the one specified token.
class ABSL_DEPRECATED("Use BitmapLogitMask::CreateSingleAllowedToken instead.")
    SingleAllowedTokenBitmap : public Bitmap {
 public:
  explicit SingleAllowedTokenBitmap(int allowed_token_id)
      : allowed_token_id_(allowed_token_id) {}

  bool Get(int index) const override { return index == allowed_token_id_; }

 private:
  const int allowed_token_id_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_BITMAP_H_
