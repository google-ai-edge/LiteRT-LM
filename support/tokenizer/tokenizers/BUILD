# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# C++ port of the HuggingFace `tokenizers` crate (v0.23.2).

# [Google-internal load of `cc_library`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "tokenizers",
    hdrs = ["tokenizers.h"],
    deps = [
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/processors",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:added_vocabulary",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:padding",
        "//support/tokenizer/tokenizers/tokenizer:pattern",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/tokenizer:truncation",
    ],
)
