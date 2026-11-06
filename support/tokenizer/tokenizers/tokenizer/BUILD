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

# C++ port of the `tokenizer` module of the HuggingFace `tokenizers` crate
# (v0.23.2): the pipeline traits, `Encoding`, `NormalizedString`,
# `PreTokenizedString`, `AddedVocabulary`, padding and truncation.
# `tokenizer.{h,cc}` / `serialization.{h,cc}` (the `Tokenizer` itself) are the
# :tokenizer target.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# `Offsets` and `Token` from tokenizer/mod.rs. Split out of :api because
# :encoding / :pre_tokenizer need them while api.cc needs :encoding /
# :pre_tokenizer.
cc_library(
    name = "token",
    hdrs = ["token.h"],
)

# The `Normalizer`, `PreTokenizer`, `Model`, `PostProcessor` and `Decoder`
# traits (tokenizer/mod.rs lines 1-220).
cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":encoding",
        ":normalizer",
        ":pre_tokenizer",
        ":token",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/util:litert_status_util",
    ],
)

# tokenizer/encoding.rs. Also defines `PaddingDirection` and
# `TruncationDirection` (see the PORT NOTE in encoding.h).
cc_library(
    name = "encoding",
    srcs = ["encoding.cc"],
    hdrs = ["encoding.h"],
    deps = [
        ":token",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "encoding_test",
    srcs = ["encoding_test.cc"],
    deps = [
        ":encoding",
        ":token",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

# utils/padding.rs
cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        ":encoding",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    deps = [
        ":encoding",
        ":padding",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "//support/util:test_utils",
    ],
)

# utils/truncation.rs
cc_library(
    name = "truncation",
    srcs = ["truncation.cc"],
    hdrs = ["truncation.h"],
    deps = [
        ":encoding",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "truncation_test",
    srcs = ["truncation_test.cc"],
    deps = [
        ":encoding",
        ":truncation",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

# tokenizer/pattern.rs
cc_library(
    name = "pattern",
    srcs = ["pattern.cc"],
    hdrs = ["pattern.h"],
    deps = [
        ":token",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "pattern_test",
    srcs = ["pattern_test.cc"],
    deps = [
        ":pattern",
        ":token",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/util:test_utils",
    ],
)

# tokenizer/normalizer.rs
cc_library(
    name = "normalizer",
    srcs = ["normalizer.cc"],
    hdrs = ["normalizer.h"],
    deps = [
        ":pattern",
        ":token",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/unicode:categories",
        "//support/tokenizer/tokenizers/unicode:normalization",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "normalizer_test",
    srcs = ["normalizer_test.cc"],
    deps = [
        ":normalizer",
        ":pattern",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/unicode:categories",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/util:test_utils",
    ],
)

# tokenizer/pre_tokenizer.rs
cc_library(
    name = "pre_tokenizer",
    srcs = ["pre_tokenizer.cc"],
    hdrs = ["pre_tokenizer.h"],
    deps = [
        ":encoding",
        ":normalizer",
        ":token",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "pre_tokenizer_test",
    srcs = ["pre_tokenizer_test.cc"],
    deps = [
        ":encoding",
        ":normalizer",
        ":pre_tokenizer",
        ":token",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//support/util:test_utils",
    ],
)

# Replacement for the `daachorse` crate (leftmost-longest matching).
cc_library(
    name = "aho_corasick",
    srcs = ["aho_corasick.cc"],
    hdrs = ["aho_corasick.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "aho_corasick_test",
    srcs = ["aho_corasick_test.cc"],
    deps = [
        ":aho_corasick",
        "@com_google_googletest//:gtest_main",
        "//support/util:test_utils",
    ],
)

# tokenizer/added_vocabulary.rs
cc_library(
    name = "added_vocabulary",
    srcs = ["added_vocabulary.cc"],
    hdrs = ["added_vocabulary.h"],
    deps = [
        ":aho_corasick",
        ":api",
        ":normalizer",
        ":pre_tokenizer",
        ":token",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "added_vocabulary_test",
    srcs = ["added_vocabulary_test.cc"],
    deps = [
        ":added_vocabulary",
        ":api",
        ":normalizer",
        ":pre_tokenizer",
        ":token",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:test_utils",
    ],
)

# tokenizer/mod.rs (lines 221-end) and tokenizer/serialization.rs
cc_library(
    name = "tokenizer",
    srcs = [
        "serialization.cc",
        "tokenizer.cc",
    ],
    hdrs = [
        "serialization.h",
        "tokenizer.h",
    ],
    deps = [
        ":added_vocabulary",
        ":api",
        ":encoding",
        ":normalizer",
        ":padding",
        ":pre_tokenizer",
        ":truncation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/processors",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.cc"],
    deps = [
        ":added_vocabulary",
        ":encoding",
        ":tokenizer",
        ":truncation",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    deps = [
        ":tokenizer",
        "@com_google_googletest//:gtest_main",
        "//support/util:test_utils",
    ],
)
