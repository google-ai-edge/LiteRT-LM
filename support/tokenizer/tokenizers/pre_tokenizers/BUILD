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

# C++ port of the `pre_tokenizers` module of the HuggingFace `tokenizers`
# crate (v0.23.2). See ../README.md.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "pre_tokenizers",
    srcs = [
        "bert.cc",
        "byte_level.cc",
        "delimiter.cc",
        "digits.cc",
        "fixed_length.cc",
        "metaspace.cc",
        "pre_tokenizers.cc",
        "punctuation.cc",
        "sequence.cc",
        "split.cc",
        "unicode_scripts/pre_tokenizer.cc",
        "unicode_scripts/scripts.cc",
        "unicode_scripts/unicode_scripts.cc",
        "whitespace.cc",
    ],
    hdrs = [
        "bert.h",
        "byte_level.h",
        "delimiter.h",
        "digits.h",
        "fixed_length.h",
        "metaspace.h",
        "pre_tokenizers.h",
        "punctuation.h",
        "sequence.h",
        "split.h",
        "unicode_scripts/pre_tokenizer.h",
        "unicode_scripts/scripts.h",
        "unicode_scripts/unicode_scripts.h",
        "whitespace.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pattern",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/unicode:categories",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "pre_tokenizers_test",
    srcs = ["pre_tokenizers_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "bert_test",
    srcs = ["bert_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "byte_level_test",
    srcs = ["byte_level_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "digits_test",
    srcs = ["digits_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "fixed_length_test",
    srcs = ["fixed_length_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "metaspace_test",
    srcs = ["metaspace_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pattern",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "punctuation_test",
    srcs = ["punctuation_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "sequence_test",
    srcs = ["sequence_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "split_test",
    srcs = ["split_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "whitespace_test",
    srcs = ["whitespace_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unicode_scripts_pre_tokenizer_test",
    srcs = ["unicode_scripts/pre_tokenizer_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unicode_scripts_scripts_test",
    srcs = ["unicode_scripts/scripts_test.cc"],
    deps = [
        ":pre_tokenizers",
        "@com_google_googletest//:gtest_main",
    ],
)
