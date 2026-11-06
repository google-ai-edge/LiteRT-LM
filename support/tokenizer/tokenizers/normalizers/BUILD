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

# C++ port of `src/normalizers/` of the HuggingFace `tokenizers` crate (plus
# the `spm_precompiled` crate it re-exports as `Precompiled`).

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
    name = "normalizers",
    srcs = [
        "bert.cc",
        "byte_level.cc",
        "normalizers.cc",
        "precompiled.cc",
        "prepend.cc",
        "replace.cc",
        "spm_precompiled.cc",
        "strip.cc",
        "unicode.cc",
        "utils.cc",
    ],
    hdrs = [
        "bert.h",
        "byte_level.h",
        "normalizers.h",
        "precompiled.h",
        "prepend.h",
        "replace.h",
        "spm_precompiled.h",
        "strip.h",
        "unicode.h",
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:pattern",
        "//support/tokenizer/tokenizers/unicode:categories",
        "//support/tokenizer/tokenizers/unicode:graphemes",
        "//support/tokenizer/tokenizers/unicode:normalization",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:regex",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "normalizers_test",
    srcs = ["normalizers_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "byte_level_test",
    srcs = ["byte_level_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "precompiled_test",
    srcs = ["precompiled_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
    ],
)

cc_test(
    name = "prepend_test",
    srcs = ["prepend_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "replace_test",
    srcs = ["replace_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "strip_test",
    srcs = ["strip_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/unicode:normalization",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unicode_test",
    srcs = ["unicode_test.cc"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "spm_precompiled_test",
    srcs = ["spm_precompiled_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":normalizers",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)
