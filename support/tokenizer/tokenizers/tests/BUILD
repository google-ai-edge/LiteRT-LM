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

# Integration tests for the C++ port of the HuggingFace `tokenizers` crate
# (v0.23.2).

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
    name = "common",
    testonly = True,
    hdrs = ["common.h"],
    deps = [
        "@com_google_googletest//:gtest_for_library",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/processors",
        "//support/tokenizer/tokenizers/tokenizer",
    ],
)

cc_test(
    name = "added_tokens_test",
    srcs = ["added_tokens_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:added_vocabulary",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "offsets_test",
    srcs = ["offsets_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:added_vocabulary",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/processors",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unigram_test",
    srcs = ["unigram_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "stream_test",
    srcs = ["stream_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:added_vocabulary",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "documentation_test",
    srcs = ["documentation_test.cc"],
    data = ["//support/tokenizer/tokenizers/testdata"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/decoders",
        "//support/tokenizer/tokenizers/models",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/processors",
        "//support/tokenizer/tokenizers/tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:normalizer",
        "//support/tokenizer/tokenizers/tokenizer:padding",
        "//support/tokenizer/tokenizers/tokenizer:pre_tokenizer",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/util:test_utils",
    ],
)
