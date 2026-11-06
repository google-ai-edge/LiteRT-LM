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

# C++ port of the `decoders` module of the HuggingFace `tokenizers` crate
# v0.23.2 (`src/decoders/*.rs`).

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
    name = "decoders",
    srcs = [
        "bpe.cc",
        "byte_fallback.cc",
        "ctc.cc",
        "decoders.cc",
        "fuse.cc",
        "sequence.cc",
        "strip.cc",
        "wordpiece.cc",
    ],
    hdrs = [
        "bpe.h",
        "byte_fallback.h",
        "ctc.h",
        "decoders.h",
        "fuse.h",
        "sequence.h",
        "strip.h",
        "wordpiece.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/normalizers",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "bpe_test",
    srcs = ["bpe_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "byte_fallback_test",
    srcs = ["byte_fallback_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "ctc_test",
    srcs = ["ctc_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "decoders_test",
    srcs = ["decoders_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "fuse_test",
    srcs = ["fuse_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "sequence_test",
    srcs = ["sequence_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "strip_test",
    srcs = ["strip_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "wordpiece_test",
    srcs = ["wordpiece_test.cc"],
    deps = [
        ":decoders",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)
