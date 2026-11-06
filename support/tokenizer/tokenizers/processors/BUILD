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

# C++ port of the `processors` module of the HuggingFace `tokenizers` crate
# v0.23.2 (`src/processors/*.rs`).

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
    name = "processors",
    srcs = [
        "bert.cc",
        "processors.cc",
        "roberta.cc",
        "sequence.cc",
        "template.cc",
    ],
    hdrs = [
        "bert.h",
        "processors.h",
        "roberta.h",
        "sequence.h",
        "template.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//support/tokenizer/tokenizers/pre_tokenizers",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "bert_test",
    srcs = ["bert_test.cc"],
    deps = [
        ":processors",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "processors_test",
    srcs = ["processors_test.cc"],
    deps = [
        ":processors",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "roberta_test",
    srcs = ["roberta_test.cc"],
    deps = [
        ":processors",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "sequence_test",
    srcs = ["sequence_test.cc"],
    deps = [
        ":processors",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "template_test",
    srcs = ["template_test.cc"],
    deps = [
        ":processors",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/tokenizer:encoding",
        "//support/tokenizer/tokenizers/tokenizer:token",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:test_utils",
    ],
)
