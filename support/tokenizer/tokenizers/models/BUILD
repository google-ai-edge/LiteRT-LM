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

# C++ port of `src/models/` of the HuggingFace `tokenizers` crate (v0.23.2):
# `ModelWrapper`, `BPE`, `WordPiece`, `WordLevel`, and `Unigram`.

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
    name = "models",
    srcs = [
        "bpe/bpe.cc",
        "bpe/model.cc",
        "bpe/serialization.cc",
        "bpe/word.cc",
        "models.cc",
        "unigram/lattice.cc",
        "unigram/model.cc",
        "unigram/serialization.cc",
        "unigram/trie.cc",
        "unigram/unigram.cc",
        "wordlevel/serialization.cc",
        "wordlevel/wordlevel.cc",
        "wordpiece/serialization.cc",
        "wordpiece/wordpiece.cc",
    ],
    hdrs = [
        "bpe/bpe.h",
        "bpe/model.h",
        "bpe/serialization.h",
        "bpe/word.h",
        "models.h",
        "unigram/lattice.h",
        "unigram/model.h",
        "unigram/serialization.h",
        "unigram/trie.h",
        "unigram/unigram.h",
        "wordlevel/serialization.h",
        "wordlevel/wordlevel.h",
        "wordpiece/serialization.h",
        "wordpiece/wordpiece.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/unicode:categories",
        "//support/tokenizer/tokenizers/utils:cache",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/tokenizer/tokenizers/utils:utf8",
        "//support/util:litert_status_util",
    ],
)

cc_test(
    name = "models_test",
    srcs = ["models_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "bpe_word_test",
    srcs = ["bpe/word_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bpe_model_test",
    srcs = ["bpe/model_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "bpe_serialization_test",
    srcs = ["bpe/serialization_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "wordpiece_test",
    srcs = ["wordpiece/wordpiece_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "wordpiece_serialization_test",
    srcs = ["wordpiece/serialization_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "wordlevel_test",
    srcs = ["wordlevel/wordlevel_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "wordlevel_serialization_test",
    srcs = ["wordlevel/serialization_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unigram_lattice_test",
    srcs = ["unigram/lattice_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "unigram_model_test",
    srcs = ["unigram/model_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/tokenizer:api",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "unigram_serialization_test",
    srcs = ["unigram/serialization_test.cc"],
    deps = [
        ":models",
        "@com_google_googletest//:gtest_main",
        "//support/tokenizer/tokenizers/utils:json",
        "//support/util:litert_status_util",
        "//support/util:test_utils",
    ],
)
