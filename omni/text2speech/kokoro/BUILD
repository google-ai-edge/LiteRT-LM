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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "kokoro_model_config",
    hdrs = ["kokoro_model_config.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":kokoro_io_types",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//runtime/components:model_resources",
    ],
)

cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    deps = [
        ":common",
        ":kokoro_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:scoped_file",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "phonemizer",
    srcs = ["phonemizer.cc"],
    hdrs = ["phonemizer.h"],
    deps = [
        ":chinese_g2p",
        ":cjk_blob",
        ":common",
        ":english_g2p",
        ":french_g2p",
        ":hindi_g2p",
        ":italian_g2p",
        ":japanese_g2p",
        ":language_g2p",
        ":portuguese_g2p",
        ":spanish_g2p",
        ":traditional_chinese_converter",
        ":unicode_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//omni/text2speech:text_normalizer",
    ],
)

cc_test(
    name = "phonemizer_test",
    srcs = ["phonemizer_test.cc"],
    data = [
        "//omni/text2speech:text_normalizer_rule_tables",
    ],
    deps = [
        ":cjk_test_blobs",
        ":phonemizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "kokoro_io_types",
    hdrs = ["kokoro_io_types.h"],
)

cc_library(
    name = "kokoro_acoustic_stage",
    srcs = ["kokoro_acoustic_stage.cc"],
    hdrs = ["kokoro_acoustic_stage.h"],
    deps = [
        ":common",
        ":kokoro_io_types",
        ":kokoro_model_config",
        ":phonemizer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/components:model_resources",
    ],
)

cc_test(
    name = "kokoro_acoustic_stage_test",
    srcs = ["kokoro_acoustic_stage_test.cc"],
    deps = [
        ":kokoro_acoustic_stage",
        ":kokoro_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_environment",
        "//omni/base:model_resources",
        "//omni/text2speech:stream_text_source",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "kokoro_vocoder_stage",
    srcs = ["kokoro_vocoder_stage.cc"],
    hdrs = ["kokoro_vocoder_stage.h"],
    deps = [
        ":common",
        ":kokoro_io_types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@kissfft//:kissfftr",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2speech:vocoder",
    ],
)

cc_test(
    name = "kokoro_vocoder_stage_test",
    srcs = ["kokoro_vocoder_stage_test.cc"],
    deps = [
        ":kokoro_io_types",
        ":kokoro_vocoder_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_environment",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "kokoro_factory",
    srcs = ["kokoro_factory.cc"],
    hdrs = ["kokoro_factory.h"],
    deps = [
        ":common",
        ":kokoro_acoustic_stage",
        ":kokoro_model_config",
        ":kokoro_vocoder_stage",
        ":phonemizer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2speech:stream_text_source",
        "//omni/text2speech:text_chunk_utils",
        "//omni/text2speech:text_source",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_test(
    name = "kokoro_factory_test",
    srcs = ["kokoro_factory_test.cc"],
    deps = [
        ":common",
        ":kokoro_factory",
        ":kokoro_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//omni/text2speech:stream_text_source",
        "//omni/text2speech:text_chunk_utils",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:scoped_file",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "cjk_blob",
    srcs = ["cjk_blob.cc"],
    hdrs = ["cjk_blob.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cjk_blob_test",
    srcs = ["cjk_blob_test.cc"],
    deps = [
        ":cjk_blob",
        ":cjk_test_blobs",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "cjk_lattice",
    srcs = ["cjk_lattice.cc"],
    hdrs = ["cjk_lattice.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "cjk_lattice_test",
    srcs = ["cjk_lattice_test.cc"],
    deps = [
        ":cjk_lattice",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "cjk_test_blobs",
    testonly = True,
    srcs = ["cjk_test_blobs.cc"],
    hdrs = ["cjk_test_blobs.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_library(
    name = "mecab_lattice",
    srcs = ["mecab_lattice.cc"],
    hdrs = ["mecab_lattice.h"],
    deps = [
        ":cjk_blob",
        ":cjk_lattice",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "mecab_lattice_test",
    srcs = ["mecab_lattice_test.cc"],
    deps = [
        ":cjk_blob",
        ":cjk_lattice",
        ":cjk_test_blobs",
        ":mecab_lattice",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "japanese_g2p",
    srcs = ["japanese_g2p.cc"],
    hdrs = ["japanese_g2p.h"],
    deps = [
        ":cjk_blob",
        ":cjk_lattice",
        ":mecab_lattice",
        ":unicode_utils",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "japanese_g2p_test",
    srcs = ["japanese_g2p_test.cc"],
    deps = [
        ":cjk_blob",
        ":cjk_test_blobs",
        ":japanese_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "chinese_g2p",
    srcs = ["chinese_g2p.cc"],
    hdrs = ["chinese_g2p.h"],
    deps = [
        ":cjk_blob",
        ":unicode_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cppjieba",
    ],
)

cc_library(
    name = "traditional_chinese_converter",
    srcs = ["traditional_chinese_converter.cc"],
    hdrs = ["traditional_chinese_converter.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "traditional_chinese_converter_test",
    srcs = ["traditional_chinese_converter_test.cc"],
    data = [
    ],
    deps = [
        ":traditional_chinese_converter",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "chinese_g2p_test",
    srcs = ["chinese_g2p_test.cc"],
    deps = [
        ":chinese_g2p",
        ":cjk_blob",
        ":cjk_test_blobs",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "language_g2p",
    srcs = ["language_g2p.cc"],
    hdrs = ["language_g2p.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "unicode_utils",
    srcs = ["unicode_utils.cc"],
    hdrs = ["unicode_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "unicode_utils_test",
    srcs = ["unicode_utils_test.cc"],
    deps = [
        ":unicode_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "english_g2p",
    srcs = ["english_g2p.cc"],
    hdrs = ["english_g2p.h"],
    deps = [
        ":cjk_blob",
        ":unicode_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "english_g2p_test",
    srcs = ["english_g2p_test.cc"],
    deps = [
        ":cjk_blob",
        ":cjk_test_blobs",
        ":english_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "spanish_g2p",
    srcs = ["spanish_g2p.cc"],
    hdrs = ["spanish_g2p.h"],
    deps = [
        ":language_g2p",
        ":unicode_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "spanish_g2p_test",
    srcs = ["spanish_g2p_test.cc"],
    deps = [
        ":spanish_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "hindi_g2p",
    srcs = ["hindi_g2p.cc"],
    hdrs = ["hindi_g2p.h"],
    deps = [
        ":language_g2p",
        ":unicode_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "hindi_g2p_test",
    srcs = ["hindi_g2p_test.cc"],
    deps = [
        ":hindi_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "portuguese_g2p",
    srcs = ["portuguese_g2p.cc"],
    hdrs = ["portuguese_g2p.h"],
    deps = [
        ":language_g2p",
        ":unicode_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "portuguese_g2p_test",
    srcs = ["portuguese_g2p_test.cc"],
    deps = [
        ":portuguese_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "italian_g2p",
    srcs = ["italian_g2p.cc"],
    hdrs = ["italian_g2p.h"],
    deps = [
        ":language_g2p",
        ":unicode_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "italian_g2p_test",
    srcs = ["italian_g2p_test.cc"],
    deps = [
        ":italian_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_library(
    name = "french_g2p",
    srcs = ["french_g2p.cc"],
    hdrs = ["french_g2p.h"],
    deps = [
        ":language_g2p",
        ":unicode_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "french_g2p_test",
    srcs = ["french_g2p_test.cc"],
    deps = [
        ":french_g2p",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
    ],
)
