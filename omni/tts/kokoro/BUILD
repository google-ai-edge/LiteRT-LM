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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@espeak_ng//:espeak_ng_base_headers",
        "@espeak_ng//:espeak_ng_speak_lib_header",
        "@espeak_ng//:libespeak_ng_la_sources",  # buildcleaner: keep
    ],
)

cc_test(
    name = "phonemizer_test",
    srcs = ["phonemizer_test.cc"],
    data = [
        "@espeak_ng//:en_dictionary",
        "@espeak_ng//:es_dictionary",
        "@espeak_ng//:espeak_ng_data",
        "@espeak_ng//:fr_dictionary",
        "@espeak_ng//:hi_dictionary",
        "@espeak_ng//:it_dictionary",
        "@espeak_ng//:pt_dictionary",
        "@espeak_ng//:zh_dictionary",
    ],
    deps = [
        ":cjk_test_blobs",
        ":phonemizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
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
        "//omni/tts:stream_text_source",
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
        "//omni/tts:vocoder",
    ],
)

cc_test(
    name = "kokoro_vocoder_stage_test",
    srcs = ["kokoro_vocoder_stage_test.cc"],
    deps = [
        ":kokoro_io_types",
        ":kokoro_model_config",
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
    name = "espeak_assets",
    srcs = ["espeak_assets.cc"],
    hdrs = ["espeak_assets.h"],
    deps = [
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/components:model_resources",
        "//runtime/util:zip_utils",
    ],
)

cc_test(
    name = "espeak_assets_test",
    srcs = ["espeak_assets_test.cc"],
    deps = [
        ":espeak_assets",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/util:scoped_file",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "kokoro_factory",
    srcs = ["kokoro_factory.cc"],
    hdrs = ["kokoro_factory.h"],
    deps = [
        ":common",
        ":espeak_assets",
        ":kokoro_acoustic_stage",
        ":kokoro_model_config",
        ":kokoro_vocoder_stage",
        ":phonemizer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/tts:stream_text_source",
        "//omni/tts:text_chunk_utils",
        "//omni/tts:text_source",
        "//omni/tts:tts_session",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_test(
    name = "kokoro_factory_test",
    srcs = ["kokoro_factory_test.cc"],
    deps = [
        ":espeak_assets",
        ":kokoro_factory",
        ":kokoro_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "//omni/base:model_resources",
        "//omni/tts:stream_text_source",
        "//omni/tts:text_chunk_utils",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:scoped_file",
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
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "cjk_test_blobs",
    testonly = True,
    srcs = ["cjk_test_blobs.cc"],
    hdrs = ["cjk_test_blobs.h"],
)

cc_library(
    name = "chinese_g2p",
    srcs = ["chinese_g2p.cc"],
    hdrs = ["chinese_g2p.h"],
    deps = [
        ":cjk_blob",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cppjieba",
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
