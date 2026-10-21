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

# [Google-internal load of `cc_binary`]
# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "text_chunk_utils",
    srcs = ["text_chunk_utils.cc"],
    hdrs = ["text_chunk_utils.h"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

# Rule tables consumed by :text_normalizer. These ship as data alongside the
# model, so that language-specific normalization stays out of the binary.
filegroup(
    name = "text_normalizer_rule_tables",
    srcs = [
        "data/en_textnorm.txt",
        "data/es_textnorm.txt",
        "data/fr_textnorm.txt",
        "data/hi_textnorm.txt",
        "data/it_textnorm.txt",
        "data/ja_textnorm.txt",
        "data/pt_textnorm.txt",
        "data/zh_textnorm.txt",
    ],
)

cc_library(
    name = "text_normalizer",
    srcs = ["text_normalizer.cc"],
    hdrs = ["text_normalizer.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "text_normalizer_test",
    srcs = ["text_normalizer_test.cc"],
    data = [":text_normalizer_rule_tables"],
    deps = [
        ":text_normalizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "text_source",
    hdrs = ["text_source.h"],
    deps = ["//omni/base:stage"],
)

cc_library(
    name = "file_text_source",
    srcs = ["file_text_source.cc"],
    hdrs = ["file_text_source.h"],
    deps = [
        ":text_chunk_utils",
        ":text_source",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "stream_text_source",
    srcs = ["stream_text_source.cc"],
    hdrs = ["stream_text_source.h"],
    deps = [
        ":text_chunk_utils",
        ":text_source",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "vocoder",
    hdrs = ["vocoder.h"],
    deps = [
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "tts_session",
    srcs = ["tts_session.cc"],
    hdrs = ["tts_session.h"],
    deps = [
        ":stream_text_source",
        ":text_chunk_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_session",
        "//omni/base:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
    ],
)

cc_library(
    name = "kokoro",
    srcs = ["kokoro.cc"],
    hdrs = ["kokoro.h"],
    deps = [
        ":stream_text_source",
        ":text_chunk_utils",
        ":tts_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2speech/kokoro:kokoro_factory",
        "//omni/text2speech/kokoro:kokoro_model_config",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/proto:text2speech_model_type_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "qwen3_tts",
    srcs = ["qwen3_tts.cc"],
    deps = [
        ":stream_text_source",
        ":tts_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2speech/qwen3_tts:common",
        "//omni/text2speech/qwen3_tts:qwen3_tts_factory",
        "//omni/text2speech/qwen3_tts:qwen3_tts_model_config",
        "//runtime/components:model_resources",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/proto:text2speech_model_type_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tts_session_test",
    srcs = ["tts_session_test.cc"],
    deps = [
        ":kokoro",
        ":qwen3_tts",  # build_cleaner: keep
        ":stream_text_source",
        ":text_chunk_utils",
        ":text_source",
        ":tts_session",
        ":vocoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_environment",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/text2speech/kokoro:common",
        "//omni/text2speech/kokoro:kokoro_factory",
        "//omni/text2speech/qwen3_tts:common",
        "//runtime/framework:threadpool",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "text_chunk_utils_test",
    srcs = ["text_chunk_utils_test.cc"],
    deps = [
        ":text_chunk_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "file_text_source_test",
    srcs = ["file_text_source_test.cc"],
    deps = [
        ":file_text_source",
        ":text_chunk_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "stream_text_source_test",
    srcs = ["stream_text_source_test.cc"],
    deps = [
        ":stream_text_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
    ],
)

cc_binary(
    name = "text2speech_runner",
    srcs = ["text2speech_runner.cc"],
    deps = [
        ":kokoro",  # build_cleaner: keep
        ":qwen3_tts",  # build_cleaner: keep
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_utils",
        "//omni/text2speech/qwen3_tts:common",
        "//runtime/executor:default_static_gpu_accelerator",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/proto:text2speech_model_type_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)
