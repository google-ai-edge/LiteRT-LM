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
    name = "audio_source",
    hdrs = ["audio_source.h"],
    deps = ["//omni/base:stage"],
)

cc_library(
    name = "file_audio_source",
    srcs = ["file_audio_source.cc"],
    hdrs = ["file_audio_source.h"],
    deps = [
        ":audio_source",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)

cc_test(
    name = "file_audio_source_test",
    srcs = ["file_audio_source_test.cc"],
    deps = [
        ":file_audio_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//support/util:test_utils",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)

cc_library(
    name = "audio_preprocessor",
    hdrs = ["audio_preprocessor.h"],
    deps = [
        ":audio_source",
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "log_mel_spectrogram_processor",
    srcs = ["log_mel_spectrogram_processor.cc"],
    hdrs = ["log_mel_spectrogram_processor.h"],
    deps = [
        ":audio_preprocessor",
        ":audio_source",
        ":utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:stage",
        "//omni/utils:audio_utils",
        "//support/preprocessor:audio_preprocessor",
        "//support/preprocessor:audio_preprocessor_miniaudio",
        "//support/util:io_types",
    ],
)

cc_library(
    name = "dummy_preprocessor",
    srcs = ["dummy_preprocessor.cc"],
    hdrs = ["dummy_preprocessor.h"],
    deps = [
        ":audio_preprocessor",
        ":audio_source",
        ":utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "//omni/base:stage",
    ],
)

cc_test(
    name = "log_mel_spectrogram_processor_test",
    srcs = ["log_mel_spectrogram_processor_test.cc"],
    deps = [
        ":audio_source",
        ":dummy_preprocessor",
        ":log_mel_spectrogram_processor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "speech_recognizer",
    hdrs = ["speech_recognizer.h"],
    deps = [
        ":audio_preprocessor",
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":speech_recognizer",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":speech_recognizer",
        ":utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ctc_decoder",
    srcs = ["ctc_decoder.cc"],
    hdrs = ["ctc_decoder.h"],
    deps = [
        ":litert_speech_recognizer",
        ":speech_recognizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "ctc_decoder_test",
    srcs = ["ctc_decoder_test.cc"],
    deps = [
        ":audio_preprocessor",
        ":ctc_decoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "stateless_decoder",
    srcs = ["stateless_decoder.cc"],
    hdrs = ["stateless_decoder.h"],
    deps = [
        ":litert_speech_recognizer",
        ":speech_recognizer",
        ":utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_runner",
    ],
)

cc_test(
    name = "stateless_decoder_test",
    srcs = ["stateless_decoder_test.cc"],
    deps = [
        ":stateless_decoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:mock_litert_runner",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "tdt_decoder",
    srcs = ["tdt_decoder.cc"],
    hdrs = ["tdt_decoder.h"],
    deps = [
        ":litert_speech_recognizer",
        ":speech_recognizer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//omni/base:litert_runner",
        "//omni/base:stateful_litert_runner",
    ],
)

cc_test(
    name = "tdt_decoder_test",
    srcs = ["tdt_decoder_test.cc"],
    deps = [
        ":audio_preprocessor",
        ":tdt_decoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:mock_litert_runner",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "lm_decoder",
    srcs = ["lm_decoder.cc"],
    hdrs = ["lm_decoder.h"],
    deps = [
        ":litert_speech_recognizer",
        ":speech_recognizer",
        ":utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_lm_runner",
        "//omni/base:model_utils",
        "//runtime/proto:token_cc_proto",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_test(
    name = "lm_decoder_test",
    srcs = ["lm_decoder_test.cc"],
    deps = [
        ":lm_decoder",
        ":speech_recognizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:mock_litert_lm_runner",
        "//runtime/executor:llm_executor_io_types",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "lm_engine_decoder",
    srcs = ["lm_engine_decoder.cc"],
    hdrs = ["lm_engine_decoder.h"],
    deps = [
        ":litert_speech_recognizer",
        ":speech_recognizer",
        ":utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_lm_engine_runner",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//runtime/proto:token_cc_proto",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_test(
    name = "lm_engine_decoder_test",
    srcs = ["lm_engine_decoder_test.cc"],
    deps = [
        ":lm_engine_decoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_lm_engine_runner",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "litert_speech_recognizer",
    srcs = ["litert_speech_recognizer.cc"],
    hdrs = ["litert_speech_recognizer.h"],
    deps = [
        ":audio_preprocessor",
        ":speech_recognizer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_runner",
        "//omni/base:stage",
    ],
)

cc_test(
    name = "litert_speech_recognizer_test",
    srcs = ["litert_speech_recognizer_test.cc"],
    deps = [
        ":audio_preprocessor",
        ":litert_speech_recognizer",
        ":speech_recognizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:mock_litert_runner",
        "//omni/base:stage",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "detokenizer",
    hdrs = ["detokenizer.h"],
    deps = [
        ":speech_recognizer",
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "tokenizer_detokenizer",
    srcs = ["tokenizer_detokenizer.cc"],
    hdrs = ["tokenizer_detokenizer.h"],
    deps = [
        ":detokenizer",
        ":speech_recognizer",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//omni/base:stage",
        "//omni/utils:text_utils",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "tokenizer_detokenizer_test",
    srcs = ["tokenizer_detokenizer_test.cc"],
    deps = [
        ":speech_recognizer",
        ":tokenizer_detokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//omni/base:stage",
        "//support/tokenizer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "text_merger",
    hdrs = ["text_merger.h"],
    deps = [
        ":detokenizer",
        "@com_google_absl//absl/base:nullability",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "levenshtein_align",
    srcs = ["levenshtein_align.cc"],
    hdrs = ["levenshtein_align.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//omni/utils:text_utils",
    ],
)

cc_test(
    name = "levenshtein_align_test",
    srcs = ["levenshtein_align_test.cc"],
    deps = [
        ":levenshtein_align",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "levenshtein_text_merger",
    srcs = ["levenshtein_text_merger.cc"],
    hdrs = ["levenshtein_text_merger.h"],
    deps = [
        ":detokenizer",
        ":levenshtein_align",
        ":text_merger",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//omni/base:stage",
        "//omni/utils:text_utils",
    ],
)

cc_test(
    name = "levenshtein_text_merger_test",
    srcs = ["levenshtein_text_merger_test.cc"],
    deps = [
        ":detokenizer",
        ":levenshtein_text_merger",
        ":text_merger",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "timestamp_text_merger",
    srcs = ["timestamp_text_merger.cc"],
    hdrs = ["timestamp_text_merger.h"],
    deps = [
        ":detokenizer",
        ":levenshtein_align",
        ":text_merger",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "//omni/base:stage",
        "//omni/utils:text_utils",
    ],
)

cc_test(
    name = "timestamp_text_merger_test",
    srcs = ["timestamp_text_merger_test.cc"],
    deps = [
        ":detokenizer",
        ":text_merger",
        ":timestamp_text_merger",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
        "//omni/base:stage",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "asr_session",
    srcs = ["asr_session.cc"],
    hdrs = ["asr_session.h"],
    deps = [
        ":audio_preprocessor",
        ":audio_source",
        ":dummy_preprocessor",
        ":levenshtein_text_merger",
        ":litert_speech_recognizer",
        ":log_mel_spectrogram_processor",
        ":text_merger",
        ":timestamp_text_merger",
        ":tokenizer_detokenizer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_environment",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_session",
        "//omni/base:litert_runner",
        "//omni/base:stage",
        "//omni/utils:audio_resampler",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "asr_session_test",
    srcs = ["asr_session_test.cc"],
    deps = [
        ":asr_session",
        ":audio_preprocessor",
        ":audio_source",
        ":detokenizer",
        ":levenshtein_text_merger",
        ":moonshine",  # build_cleaner: keep
        ":parakeet_ctc",  # build_cleaner: keep
        ":parakeet_tdt",  # build_cleaner: keep
        ":qwen3_asr",  # build_cleaner: keep
        ":speech_recognizer",
        ":tinygemma_asr",  # build_cleaner: keep
        ":whisper",  # build_cleaner: keep
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//support/tokenizer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "parakeet_tdt",
    srcs = ["parakeet_tdt.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":litert_speech_recognizer",
        ":speech_recognizer",
        ":tdt_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_api_types",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_runtime_options",
        "@litert//litert/experimental/custom_ops/bidirectional_lstm:bidirectional_lstm_custom_op",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_runner",
        "//omni/base:model_cache",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//support/tokenizer",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parakeet_ctc",
    srcs = ["parakeet_ctc.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":ctc_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_runner",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//support/tokenizer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "moonshine",
    srcs = ["moonshine.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":stateless_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_runner",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//support/tokenizer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "whisper",
    srcs = ["whisper.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":stateless_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_runner",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//support/tokenizer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "qwen3_asr",
    srcs = ["qwen3_asr.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":lm_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_lm_runner",
        "//omni/base:litert_runner",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//support/tokenizer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tinygemma_asr",
    srcs = ["tinygemma_asr.cc"],
    deps = [
        ":asr_session",
        ":audio_source",
        ":lm_engine_decoder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:litert_lm_engine_runner",
        "//omni/base:litert_runner",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/core:engine_advanced_impl",  # buildcleaner: keep
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_model_type_cc_proto",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//support/tokenizer",
    ],
    alwayslink = 1,
)

cc_test(
    name = "asr_registration_test",
    srcs = ["asr_registration_test.cc"],
    deps = [
        ":parakeet_tdt",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//omni:omni_registry",
        "//support/util:test_utils",
    ],
)

cc_binary(
    name = "speech2text_runner",
    srcs = ["speech2text_runner.cc"],
    deps = [
        ":file_audio_source",
        ":moonshine",  # build_cleaner: keep
        ":parakeet_ctc",  # build_cleaner: keep
        ":parakeet_tdt",  # build_cleaner: keep
        ":qwen3_asr",  # build_cleaner: keep
        ":tinygemma_asr",  # build_cleaner: keep
        ":whisper",  # build_cleaner: keep
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:model_utils",
        "//runtime/executor:default_static_gpu_accelerator",
        "//runtime/proto:speech2text_session_config_cc_proto",
    ],
)
