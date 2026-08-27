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
        "@com_google_absl//absl/base:core_headers",
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
        "//support/preprocessor:audio_preprocessor",
        "//support/preprocessor:audio_preprocessor_miniaudio",
        "//support/util:io_types",
    ],
)

cc_library(
    name = "dummy_preprocessor",
    hdrs = ["dummy_preprocessor.h"],
    deps = [
        ":audio_preprocessor",
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
        ":log_mel_spectrogram_processor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//omni/base:stage",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "speech_recognizer",
    hdrs = ["speech_recognizer.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
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
        "//omni/base:litert_runner",
    ],
)

cc_test(
    name = "tdt_decoder_test",
    srcs = ["tdt_decoder_test.cc"],
    deps = [
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
    name = "litert_speech_recognizer",
    srcs = ["litert_speech_recognizer.cc"],
    hdrs = ["litert_speech_recognizer.h"],
    deps = [
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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//omni/base:stage",
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
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//omni/base:stage",
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
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "//omni/base:stage",
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
        ":detokenizer",
        ":speech_recognizer",
        ":text_merger",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//omni/base:async_stage_scheduler",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
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
        ":speech_recognizer",
        ":text_merger",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "asr_engine",
    srcs = ["asr_engine.cc"],
    hdrs = ["asr_engine.h"],
    data = ["model_metadata.json"],
    deps = [
        ":asr_session",
        ":audio_preprocessor",
        ":audio_source",
        ":ctc_decoder",
        ":dummy_preprocessor",
        ":levenshtein_text_merger",
        ":litert_speech_recognizer",
        ":log_mel_spectrogram_processor",
        ":stateless_decoder",
        ":tdt_decoder",
        ":text_merger",
        ":timestamp_text_merger",
        ":tokenizer_detokenizer",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//omni/base:litert_runner",
        "//support/tokenizer",
        "//support/tokenizer:huggingface_tokenizer",
    ],
)

cc_binary(
    name = "asr_runner",
    srcs = ["asr_runner.cc"],
    deps = [
        ":asr_engine",
        ":file_audio_source",
        ":log_mel_spectrogram_processor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/executor:default_static_gpu_accelerator",
    ],
)
