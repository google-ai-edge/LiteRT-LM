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
        "@com_google_absl//absl/synchronization",
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
        "@litert//support/util:test_utils",
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
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//support/preprocessor:audio_preprocessor",
        "@litert//support/preprocessor:audio_preprocessor_miniaudio",
        "@litert//support/util:io_types",
        "//omni/base:stage",
    ],
)

cc_test(
    name = "log_mel_spectrogram_processor_test",
    srcs = ["log_mel_spectrogram_processor_test.cc"],
    deps = [
        ":log_mel_spectrogram_processor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//support/util:test_utils",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "speech_decoder",
    hdrs = ["speech_decoder.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "detokenizer",
    hdrs = ["detokenizer.h"],
    deps = [
        ":speech_decoder",
        "@com_google_absl//absl/base:nullability",
        "//omni/base:stage",
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
        "@com_google_absl//absl/status:statusor",
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
    name = "asr_session",
    srcs = ["asr_session.cc"],
    hdrs = ["asr_session.h"],
    deps = [
        ":audio_preprocessor",
        ":audio_source",
        ":detokenizer",
        ":speech_decoder",
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
        ":speech_decoder",
        ":text_merger",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@litert//support/util:test_utils",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
    ],
)
