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
    name = "omni_session",
    hdrs = ["omni_session.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "multi_staged_session",
    srcs = ["multi_staged_session.cc"],
    hdrs = ["multi_staged_session.h"],
    deps = [
        ":omni_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//omni/base:async_stage_scheduler",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "multi_staged_session_test",
    srcs = ["multi_staged_session_test.cc"],
    deps = [
        ":multi_staged_session",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "omni_engine_header",
    hdrs = ["omni_engine.h"],
    deps = [
        ":omni_session",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "omni_engine",
    srcs = ["omni_engine.cc"],
    hdrs = ["omni_engine.h"],
    deps = [
        ":omni_session",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//omni/asr:asr_engine",
        "//omni/asr:asr_session",
        "//omni/asr:model_metadata",
        "//omni/text2image:text2image_engine",
        "//omni/text2image:text2image_session",
        "//omni/text2image/flux2:flux2_model_config",
        "//omni/tts:tts_engine",
        "//omni/tts:tts_session",
        "//omni/tts/kokoro:kokoro_model_config",
        "//omni/tts/qwen3_tts:qwen3_tts_model_config",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_test(
    name = "omni_engine_test",
    srcs = ["omni_engine_test.cc"],
    deps = [
        ":multi_staged_session",
        ":omni_engine",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "//omni/asr:asr_session",
        "//omni/asr:audio_preprocessor",
        "//omni/asr:audio_source",
        "//omni/asr:detokenizer",
        "//omni/asr:levenshtein_text_merger",
        "//omni/asr:speech_recognizer",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/tts:stream_text_source",
        "//omni/tts:text_chunk_utils",
        "//omni/tts:text_source",
        "//omni/tts:tts_session",
        "//omni/tts:vocoder",
        "//runtime/framework:threadpool",
        "//support/util:test_utils",
    ],
)
