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
    deps = ["@com_google_absl//absl/strings"],
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
        "@com_google_absl//absl/status",
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
        ":vocoder",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//omni/base:async_stage_scheduler",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "tts_session_test",
    srcs = ["tts_session_test.cc"],
    deps = [
        ":stream_text_source",
        ":tts_session",
        ":vocoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
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

cc_library(
    name = "tts_engine",
    srcs = ["tts_engine.cc"],
    hdrs = ["tts_engine.h"],
    deps = [
        ":text_chunk_utils",
        ":tts_session",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/tts/kokoro:kokoro_factory",
        "//omni/tts/kokoro:kokoro_model_config",
        "//omni/tts/qwen3_tts:qwen3_tts_factory",
        "//omni/tts/qwen3_tts:qwen3_tts_model_config",
        "//runtime/executor:executor_settings_base",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "tts_engine_test",
    srcs = ["tts_engine_test.cc"],
    data = [
        "@espeak_ng//:espeak_ng_data",
    ],
    deps = [
        ":stream_text_source",
        ":tts_engine",
        ":tts_session",
        ":vocoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/tts/kokoro:common",
        "//omni/tts/kokoro:phonemizer",
        "//runtime/framework:threadpool",
        "//support/util:test_utils",
    ],
)
