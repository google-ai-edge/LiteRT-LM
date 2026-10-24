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
    name = "omni_session",
    hdrs = ["omni_session.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//omni/base:io_types",
        "//omni/base:stage",
        "@com_google_protobuf//:protobuf_lite",
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
    name = "omni_registry",
    srcs = ["omni_registry.cc"],
    hdrs = ["omni_registry.h"],
    deps = [
        ":omni_engine_header",
        ":omni_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/c/options:litert_gpu_options",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//omni/base:model_cache",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
    ],
)

cc_test(
    name = "omni_registry_test",
    srcs = ["omni_registry_test.cc"],
    deps = [
        ":omni_engine_header",
        ":omni_registry",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",  # build_cleaner: keep
        "@litert//litert/c:litert_common",
        "@litert//litert/c/options:litert_gpu_options",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/components:model_resources",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
        "//runtime/util:file_util",
        "//schema/core:litertlm_header_schema",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "omni_engine",
    srcs = ["omni_engine.cc"],
    hdrs = ["omni_engine.h"],
    deps = [
        ":omni_registry",
        ":omni_session",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "omni_engine_test",
    srcs = ["omni_engine_test.cc"],
    deps = [
        ":multi_staged_session",
        ":omni_engine",
        ":omni_registry",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/speech2text:asr_session",
        "//omni/speech2text:audio_preprocessor",
        "//omni/speech2text:audio_source",
        "//omni/speech2text:detokenizer",
        "//omni/speech2text:levenshtein_text_merger",
        "//omni/speech2text:moonshine",  # build_cleaner: keep
        "//omni/speech2text:speech_recognizer",
        "//omni/text2image:bonsai_flux2",  # build_cleaner: keep
        "//omni/text2speech:kokoro",  # build_cleaner: keep
        "//omni/text2speech:stream_text_source",
        "//omni/text2speech:text_chunk_utils",
        "//omni/text2speech:text_source",
        "//omni/text2speech:tts_session",
        "//omni/text2speech:vocoder",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
        "//support/util:test_utils",
    ],
)

cc_binary(
    name = "omni_runner",
    srcs = ["omni_runner.cc"],
    deps = [
        ":omni_engine",
        ":omni_registry",
        ":omni_session",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//omni/base:io_types",
        "//omni/base:model_utils",
        "//omni/speech2text:moonshine",  # build_cleaner: keep
        "//omni/speech2text:parakeet_ctc",  # build_cleaner: keep
        "//omni/speech2text:parakeet_tdt",  # build_cleaner: keep
        "//omni/speech2text:qwen3_asr",  # build_cleaner: keep
        "//omni/speech2text:whisper",  # build_cleaner: keep
        "//omni/text2image:bonsai_flux2",  # build_cleaner: keep
        "//omni/text2image:flux2_klein",  # build_cleaner: keep
        "//omni/text2speech:kokoro",  # build_cleaner: keep
        "//omni/text2speech:qwen3_tts",  # build_cleaner: keep
        "//runtime/executor:default_static_gpu_accelerator",
        "//runtime/proto:speech2text_session_config_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
        "//runtime/proto:text2speech_session_config_cc_proto",
        "@stb//:stb_image_write",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)
