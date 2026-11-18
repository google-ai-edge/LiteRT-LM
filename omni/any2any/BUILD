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

licenses(["notice"])

cc_library(
    name = "io_types",
    hdrs = ["io_types.h"],
)

cc_library(
    name = "any2any_input_fan_out_stage",
    srcs = ["any2any_input_fan_out_stage.cc"],
    hdrs = ["any2any_input_fan_out_stage.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "any2any_image_stage",
    srcs = ["any2any_image_stage.cc"],
    hdrs = ["any2any_image_stage.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//runtime/components/preprocessor:image_preprocessor",
        "//runtime/components/preprocessor:image_preprocessor_impl",  # build_cleaner: keep
    ],
)

cc_test(
    name = "any2any_image_stage_test",
    srcs = ["any2any_image_stage_test.cc"],
    deps = [
        ":any2any_image_stage",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:mock_litert_runner",
        "//omni/base:sequenced",
        "//runtime/components/preprocessor:image_preprocessor",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "any2any_whisper_audio_stage",
    srcs = ["any2any_whisper_audio_stage.cc"],
    hdrs = ["any2any_whisper_audio_stage.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//omni/utils:audio_resampler",
        "//omni/utils:whisper_log_mel_extractor",
    ],
)

cc_test(
    name = "any2any_whisper_audio_stage_test",
    srcs = ["any2any_whisper_audio_stage_test.cc"],
    deps = [
        ":any2any_whisper_audio_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:mock_litert_runner",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "any2any_test_utils",
    testonly = True,
    srcs = ["any2any_test_utils.cc"],
    hdrs = ["any2any_test_utils.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "any2any_input_fan_out_stage_test",
    srcs = ["any2any_input_fan_out_stage_test.cc"],
    deps = [
        ":any2any_input_fan_out_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:sequenced",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "any2any_session",
    srcs = ["any2any_session.cc"],
    hdrs = ["any2any_session.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:any2any_session_config_cc_proto",
    ],
)

cc_library(
    name = "minicpm_o",
    srcs = ["minicpm_o.cc"],
    deps = [
        ":any2any_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
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
        "//omni/any2any/minicpm_o:minicpm_o_factory",
        "//omni/any2any/minicpm_o:minicpm_o_model_config",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:any2any_model_type_cc_proto",
        "//runtime/proto:any2any_session_config_cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "any2any_session_test",
    srcs = ["any2any_session_test.cc"],
    data = [
        "//runtime/testdata:checksum_fake_lm_models",
    ],
    deps = [
        ":any2any_session",
        ":any2any_test_utils",
        ":minicpm_o",  # build_cleaner: keep
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@flatbuffers",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:any2any_model_type_cc_proto",
        "//runtime/proto:any2any_session_config_cc_proto",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_section",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "any2any_runner",
    srcs = ["any2any_runner.cc"],
    deps = [
        ":minicpm_o",  # build_cleaner: keep
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_cache",
        "//omni/base:model_utils",
        "//omni/speech2text:file_audio_source",
        "//runtime/executor:default_gpu_accelerator",
        "//runtime/proto:any2any_session_config_cc_proto",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)
