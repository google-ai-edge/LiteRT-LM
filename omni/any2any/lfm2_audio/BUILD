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
    name = "lfm2_audio_model_config",
    hdrs = ["lfm2_audio_model_config.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "lfm2_audio_stage",
    srcs = ["lfm2_audio_stage.cc"],
    hdrs = ["lfm2_audio_stage.h"],
    deps = [
        ":lfm2_audio_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_lm_runner",
        "//omni/base:litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//omni/utils:audio_resampler",
        "//runtime/executor:llm_executor_io_types",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "lfm2_audio_factory",
    srcs = ["lfm2_audio_factory.cc"],
    hdrs = ["lfm2_audio_factory.h"],
    deps = [
        ":lfm2_audio_model_config",
        ":lfm2_audio_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni:omni_session",
        "//omni/any2any:any2any_session",
        "//omni/base:io_types",
        "//omni/base:litert_lm_runner",
        "//omni/base:litert_runner",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "lfm2_audio_test",
    srcs = ["lfm2_audio_test.cc"],
    data = [
        "//runtime/testdata:checksum_fake_lm_models",
    ],
    deps = [
        ":lfm2_audio_factory",
        ":lfm2_audio_model_config",
        ":lfm2_audio_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/any2any:any2any_session",
        "//omni/any2any:lfm2_audio",  # build_cleaner: keep
        "//omni/base:io_types",
        "//omni/base:litert_lm_runner",
        "//omni/base:mock_litert_runner",
        "//omni/base:model_resources",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:any2any_model_type_cc_proto",
        "//runtime/proto:any2any_session_config_cc_proto",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_section",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)
