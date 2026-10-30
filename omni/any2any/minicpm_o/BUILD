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
    name = "minicpm_o_model_config",
    hdrs = ["minicpm_o_model_config.h"],
    deps = [
        "//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "minicpm_o_io_types",
    hdrs = ["minicpm_o_io_types.h"],
)

cc_library(
    name = "minicpm_o_fan_out_stage",
    srcs = ["minicpm_o_fan_out_stage.cc"],
    hdrs = ["minicpm_o_fan_out_stage.h"],
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
    name = "minicpm_o_text_stage",
    srcs = ["minicpm_o_text_stage.cc"],
    hdrs = ["minicpm_o_text_stage.h"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
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
        "//support/tokenizer",
    ],
)

cc_library(
    name = "minicpm_o_audio_stage",
    srcs = ["minicpm_o_audio_stage.cc"],
    hdrs = ["minicpm_o_audio_stage.h"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
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

cc_library(
    name = "minicpm_o_image_stage",
    srcs = ["minicpm_o_image_stage.cc"],
    hdrs = ["minicpm_o_image_stage.h"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
        "//runtime/components/preprocessor:minicpmv_image_preprocess",
    ],
)

cc_library(
    name = "minicpm_o_frontend_stage",
    srcs = ["minicpm_o_frontend_stage.cc"],
    hdrs = ["minicpm_o_frontend_stage.h"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
        ":minicpm_o_text_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//omni/base:fan_in_stage",
        "//omni/base:sequenced",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "minicpm_o_llm_stage",
    srcs = ["minicpm_o_llm_stage.cc"],
    hdrs = ["minicpm_o_llm_stage.h"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_lm_runner",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/executor:llm_executor_io_types",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "minicpm_o_llm_stage_test",
    srcs = ["minicpm_o_llm_stage_test.cc"],
    deps = [
        ":minicpm_o_io_types",
        ":minicpm_o_llm_stage",
        ":minicpm_o_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//omni/base:mock_litert_lm_runner",
        "//omni/base:stage",
        "//runtime/executor:llm_executor_io_types",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
    ],
)

cc_test(
    name = "minicpm_o_frontend_test",
    srcs = ["minicpm_o_frontend_test.cc"],
    deps = [
        ":minicpm_o_audio_stage",
        ":minicpm_o_fan_out_stage",
        ":minicpm_o_frontend_stage",
        ":minicpm_o_image_stage",
        ":minicpm_o_io_types",
        ":minicpm_o_model_config",
        ":minicpm_o_text_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:mock_litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
    ],
)
