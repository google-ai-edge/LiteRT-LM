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
    name = "qwen3_tts_model_config",
    hdrs = ["qwen3_tts_model_config.h"],
)

cc_library(
    name = "qwen3_tts_io_types",
    hdrs = ["qwen3_tts_io_types.h"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "qwen3_tts_frontend_stage",
    srcs = ["qwen3_tts_frontend_stage.cc"],
    hdrs = ["qwen3_tts_frontend_stage.h"],
    deps = [
        ":common",
        ":qwen3_tts_io_types",
        ":qwen3_tts_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/components:model_resources",
        "//support/tokenizer",
    ] + select({
        "//build_config:disable_huggingface_tokenizer": [
            "//support/tokenizer:huggingface_tokenizer_noop",
        ],
        "//conditions:default": [
            "//support/tokenizer:huggingface_tokenizer",
        ],
    }),
)

cc_library(
    name = "qwen3_tts_acoustic_predictor_stage",
    srcs = ["qwen3_tts_acoustic_predictor_stage.cc"],
    hdrs = ["qwen3_tts_acoustic_predictor_stage.h"],
    deps = [
        ":common",
        ":qwen3_tts_io_types",
        ":qwen3_tts_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_api_types",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:litert_lm_runner",
        "//omni/base:litert_runner",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/base:stateful_litert_runner",
        "//runtime/executor:llm_executor_io_types",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_test(
    name = "qwen3_tts_acoustic_predictor_stage_test",
    srcs = ["qwen3_tts_acoustic_predictor_stage_test.cc"],
    deps = [
        ":common",
        ":qwen3_tts_acoustic_predictor_stage",
        ":qwen3_tts_io_types",
        ":qwen3_tts_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "qwen3_tts_latent_decoder_stage",
    srcs = ["qwen3_tts_latent_decoder_stage.cc"],
    hdrs = ["qwen3_tts_latent_decoder_stage.h"],
    deps = [
        ":qwen3_tts_io_types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "qwen3_tts_vocoder_stage",
    srcs = ["qwen3_tts_vocoder_stage.cc"],
    hdrs = ["qwen3_tts_vocoder_stage.h"],
    deps = [
        ":common",
        ":qwen3_tts_io_types",
        ":qwen3_tts_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//omni/text2speech:vocoder",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "qwen3_tts_factory",
    srcs = ["qwen3_tts_factory.cc"],
    hdrs = ["qwen3_tts_factory.h"],
    deps = [
        ":qwen3_tts_acoustic_predictor_stage",
        ":qwen3_tts_frontend_stage",
        ":qwen3_tts_latent_decoder_stage",
        ":qwen3_tts_model_config",
        ":qwen3_tts_vocoder_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//omni/base:io_types",
        "//omni/base:litert_lm_runner",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2speech:stream_text_source",
        "//omni/text2speech:text_source",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor_factory",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/util:scoped_file",
    ],
)
