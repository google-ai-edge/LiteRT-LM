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

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "qwen_omni_model_config",
    hdrs = ["qwen_omni_model_config.h"],
    deps = [
        "//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "qwen_omni_io_types",
    hdrs = ["qwen_omni_io_types.h"],
)

cc_library(
    name = "qwen_omni_text_stage",
    srcs = ["qwen_omni_text_stage.cc"],
    hdrs = ["qwen_omni_text_stage.h"],
    deps = [
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
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
        "//support/tokenizer",
    ],
)

cc_library(
    name = "qwen_omni_audio_stage",
    srcs = ["qwen_omni_audio_stage.cc"],
    hdrs = ["qwen_omni_audio_stage.h"],
    deps = [
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "//omni:omni_session",
        "//omni/any2any:any2any_whisper_audio_stage",
        "//omni/any2any:io_types",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "qwen_omni_image_stage",
    srcs = ["qwen_omni_image_stage.cc"],
    hdrs = ["qwen_omni_image_stage.h"],
    deps = [
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/any2any:any2any_image_stage",
        "//omni/any2any:io_types",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//runtime/components/preprocessor:pil_resize",
    ],
)

cc_library(
    name = "qwen_omni_frontend_stage",
    srcs = ["qwen_omni_frontend_stage.cc"],
    hdrs = ["qwen_omni_frontend_stage.h"],
    deps = [
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        ":qwen_omni_text_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//omni/any2any:io_types",
        "//omni/base:fan_in_stage",
        "//omni/base:sequenced",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "qwen_omni_llm_stage",
    srcs = ["qwen_omni_llm_stage.cc"],
    hdrs = ["qwen_omni_llm_stage.h"],
    deps = [
        ":qwen_omni_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "//omni/any2any:any2any_llm_stage",
        "//omni/any2any:io_types",
        "//omni/base:litert_lm_runner",
        "//omni/base:stage",
        "//support/tokenizer",
    ],
)

cc_library(
    name = "qwen_omni_tts_stage",
    srcs = ["qwen_omni_tts_stage.cc"],
    hdrs = ["qwen_omni_tts_stage.h"],
    deps = [
        ":qwen_omni_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//omni/any2any:any2any_tts_decoder",
        "//omni/any2any:io_types",
        "//omni/base:litert_runner",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "qwen_omni_test_utils",
    testonly = True,
    srcs = ["qwen_omni_test_utils.cc"],
    hdrs = ["qwen_omni_test_utils.h"],
    deps = [
        ":qwen_omni_model_config",
    ],
)

cc_test(
    name = "qwen_omni_frontend_test",
    srcs = ["qwen_omni_frontend_test.cc"],
    deps = [
        ":qwen_omni_frontend_stage",
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        ":qwen_omni_test_utils",
        ":qwen_omni_text_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/any2any:any2any_input_fan_out_stage",
        "//omni/any2any:any2any_test_utils",
        "//omni/any2any:io_types",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:mock_litert_runner",
        "//omni/base:model_utils",
        "//omni/base:sequenced",
        "//omni/base:stage",
        "//runtime/executor:executor_settings_base",
        "//runtime/framework:threadpool",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "qwen_omni_audio_stage_test",
    srcs = ["qwen_omni_audio_stage_test.cc"],
    deps = [
        ":qwen_omni_audio_stage",
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        ":qwen_omni_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/any2any:any2any_input_fan_out_stage",
        "//omni/any2any:any2any_test_utils",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:mock_litert_runner",
        "//omni/base:model_utils",
        "//runtime/executor:executor_settings_base",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "qwen_omni_image_stage_test",
    srcs = ["qwen_omni_image_stage_test.cc"],
    deps = [
        ":qwen_omni_image_stage",
        ":qwen_omni_io_types",
        ":qwen_omni_model_config",
        ":qwen_omni_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni:omni_session",
        "//omni/any2any:any2any_input_fan_out_stage",
        "//omni/any2any:any2any_test_utils",
        "//omni/base:fan_out_stage",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:mock_litert_runner",
        "//omni/base:model_utils",
        "//runtime/components/preprocessor:pil_resize",
        "//runtime/executor:executor_settings_base",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "qwen_omni_llm_stage_test",
    srcs = ["qwen_omni_llm_stage_test.cc"],
    deps = [
        ":qwen_omni_llm_stage",
        ":qwen_omni_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/any2any:io_types",
        "//omni/base:mock_litert_lm_runner",
        "//omni/base:stage",
        "//runtime/executor:llm_executor_io_types",
        "//support/tokenizer",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "qwen_omni_tts_stage_test",
    srcs = ["qwen_omni_tts_stage_test.cc"],
    deps = [
        ":qwen_omni_model_config",
        ":qwen_omni_test_utils",
        ":qwen_omni_tts_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/any2any:any2any_test_utils",
        "//omni/any2any:io_types",
        "//omni/base:litert_runner",
        "//omni/base:mock_litert_runner",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//runtime/executor:executor_settings_base",
        "//support/util:convert_tensor_buffer",
        "//support/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)
