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
        "//omni/base:fan_in_stage",
        "//omni/base:sequenced",
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
