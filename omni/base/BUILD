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
    name = "io_types",
    hdrs = ["io_types.h"],
)

cc_library(
    name = "model_resources",
    srcs = ["model_resources.cc"],
    hdrs = ["model_resources.h"],
    deps = [
        ":litert_lm_runner",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
    ],
)

cc_test(
    name = "model_resources_test",
    srcs = ["model_resources_test.cc"],
    deps = [
        ":model_resources",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@litert//litert/cc:litert_environment",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    deps = [
        ":litert_lm_runner",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_intel_openvino_options",
        "@litert//litert/cc/options:litert_mediatek_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "@litert//litert/cc/options:litert_samsung_options",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor_factory",
        "//runtime/util:file_format_util",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    deps = [
        ":model_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "stage",
    hdrs = ["stage.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "stage_test",
    srcs = ["stage_test.cc"],
    deps = [
        ":stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "async_stage_scheduler",
    hdrs = ["async_stage_scheduler.h"],
    deps = [
        ":stage",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "async_stage_scheduler_test",
    srcs = ["async_stage_scheduler_test.cc"],
    deps = [
        ":async_stage_scheduler",
        ":stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/framework:threadpool",
    ],
)

cc_library(
    name = "litert_runner",
    srcs = ["litert_runner.cc"],
    hdrs = ["litert_runner.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "mock_litert_runner",
    testonly = 1,
    hdrs = ["mock_litert_runner.h"],
    deps = [
        ":litert_runner",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "litert_lm_runner",
    srcs = ["litert_lm_runner.cc"],
    hdrs = ["litert_lm_runner.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:model_resources",
        "//runtime/executor:llm_executor_base",
        "//runtime/executor:llm_executor_io_types",
    ],
)

cc_library(
    name = "mock_litert_lm_runner",
    testonly = 1,
    hdrs = ["mock_litert_lm_runner.h"],
    deps = [
        ":litert_lm_runner",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor:llm_executor_io_types",
    ],
)

cc_test(
    name = "litert_lm_runner_test",
    srcs = ["litert_lm_runner_test.cc"],
    deps = [
        ":mock_litert_lm_runner",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor:llm_executor_io_types",
    ],
)

cc_library(
    name = "stateful_litert_runner",
    srcs = ["stateful_litert_runner.cc"],
    hdrs = ["stateful_litert_runner.h"],
    deps = [
        ":litert_runner",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "mock_stateful_litert_runner",
    testonly = 1,
    hdrs = ["mock_stateful_litert_runner.h"],
    deps = [
        ":stateful_litert_runner",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "stateful_litert_runner_test",
    srcs = ["stateful_litert_runner_test.cc"],
    deps = [
        ":mock_litert_runner",
        ":stateful_litert_runner",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model_types",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
    ],
)
