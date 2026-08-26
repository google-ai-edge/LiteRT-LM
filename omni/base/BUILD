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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/util:scoped_file",
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
        "@litert//litert/cc:litert_environment",
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
