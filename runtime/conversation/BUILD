# Copyright 2025 The ODML Authors.
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
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "io_types",
    hdrs = ["io_types.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "internal_callbacks_adapter",
    srcs = ["internal_callbacks_adapter.cc"],
    hdrs = ["internal_callbacks_adapter.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/engine:io_types",
    ],
)

cc_test(
    name = "internal_callbacks_adapter_test",
    srcs = ["internal_callbacks_adapter_test.cc"],
    deps = [
        ":internal_callbacks_adapter",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:gemma3_data_processor",
        "//runtime/conversation/model_data_processor:gemma3_data_processor_config",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "conversation",
    srcs = ["conversation.cc"],
    hdrs = ["conversation.h"],
    deps = [
        ":internal_callbacks_adapter",
        ":io_types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:model_data_processor_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "conversation_test",
    srcs = ["conversation_test.cc"],
    data = [
        "//runtime/components/testdata",
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":conversation",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/core:session_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:litert_lm_lib",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:test_utils",
    ],
)
