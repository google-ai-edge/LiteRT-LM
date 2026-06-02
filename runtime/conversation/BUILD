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
        "//visibility:public",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = ["@nlohmann_json//:json"],
)

cc_library(
    name = "prompt_utils",
    srcs = ["prompt_utils.cc"],
    hdrs = ["prompt_utils.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation/model_data_processor",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "channel_util",
    srcs = ["channel_util.cc"],
    hdrs = ["channel_util.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/engine:io_types",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "internal_callback_util",
    srcs = ["internal_callback_util.cc"],
    hdrs = ["internal_callback_util.h"],
    deps = [
        ":channel_util",
        ":io_types",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/engine:io_types",
    ],
)

cc_test(
    name = "internal_callback_util_test",
    srcs = ["internal_callback_util_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    deps = [
        ":internal_callback_util",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        ":channel_util",
        ":internal_callback_util",
        ":io_types",
        ":prompt_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:constraint_provider",
        "//runtime/components/constrained_decoding:constraint_provider_config",
        "//runtime/components/constrained_decoding:constraint_provider_factory",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:model_data_processor_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_type_utils",
    ],
)

cc_test(
    name = "channel_util_test",
    srcs = ["channel_util_test.cc"],
    deps = [
        ":channel_util",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "conversation_test",
    size = "medium",
    srcs = ["conversation_test.cc"],
    data = [
        "//runtime/components/preprocessor/testdata",
        "//runtime/components/testdata",
        "//runtime/testdata",
    ],
    shard_count = 4,
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":conversation",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:bitmap",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:external_constraint_config",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/engine:litert_lm_lib",  # buildcleaner: keep
        "//runtime/executor:executor_settings_base",
        "//runtime/util:test_utils",
    ],
)
