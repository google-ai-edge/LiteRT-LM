# Copyright 2025 Google LLC.
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

package(
    default_visibility = ["//kotlin:__subpackages__"],
)

cc_binary(
    name = "litertlm_jni_cpp",
    srcs = ["litertlm.cc"],
    linkshared = 1,
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "@litert//litert/c/internal:litert_logging",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/components:model_resources",
        "//runtime/components:prompt_template",
        "//runtime/components/logits_processor:no_repeat_ngram_config",
        "//runtime/components/logits_processor:repetition_penalty_config",
        "//runtime/components/logits_processor:suppress_tokens_config",
        "//runtime/components/logits_processor/constrained_decoding:llg_constraint_config",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/conversation:thinking_config",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:file_util",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:logging",
        "//runtime/util:scoped_file",
        "//schema/capabilities:capabilities_c",
        "//schema/core:litertlm_header_schema",
        "@litert//tflite:minimal_logging",
    ] + select({
        "@platforms//os:android": [],
        "//conditions:default": [
            "@rules_java//toolchains:jni"
        ],
    }) + [
        "//runtime/core:engine_impl",  # buildcleaner: keep
    ],
)
