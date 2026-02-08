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
    name = "litertlm_jni",
    srcs = ["litertlm.cc"],
    linkshared = 1,
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nlohmann_json//:json",
        "@litert//litert/c/internal:litert_logging",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_impl_selected",  # buildcleaner: keep
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/engine:mezo",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "@litert//tflite:minimal_logging",
    ] + select({
        "@platforms//os:android": [],
        "//conditions:default": ["@rules_java//toolchains:jni"],
    }),
)
