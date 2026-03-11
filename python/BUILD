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

load(
    "@org_tensorflow//tensorflow:pytype.default.bzl",
    "pytype_strict_library",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
)

pytype_strict_library(
    name = "tools",
    srcs = ["tools.py"],
)

pybind_extension(
    name = "litert_lm_ext",
    srcs = ["litert_lm.cc"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nanobind",
        "@nanobind_json",
        "@litert//litert/c/internal:litert_logging",
        "//runtime/conversation",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_impl_selected",  # buildcleaner: keep
        "//runtime/engine:engine_interface",
        "@litert//tflite:minimal_logging",
        "@litert//tflite/core/c:private_c_api_types",
    ],
)
