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
load("@custom_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "litert_lm_model",
    srcs = ["litert_lm_model.py"],
    imports = ["../../.."],
    deps = [
        "//python/litert_lm",
        requirement("lm_eval"),
    ],
)

py_test(
    name = "litert_lm_model_test",
    srcs = ["litert_lm_model_test.py"],
    imports = ["../../.."],
    tags = [
        "notap",
        "nowindows",
    ],
    deps = [
        ":litert_lm_model",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
