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
    "pytype_strict_binary",
    "pytype_strict_library",
)
load("@custom_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

pytype_strict_library(
    name = "litert_lm_eval_lib",
    srcs = ["litert_lm_eval.py"],
    imports = [".."],
    deps = [
        ":utils",
        "//python/litert_lm_eval/runners/lm_eval_runner:litert_lm_model",
        requirement("lm_eval"),
    ],
)

pytype_strict_binary(
    name = "litert_lm_eval",
    srcs = ["litert_lm_eval.py"],
    deps = [
        ":litert_lm_eval_lib",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    imports = [".."],
)

py_test(
    name = "litert_lm_eval_test",
    srcs = ["litert_lm_eval_test.py"],
    imports = [".."],
    tags = [
        "notap",
        "nowindows",
    ],
    deps = [
        ":litert_lm_eval_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    imports = [".."],
    deps = [
        ":utils",
        "@absl_py//absl/testing:absltest",
    ],
)
