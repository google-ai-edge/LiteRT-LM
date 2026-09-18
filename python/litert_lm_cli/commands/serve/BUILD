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

load("@custom_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")
load(
    "@org_tensorflow//tensorflow:pytype.default.bzl",
    "pytype_strict_library",
    py_test = "pytype_strict_contrib_test",
)

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "serve",
    srcs = [
        "__init__.py",
        "gemini_handler.py",
        "module.py",
        "openai_handler.py",
        "util.py",
    ],
    imports = ["../../.."],
    deps = [
        "//python/litert_lm",
        "//python/litert_lm_builder:litertlm_builder",
        "//python/litert_lm_cli:common",
        "//python/litert_lm_cli:config",
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        requirement("click"),
        requirement("typing_extensions"),
    ],
)

py_test(
    name = "openai_streaming_test",
    srcs = ["openai_streaming_test.py"],
    data = ["//runtime/e2e_tests/data:gemma3-1b-it-int4-litertlm"],
    tags = [
        "manual",
        "nomacos",
        "nowindows",
    ],
    deps = [
        ":serve",
        "//python/litert_lm",
        "//python/litert_lm_cli:config",
        "//python/litert_lm_cli:model",
        "@absl_py//absl/testing:absltest",
    ],
)
