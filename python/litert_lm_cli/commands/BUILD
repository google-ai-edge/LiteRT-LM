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
        "gemini_handler.py",
        "openai_handler.py",
        "serve.py",
        "serve_util.py",
    ],
    imports = [".."],
    deps = [
        "//python/litert_lm",
        "//python/litert_lm_builder:litertlm_peek",
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "convert",
    srcs = ["convert.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        "//python/litert_lm_cli:venv_manager",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "list",
    srcs = ["list.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "import",
    srcs = ["import.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm_cli:common",
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:huggingface_download",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "delete",
    srcs = ["delete.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "rename",
    srcs = ["rename.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ],
)

pytype_strict_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    imports = [".."],
    deps = [
        "//python/litert_lm",
        "//python/litert_lm_cli:common",
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:huggingface_download",
        "//python/litert_lm_cli:model",
        requirement("click"),
    ] + select({
        "//conditions:default": [],
    }),
)

pytype_strict_library(
    name = "run",
    srcs = ["run.py"],
    imports = [".."],
    deps = [
        ":convert",
        "//python/litert_lm",
        "//python/litert_lm_cli:common",
        "//python/litert_lm_cli:help_formatter",
        "//python/litert_lm_cli:huggingface_download",
        "//python/litert_lm_cli:model",
        requirement("click"),
        requirement("prompt_toolkit"),
    ] + select({
        "//conditions:default": [],
    }),
)

py_test(
    name = "serve_openai_streaming_test",
    srcs = ["serve_openai_streaming_test.py"],
    data = ["//runtime/e2e_tests/data:gemma3-1b-it-int4-litertlm"],
    tags = [
        "manual",
        "nomacos",
        "nowindows",
    ],
    deps = [
        ":serve",
        "//python/litert_lm_cli:model",
        "@absl_py//absl/testing:absltest",
    ],
)
