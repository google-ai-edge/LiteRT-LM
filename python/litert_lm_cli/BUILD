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
load("//:version.bzl", "VERSION")

load("@custom_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("//:version.bzl", "VERSION")

version = select({
    "//python:dev_build": VERSION + ".dev$(DEV_VERSION)",
    "//conditions:default": VERSION,
})

genrule(
    name = "version_py",
    outs = ["version.py"],
    cmd = "echo 'VERSION = \"'" + version + "'\"' > $@",
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//python/litert_lm",
        requirement("click"),
        requirement("prompt_toolkit"),
    ] + select({
        "//conditions:default": [],
    }),
    imports = [".."],
)

pytype_strict_library(
    name = "help_formatter",
    srcs = ["help_formatter.py"],
    deps = [requirement("click")],
    imports = [".."],
)

py_test(
    name = "help_formatter_test",
    srcs = ["help_formatter_test.py"],
    tags = ["nowindows"],
    deps = [
        ":help_formatter",
        "@absl_py//absl/testing:absltest",
        requirement("click"),
    ],
    imports = [".."],
)

pytype_strict_library(
    name = "venv_manager",
    srcs = ["venv_manager.py"],
    deps = [requirement("click")],
    imports = [".."],
)

pytype_strict_library(
    name = "serve",
    srcs = ["serve.py"],
    deps = [
        ":model",
        "//python/litert_lm",
        requirement("click"),
    ],
    imports = [".."],
)

py_test(
    name = "venv_manager_test",
    srcs = ["venv_manager_test.py"],
    tags = ["nowindows"],
    deps = [
        ":venv_manager",
        "@absl_py//absl/testing:absltest",
    ],
    imports = [".."],
)

pytype_strict_library(
    name = "version",
    srcs = [":version_py"],
)

pytype_strict_library(
    name = "main_lib",
    srcs = ["main.py"],
    deps = [
        ":help_formatter",
        ":model",
        ":serve",
        ":venv_manager",
        ":version",
        "//python/litert_lm",
        requirement("click"),
        requirement("huggingface_hub"),
    ],
    imports = [".."],
)

py_test(
    name = "main_test",
    srcs = ["main_test.py"],
    tags = [
        "nowindows",
    ],
    deps = [
        ":main_lib",
        ":model",
        "@absl_py//absl/testing:absltest",
        requirement("click"),
        requirement("prompt_toolkit"),
    ],
    imports = [".."],
)

py_test(
    name = "serve_test",
    srcs = ["serve_test.py"],
    tags = [
        "nowindows",
    ],
    deps = [
        ":serve",
        "//python/litert_lm",
        "//python/litert_lm:interfaces",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
    imports = [".."],
)

pytype_strict_binary(
    name = "litert-lm",
    srcs = [
        "main.py",
    ],
    main = "main.py",
    tags = [
        "noios",
        "nomacos",
        "nowindows",
    ],
    deps = [
        ":main_lib",
    ],
    imports = [".."],
)

py_package(
    name = "package",
    packages = ["python.litert_lm_cli"],
    deps = [":litert-lm"],
)

py_wheel(
    name = "wheel",
    abi = "none",
    distribution = select({
        "//python:dev_build": "litert-lm-nightly",
        "//conditions:default": "litert-lm",
    }),
    platform = "any",
    python_tag = "py3",
    summary = "LiteRT-LM command line tool",
    description_file = "README.md",
    license = "Apache-2.0",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    strip_path_prefixes = ["python"],
    version = version,
    entry_points = {
        "console_scripts": [
            "litert-lm = litert_lm_cli.main:main",
        ],
    },
    requires = select({
        "//python:dev_build": ["litert-lm-api-nightly"],
        "//conditions:default": ["litert-lm-api"],
    }) + [
        "click",
        "huggingface_hub",
        "prompt_toolkit",
    ],
    deps = [":package"],
)
