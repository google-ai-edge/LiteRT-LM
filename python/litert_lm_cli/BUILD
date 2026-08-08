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

package(
    default_visibility = ["//python/litert_lm_cli:__subpackages__"],
)

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
    name = "config",
    srcs = ["config.py"],
    data = ["config_schema.json"],
    deps = [
        requirement("click"),
    ],
    imports = [".."],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":config",
        "//python/litert_lm",
        "//python/litert_lm_builder:litertlm_builder",
        "//python/litert_lm_builder:litertlm_peek",
        requirement("click"),
    ],
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
    name = "common",
    srcs = ["common.py"],
    deps = [
        ":config",
        requirement("click"),
    ],
    imports = [".."],
)

pytype_strict_library(
    name = "huggingface_download",
    srcs = ["huggingface_download.py"],
    deps = [
        ":common",
        ":model",
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

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    tags = ["nowindows"],
    deps = [
        ":common",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("click"),
    ],
    imports = [".."],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    tags = ["nowindows"],
    deps = [
        ":config",
        ":model",
        "//python/litert_lm",
        "//python/litert_lm_builder:litertlm_builder",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    tags = ["nowindows"],
    deps = [
        ":common",
        ":config",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("click"),
        requirement("jsonschema"),
    ],
    imports = [".."],
)

py_test(
    name = "huggingface_download_test",
    srcs = ["huggingface_download_test.py"],
    tags = ["nowindows"],
    deps = [
        ":huggingface_download",
        ":model",
        "@absl_py//absl/testing:absltest",
        requirement("click"),
    ],
    imports = [".."],
)

pytype_strict_library(
    name = "cli_helpers",
    srcs = ["cli_helpers.py"],
    deps = [
        ":huggingface_download",
        ":model",
        requirement("click"),
        requirement("prompt_toolkit"),
        requirement("questionary"),
    ],
    imports = [".."],
)

py_test(
    name = "cli_helpers_test",
    srcs = ["cli_helpers_test.py"],
    tags = ["nowindows"],
    deps = [
        ":cli_helpers",
        ":huggingface_download",
        ":model",
        "@absl_py//absl/testing:absltest",
        requirement("click"),
        requirement("prompt_toolkit"),
        requirement("questionary"),
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
        ":common",
        ":help_formatter",
        ":version",
        "//python/litert_lm",
        "//python/litert_lm_cli/commands:benchmark",
        "//python/litert_lm_cli/commands:convert",
        "//python/litert_lm_cli/commands:delete",
        "//python/litert_lm_cli/commands:import",  # fixdeps: keep
        "//python/litert_lm_cli/commands:list",
        "//python/litert_lm_cli/commands:pack",
        "//python/litert_lm_cli/commands:rename",
        "//python/litert_lm_cli/commands:run",
        "//python/litert_lm_cli/commands:serve",
        "//python/litert_lm_cli/commands:unpack",
        requirement("click"),
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
        ":common",
        ":main_lib",
        "//python/litert_lm",
        "//python/litert_lm_cli/commands:run",
        "@absl_py//absl/testing:absltest",
        requirement("click"),
        requirement("prompt_toolkit"),
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
        "nowindows",
    ],
    visibility = [
        "//learning/language/tunelab/gemma:__subpackages__",
        "//:__subpackages__",
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
    python_requires = ">=3.10",
    summary = "LiteRT-LM command line tool",
    description_file = "README.md",
    license = "Apache-2.0",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
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
    requires_file = ":gen_requires",
    deps = [":package"],
)

genrule(
    name = "gen_requires",
    outs = ["requires.txt"],
    cmd = select({
        "//python:dev_build": "echo 'litert-lm-api-nightly==" + VERSION + ".dev$(DEV_VERSION)' > $@ && echo 'litert-lm-builder-nightly==" + VERSION + ".dev$(DEV_VERSION)' >> $@",
        "//conditions:default": "echo 'litert-lm-api==" + VERSION + "' > $@ && echo 'litert-lm-builder==" + VERSION + "' >> $@",
    }) + """
        echo 'click' >> $@
        echo 'prompt_toolkit' >> $@
        echo 'typing-extensions' >> $@
        echo 'questionary' >> $@
    """,
)
