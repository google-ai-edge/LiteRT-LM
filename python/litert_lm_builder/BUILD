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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")
load("@rules_python//python:defs.bzl", "py_test")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")

load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("//:version.bzl", "VERSION")
load("@custom_pip_deps//:requirements.bzl", "requirement")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

flatbuffer_py_library(
    name = "litertlm_header_schema_py",
    srcs = ["//schema/core:litertlm_header_schema.fbs"],
)

pytype_strict_library(
    name = "litertlm_core",
    srcs = ["litertlm_core.py"],
    deps = [
        ":litertlm_header_schema_py",
        "@absl_py//absl:app",
    ] + select({
        "@platforms//os:macos": [],
        "//conditions:default": [
        ],
    }),
)

py_test(
    name = "litertlm_core_test",
    srcs = ["litertlm_core_test.py"],
    imports = [".."],
    deps = [
        ":litertlm_core",
        ":litertlm_header_schema_py",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "litertlm_peek",
    srcs = ["litertlm_peek.py"],
    deps = [
        ":litertlm_core",
        ":litertlm_header_schema_py",
        "//runtime/proto:executor_metadata_py_pb2",
        "//runtime/proto:llm_metadata_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "litertlm_peek_main",
    srcs = ["litertlm_peek_main.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":litertlm_core",
        ":litertlm_peek",
    ],
)

py_test(
    name = "litertlm_peek_test",
    srcs = ["litertlm_peek_test.py"],
    imports = [".."],
    data = ["//schema:testdata"],
    main = "litertlm_peek_test.py",
    deps = [
        ":litertlm_builder",
        ":litertlm_peek",
        "//runtime/proto:executor_metadata_py_pb2",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "litertlm_builder",
    srcs = ["litertlm_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_core",
        ":litertlm_header_schema_py",
        ":litertlm_peek",
        "@flatbuffers//:runtime_py",
        "//runtime/proto:executor_metadata_py_pb2",
        "//runtime/proto:llm_metadata_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        requirement("tomli"),
    ],
)

py_test(
    name = "litertlm_builder_test",
    srcs = ["litertlm_builder_test.py"],
    imports = [".."],
    deps = [
        ":litertlm_builder",
        ":litertlm_core",
        ":litertlm_header_schema_py",
        ":litertlm_peek",
        "//runtime/proto:executor_metadata_py_pb2",
        "//runtime/proto:llm_metadata_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "litertlm_builder_cli",
    srcs = ["litertlm_builder_cli.py"],
    imports = [".."],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_builder",
        ":litertlm_core",
    ],
)

py_test(
    name = "litertlm_builder_cli_test",
    srcs = ["litertlm_builder_cli_test.py"],
    imports = [".."],
    data = [
        ":litertlm_builder_cli",
    ],
    deps = [
        ":litertlm_core",
        ":litertlm_peek",
        "@rules_python//python/runfiles",
        "//runtime/proto:executor_metadata_py_pb2",
        "//runtime/proto:llm_metadata_py_pb2",
        "@absl_py//absl/testing:absltest",
    ],
)

version = select({
    "//python:dev_build": VERSION + ".dev$(DEV_VERSION)",
    "//conditions:default": VERSION,
})

pytype_strict_library(
    name = "litert_lm_builder",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_builder",
        ":litertlm_peek",
    ],
)

# Shading genrules
genrule(
    name = "shaded_sources",
    srcs = [
        "__init__.py",
        "litertlm_core.py",
        "litertlm_peek.py",
        "litertlm_builder.py",
        "litertlm_builder_cli.py",
        "litertlm_peek_main.py",
        ":litertlm_header_schema_py",
        "//runtime/proto:executor_metadata_py_pb2",
        "//runtime/proto:llm_metadata_py_pb2",
        "//runtime/proto:llm_model_type_py_pb2",
        "//runtime/proto:sampler_params_py_pb2",
        "//runtime/proto:token_py_pb2",
    ],
    outs = [
        "shaded/litert_lm_builder/__init__.py",
        "shaded/litert_lm_builder/litertlm_core.py",
        "shaded/litert_lm_builder/litertlm_peek.py",
        "shaded/litert_lm_builder/litertlm_builder.py",
        "shaded/litert_lm_builder/litertlm_builder_cli.py",
        "shaded/litert_lm_builder/litertlm_peek_main.py",
        "shaded/litert_lm_builder/litertlm_header_schema_py_generated.py",
        "shaded/litert_lm_builder/runtime/proto/executor_metadata_pb2.py",
        "shaded/litert_lm_builder/runtime/proto/llm_metadata_pb2.py",
        "shaded/litert_lm_builder/runtime/proto/llm_model_type_pb2.py",
        "shaded/litert_lm_builder/runtime/proto/sampler_params_pb2.py",
        "shaded/litert_lm_builder/runtime/proto/token_pb2.py",
    ],
    cmd = """
        for src in $(SRCS); do
            filename=$$(basename $$src)
            for out in $(OUTS); do
                if [ $$(basename $$out) = $$filename ]; then
                    cp $$src $$out
                    if [ $$filename = "litertlm_peek.py" ] || [ $$filename = "litertlm_builder.py" ]; then
                        sed -i 's/from runtime\\.proto/from litert_lm_builder.runtime.proto/g' $$out
                    elif [[ $$filename == *_pb2.py ]]; then
                        sed -i 's/runtime\\.proto/litert_lm_builder.runtime.proto/g' $$out
                    fi
                    break
                fi
            done
        done
    """,
)

py_package(
    name = "package",
    packages = ["python.litert_lm_builder.shaded.litert_lm_builder"],
    deps = [":shaded_sources"],
)

py_wheel(
    name = "wheel",
    abi = "none",
    distribution = select({
        "//python:dev_build": "litert-lm-builder-nightly",
        "//conditions:default": "litert-lm-builder",
    }),
    platform = "any",
    python_tag = "py3",
    python_requires = ">=3.10",
    summary = "Python tools for building and inspecting LiteRT-LM file formats",
    description_file = "README.md",
    license = "Apache-2.0",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    strip_path_prefixes = ["python/litert_lm_builder/shaded"],
    version = version,
    entry_points = {
        "console_scripts": [
            "litert-lm-builder = litert_lm_builder.litertlm_builder_cli:run",
            "litert-lm-peek = litert_lm_builder.litertlm_peek_main:run",
        ],
    },
    requires_file = ":gen_requires",
    deps = [":package"],
)

genrule(
    name = "gen_requires",
    outs = ["requires.txt"],
    cmd = """
        echo 'protobuf' > $@
        echo 'flatbuffers' >> $@
        echo 'absl-py' >> $@
        echo 'tomli' >> $@
    """,
)
