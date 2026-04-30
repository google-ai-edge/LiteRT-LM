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
# [Google-internal load of `cc_binary`]

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

# Shared library of the LiteRT-LM C API.
cc_binary(
    name = "litert-lm",
    srcs = [
        "//c:engine.cc",
        "//c:engine.h",
    ],
    copts = ["-fvisibility=hidden"],
    features = ["-legacy_whole_archive"],
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
            # Report unresolved symbol references as errors during linking.
            "-Wl,--no-undefined",
            # Strip all symbols to reduce library size (debug info is removed).
            "-Wl,--strip-all",
            # Remove unused sections/functions to further reduce library size.
            "-Wl,--gc-sections",
            # Ensure symbols are resolved internally first.
            "-Wl,-Bsymbolic",
            # Update the rpath
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = 1,
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/core:engine_impl",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:logging",
    ],
)

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
)

pytype_strict_library(
    name = "tools",
    srcs = ["tools.py"],
    deps = [":interfaces"],
)

pytype_strict_library(
    name = "_ffi",
    srcs = ["_ffi.py"],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":_ffi",
        ":interfaces",
    ],
)

pytype_strict_library(
    name = "session",
    srcs = ["session.py"],
    deps = [
        ":_ffi",
        ":interfaces",
    ],
)

pytype_strict_library(
    name = "conversation",
    srcs = ["conversation.py"],
    deps = [
        ":_ffi",
        ":interfaces",
    ],
)

pytype_strict_library(
    name = "engine",
    srcs = ["engine.py"],
    deps = [
        ":_ffi",
        ":conversation",
        ":interfaces",
        ":session",
        ":tools",
        ":utils",
    ],
)

pytype_strict_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    deps = [
        ":_ffi",
        ":interfaces",
    ],
)

pytype_strict_library(
    name = "litert_lm",
    srcs = ["__init__.py"],
    data = ["//python/litert_lm:litert-lm"] + select({
        "@platforms//os:macos": [":copy_prebuilt_libs_dylib"],
        "@platforms//os:windows": [], # the cc_binary rule already properly provisions this DLL into the package directory on Windows.
        "//conditions:default": [":copy_prebuilt_libs_so"],
    }),
    deps = [
        ":_ffi",
        ":benchmark",
        ":conversation",
        ":engine",
        ":interfaces",
        ":session",
        ":tools",
    ],
)

py_test(
    name = "engine_test",
    srcs = ["engine_test.py"],
    data = ["//runtime/testdata"],
    tags = ["nowindows"],
    deps = [
        ":litert_lm",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
    imports = [".."],
)

py_test(
    name = "tool_test",
    srcs = ["tool_test.py"],
    data = ["//runtime/testdata"],
    tags = ["nowindows"],
    deps = [
        ":litert_lm",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
    imports = [".."],
)

load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//:version.bzl", "VERSION")

_PREBUILT_LIBS = [
    "libGemmaModelConstraintProvider",
    "libLiteRtWebGpuAccelerator",
    "libLiteRtTopKWebGpuSampler",
    "libLiteRt",
]

genrule(
    name = "copy_prebuilt_libs_so",
    srcs = select({
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:" + lib + ".so" for lib in _PREBUILT_LIBS],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:" + lib + ".so" for lib in _PREBUILT_LIBS],
        "//conditions:default": [],
    }),
    outs = [lib + ".so" for lib in _PREBUILT_LIBS],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            for i in $${!SRCS_ARRAY[@]}; do
                cp $${SRCS_ARRAY[$$i]} $${OUTS_ARRAY[$$i]}
                # Set the rpath to '$$ORIGIN'
                patchelf --set-rpath '$$ORIGIN' $${OUTS_ARRAY[$$i]}
            done
        else
            for out in $${OUTS_ARRAY[@]}; do
                touch $$out
            done
        fi
    """,
    tags = ["manual"],
)

genrule(
    name = "copy_prebuilt_libs_dylib",
    srcs = select({
        "@platforms//os:macos": ["//prebuilt/macos_arm64:" + lib + ".dylib" for lib in _PREBUILT_LIBS],
        "//conditions:default": [],
    }),
    outs = [lib + ".dylib" for lib in _PREBUILT_LIBS],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            for i in $${!SRCS_ARRAY[@]}; do
                cp $${SRCS_ARRAY[$$i]} $${OUTS_ARRAY[$$i]}
            done
        else
            for out in $${OUTS_ARRAY[@]}; do
                touch $$out
            done
        fi
    """,
    tags = ["manual"],
)

py_package(
    name = "package",
    # Explicitly specify packages to avoid including transitively linked shared
    # libraries (like duplicate .so files in _solib_*/...) in the wheel.
    packages = ["python.litert_lm"],
    deps = [":litert_lm"],
)

# To build normal wheel:
# $ bazel build -c opt //python/litert_lm:wheel
#
# To build nightly wheel:
# $ DATE=$(date +'%Y%m%d')
# $ bazel build \
#     --define=DEV_BUILD=1 \
#     --define=DEV_VERSION=${DATE} \
#     -c opt //python/litert_lm:wheel
py_wheel(
    name = "wheel",
    abi = "none",
    distribution = select({
        "//python:dev_build": "litert-lm-api-nightly",
        "//conditions:default": "litert-lm-api",
    }),
    platform = select({
        "//build_config:linux_x86_64": "manylinux_2_35_x86_64",
        "//build_config:linux_arm64": "manylinux_2_35_aarch64",
        "@platforms//os:macos": "macosx_12_0_arm64",
        "@platforms//os:windows": "win_amd64",
        "//conditions:default": "any",
    }),
    python_tag = "py3",
    summary = "LiteRT-LM inference library",
    license = "Apache-2.0",
    description_file = "README.md",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
    ],
    strip_path_prefixes = ["python"],
    version = select({
        "//python:dev_build": VERSION + ".dev$(DEV_VERSION)",
        "//conditions:default": VERSION,
    }),
    deps = [":package"],
)
