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

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
)

pytype_strict_library(
    name = "tools",
    srcs = ["tools.py"],
)

cc_binary(
    name = "litert_lm_ext.so",
    srcs = ["litert_lm.cc"],
    copts = [
        "-fexceptions",
        "-fvisibility=hidden",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-undefined,dynamic_lookup"],
        "//conditions:default": [
            "-Wl,-Bsymbolic",
            "-Wl,-rpath,'$$ORIGIN'",
            # Marks the binary stack as non-executable. Required for
            # compatibility with modern Linux kernels (NX/SELlinux) and to
            # ensure the extension can be imported in restricted environments,
            # like pip install (b/496655352).
            "-Wl,-z,noexecstack",
        ],
    }),
    linkshared = True,
    tags = [
        "noandroid",
        "noios",
        "nowindows",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "@nanobind",
        "@nanobind_json",
        "@litert//litert/c/internal:litert_logging",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/core:engine_impl",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "@rules_python//python/cc:current_py_cc_headers",  # buildcleaner: keep
        "@litert//tflite:minimal_logging",
    ],
)

pytype_strict_library(
    name = "litert_lm",
    srcs = [
        "__init__.py",
        "litert_lm_ext.py",
    ],
    data = [":litert_lm_ext.so"] + select({
        "@platforms//os:macos": [":copy_prebuilt_libs_dylib"],
        "@platforms//os:windows": [], # the cc_binary rule already properly provisions this DLL into the package directory on Windows.
        "//conditions:default": [":copy_prebuilt_libs_so"],
    }),
    deps = [
        ":interfaces",
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
    ],
    imports = [".."],
)

py_test(
    name = "tool_test",
    srcs = ["tool_test.py"],
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

config_setting(
    name = "is_nightly",
    define_values = {"PYPI_NIGHTLY": "1"},
)

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
# $ PYTHON_VERSION=3.10
# $ bazel build \
#     --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
#     --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} \
#     -c opt //python/litert_lm:wheel
#
# To build nightly wheel:
# $ PYTHON_VERSION=3.10
# $ DATE=$(date +'%Y%m%d')
# $ bazel build \
#     --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
#     --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} \
#     --define=PYPI_NIGHTLY=1 \
#     --define=PYPI_NIGHTLY_DATE=${DATE} \
#     -c opt //python/litert_lm:wheel
py_wheel(
    name = "wheel",
    abi = select({
        "@rules_python//python/config_settings:is_python_3.10": "cp310",
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "//conditions:default": "none",
    }),
    distribution = select({
        ":is_nightly": "litert-lm-api-nightly",
        "//conditions:default": "litert-lm-api",
    }),
    platform = select({
        "//build_config:linux_x86_64": "manylinux_2_35_x86_64",
        "//build_config:linux_arm64": "manylinux_2_35_aarch64",
        "@platforms//os:macos": "macosx_12_0_arm64",
        "@platforms//os:windows": "win_amd64",
        "//conditions:default": "any",
    }),
    python_tag = select({
        "@rules_python//python/config_settings:is_python_3.10": "cp310",
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "//conditions:default": "py3",
    }),
    summary = "LiteRT-LM inference library",
    license = "Apache-2.0",
    description_file = "README.md",
    homepage = "https://github.com/google-ai-edge/LiteRT-LM",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
    ],
    strip_path_prefixes = ["python"],
    version = select({
        ":is_nightly": VERSION + ".dev$(PYPI_NIGHTLY_DATE)",
        "//conditions:default": VERSION,
    }),
    deps = [":package"],
)
