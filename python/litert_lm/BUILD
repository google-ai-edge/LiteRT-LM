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

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "interfaces",
    srcs = ["interfaces.py"],
    deps = [
        ":_ffi",
        ":_messages",
    ],
)

pytype_strict_library(
    name = "_messages",
    srcs = ["_messages.py"],
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
        ":_messages",
        ":interfaces",
    ],
)

pytype_strict_library(
    name = "engine",
    srcs = ["engine.py"],
    deps = [
        ":_ffi",
        ":_messages",
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

config_setting(
    name = "use_prebuilt_c_api",
    define_values = {
        "use_prebuilt_c_api": "true",
    },
)

pytype_strict_library(
    name = "litert_lm",
    srcs = ["__init__.py"],
    data = select({
        ":use_prebuilt_c_api": [":prebuilt_litert_lm_lib"],
        "//conditions:default": [":default_libs"],
    }),
    deps = [
        ":_ffi",
        ":_messages",
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

py_test(
    name = "messages_test",
    srcs = ["messages_test.py"],
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
load("//python/litert_lm:cc_runfiles.bzl", "cc_runfiles")

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
                $(location //build_config:patchelf_exe) --set-rpath '$$ORIGIN' $${OUTS_ARRAY[$$i]}
            done
        else
            for out in $${OUTS_ARRAY[@]}; do
                touch $$out
            done
        fi
    """,
    tags = ["manual"],
    tools = ["//build_config:patchelf_exe"],
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

copy_file(
    name = "copy_dxcompiler_dll",
    src = "@directx_shader_compiler//:dxcompiler",
    out = "dxcompiler.dll",
)

copy_file(
    name = "copy_dxil_dll",
    src = "@directx_shader_compiler//:dxil",
    out = "dxil.dll",
)

copy_file(
    name = "copy_openvino_dispatch_dll",
    src = "@litert//litert/vendors/intel_openvino/dispatch:LiteRtDispatch",
    out = "vendors/intel_openvino/dispatch/LiteRtDispatch.dll",
)

filegroup(
    name = "prebuilt_litert_lm_so_sources",
    srcs = select({
        ":wheel_platform_linux_x86_64": ["//prebuilt/linux_x86_64:liblitert-lm.so"],
        ":wheel_platform_linux_arm64": ["//prebuilt/linux_arm64:liblitert-lm.so"],
        ":wheel_platform_android_arm64": ["//prebuilt/android_arm64:liblitert-lm.so"],
        ":wheel_platform_android_x86_64": ["//prebuilt/android_x86_64:liblitert-lm.so"],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "prebuilt_litert_lm_dylib_sources",
    srcs = select({
        ":wheel_platform_macos_arm64": ["//prebuilt/macos_arm64:liblitert-lm.dylib"],
        "//conditions:default": [],
    }),
)

genrule(
    name = "copy_litert_lm_so",
    srcs = select({
        ":use_prebuilt_c_api": [":prebuilt_litert_lm_so_sources"],
        "//conditions:default": ["//c:litert-lm"],
    }),
    outs = ["liblitert-lm.so"],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            cp $${SRCS_ARRAY[0]} $${OUTS_ARRAY[0]}
        else
            touch $${OUTS_ARRAY[0]}
        fi
    """,
)

genrule(
    name = "copy_litert_lm_dylib",
    srcs = select({
        ":use_prebuilt_c_api": [":prebuilt_litert_lm_dylib_sources"],
        "//conditions:default": ["//c:litert-lm"],
    }),
    outs = ["liblitert-lm.dylib"],
    cmd = """
        SRCS_ARRAY=($(SRCS))
        OUTS_ARRAY=($(OUTS))
        if [ $${#SRCS_ARRAY[@]} -gt 0 ]; then
            cp $${SRCS_ARRAY[0]} $${OUTS_ARRAY[0]}
        else
            touch $${OUTS_ARRAY[0]}
        fi
    """,
)

copy_file(
    name = "copy_prebuilt_litert_lm_dll",
    src = "//prebuilt/windows_x86_64:litert-lm.dll",
    out = "litert-lm.dll",
    tags = ["manual"],
)

filegroup(
    name = "prebuilt_litert_lm_lib",
    srcs = select({
        # Use "wheel_platform_*" as condition to allow cross-compilation since
        # no native binaries is built.
        ":wheel_platform_macos_arm64": [":copy_litert_lm_dylib"],
        ":wheel_platform_windows_x86_64": [
            ":copy_prebuilt_litert_lm_dll",
            ":copy_dxcompiler_dll",
            ":copy_dxil_dll",
            ":copy_openvino_dispatch_dll",
        ],
        ":wheel_platform_linux_x86_64": [":copy_litert_lm_so"],
        ":wheel_platform_linux_arm64": [":copy_litert_lm_so"],
        ":wheel_platform_android_arm64": [":copy_litert_lm_so"],
        ":wheel_platform_android_x86_64": [":copy_litert_lm_so"],
        # default is empty since the prebuilt_litert_lm_lib might not exist.
        "//conditions:default": [],
    }),
)

# On Windows, copy the .dll from runfiles to avoid the Bazel duplicated
# labels.
cc_runfiles(
    name = "litert_lm_windows_runfiles",
    cc_binary = "//c:litert-lm",
)

filegroup(
    name = "default_libs",
    srcs = select({
        "@platforms//os:windows": [":litert_lm_windows_runfiles"],
        "@platforms//os:macos": [":copy_litert_lm_dylib"],
        "//conditions:default": [":copy_litert_lm_so"],
    }) + select({
        "@platforms//os:macos": [":copy_prebuilt_libs_dylib"],
        "@platforms//os:windows": [
            ":copy_dxcompiler_dll",
            ":copy_dxil_dll",
            ":copy_openvino_dispatch_dll",
        ],
        "@platforms//os:linux": [":copy_prebuilt_libs_so"],
        "//conditions:default":  [],
    }),
)

# Use "wheel_platform" to allow cross-compliling the wheel package.
config_setting(
    name = "wheel_platform_linux_x86_64",
    define_values = {"wheel_platform": "linux_x86_64"},
)

config_setting(
    name = "wheel_platform_linux_arm64",
    define_values = {"wheel_platform": "linux_arm64"},
)

config_setting(
    name = "wheel_platform_windows_x86_64",
    define_values = {"wheel_platform": "windows_x86_64"},
)

config_setting(
    name = "wheel_platform_macos_arm64",
    define_values = {"wheel_platform": "macos_arm64"},
)

config_setting(
    name = "wheel_platform_android_arm64",
    define_values = {"wheel_platform": "android_arm64"},
)

config_setting(
    name = "wheel_platform_android_x86_64",
    define_values = {"wheel_platform": "android_x86_64"},
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
        ":wheel_platform_linux_x86_64": "manylinux_2_27_x86_64",
        ":wheel_platform_linux_arm64": "manylinux_2_27_aarch64",
        ":wheel_platform_macos_arm64": "macosx_12_0_arm64",
        ":wheel_platform_windows_x86_64": "win_amd64",
        ":wheel_platform_android_arm64": "android_23_arm64_v8a",
        ":wheel_platform_android_x86_64": "android_23_x86_64",
        "//conditions:default": "any",
    }),
    python_tag = "py3",
    python_requires = ">=3.10",
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
