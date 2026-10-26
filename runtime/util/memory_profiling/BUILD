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

# Measurement of the memory a LiteRT-LM process uses, split into host (CPU)
# and accelerator (GPU / NPU) memory, for the CLI's benchmark metrics.
#
# accelerator_memory_monitor.* is platform-neutral. Each operating system
# contributes one accelerator_memory_<os>.cc implementing the platform
# contract declared in accelerator_memory_monitor.h and one
# accelerator_memory_<os>_test.cc that only runs on that OS. Building blocks an
# implementation is made of go in their own files when they can be tested
# without the OS (accelerator_memory_dmabuf.* is Android's, built on Linux too
# so that its test runs in CI). To add an OS, add its files and one entry to
# each select() below.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "accelerator_memory_monitor",
    srcs = ["accelerator_memory_monitor.cc"] + select({
        "@platforms//os:android": [
            "accelerator_memory_android.cc",
            "accelerator_memory_dmabuf.cc",
        ],
        # The dma-buf reader is Android's implementation; it is built on Linux
        # too so that accelerator_memory_dmabuf_test can run in CI.
        "@platforms//os:linux": [
            "accelerator_memory_dmabuf.cc",
            "accelerator_memory_unsupported.cc",
        ],
        "//conditions:default": ["accelerator_memory_unsupported.cc"],
    }),
    hdrs = ["accelerator_memory_monitor.h"] + select({
        "@platforms//os:android": ["accelerator_memory_dmabuf.h"],
        "@platforms//os:linux": ["accelerator_memory_dmabuf.h"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ] + select({
        "@platforms//os:android": [
            "@com_google_absl//absl/container:flat_hash_map",
        ],
        "@platforms//os:linux": [
            "@com_google_absl//absl/container:flat_hash_map",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "accelerator_memory_monitor_test",
    srcs = ["accelerator_memory_monitor_test.cc"],
    deps = [
        ":accelerator_memory_monitor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "accelerator_memory_dmabuf_test",
    srcs = ["accelerator_memory_dmabuf_test.cc"],
    # The Android reader, tested on canned fdinfo; runs on Linux in TAP. The
    # tags keep the other TAP projects from requesting it explicitly, which
    # target_compatible_with alone turns into a build error (b/202306027).
    tags = [
        "noios",
        "nomacos",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "accelerator_memory_android_test",
    srcs = ["accelerator_memory_android_test.cc"],
    # Runs on an Android device only:
    #   bazel test --config=android_arm64 :accelerator_memory_android_test
    # No TAP project can run it: TAP's Android project builds cc_test targets
    # for the Linux host (--android_platforms only applies to android_* rules),
    # and Tricorder's builder uses the default platform. The tags make them
    # skip it; target_compatible_with alone is not enough, as an explicitly
    # requested incompatible target is a build error (b/202306027).
    tags = [
        "noandroid",
        "nobuilder",
        "noios",
        "nolinux",
        "nomacos",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        "@com_google_googletest//:gtest_main",
    ],
)
