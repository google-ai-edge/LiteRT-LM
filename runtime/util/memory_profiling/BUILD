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
#   host_memory                 the host side: HostMemoryUsage and the per-OS
#                               contract that reads it
#   accelerator_memory_monitor  the accelerator side: types, the per-OS
#                               platform contract, formatting, and the
#                               accelerator-only AcceleratorMemoryMonitor
#   memory_snapshot             MemorySnapshot / TakeMemorySnapshot(): host and
#                               accelerator memory at one instant
#   memory_monitor              MemoryMonitor: peaks of both over time, sampled
#                               from a background thread
#
# The host and accelerator sides follow the same layout: a platform-neutral
# header declares a platform contract, and each operating system contributes
# one <side>_<os>.cc implementing it (plus an internal <side>_<os>.h when its
# test needs the building blocks) and one <side>_<os>_test.cc that only runs
# on that OS. <side>_unsupported.cc covers every other OS. To add an OS, add
# its files and one entry to each select() below.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]
# [Google-internal load of `objc_library`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "host_memory",
    srcs = select({
        "@platforms//os:android": ["host_memory_linux.cc"],
        "@platforms//os:linux": ["host_memory_linux.cc"],
        "@platforms//os:windows": ["host_memory_windows.cc"],
        "@platforms//os:macos": ["host_memory_apple.cc"],
        "@platforms//os:ios": ["host_memory_apple.cc"],
        "//conditions:default": ["host_memory_unsupported.cc"],
    }),
    hdrs = ["host_memory.h"] + select({
        "@platforms//os:android": ["host_memory_linux.h"],
        "@platforms//os:linux": ["host_memory_linux.h"],
        "//conditions:default": [],
    }),
    deps = select({
        "@platforms//os:android": ["@com_google_absl//absl/strings"],
        "@platforms//os:linux": ["@com_google_absl//absl/strings"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "host_memory_linux_test",
    srcs = ["host_memory_linux_test.cc"],
    # Linux and Android share host_memory_linux.cc; TAP runs this on Linux.
    # The tags keep the other TAP projects from requesting it explicitly,
    # which target_compatible_with alone turns into a build error
    # (b/202306027).
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
        ":host_memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "host_memory_test",
    srcs = ["host_memory_test.cc"],
    # Live test for the single-system-call implementations (Windows, Apple).
    # Runs in TAP's Windows and macOS projects; the tags keep the other TAP
    # projects and Tricorder's builder (default platform) from requesting it
    # explicitly, which target_compatible_with alone turns into a build error
    # (b/202306027).
    tags = [
        "noandroid",
        "nobuilder",
        "nolinux",
    ],
    target_compatible_with = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":host_memory",
        "@com_google_googletest//:gtest_main",
    ],
)

# Reader for /proc/self/fdinfo, shared by the Linux and Android accelerator
# implementations (Android reads dma-buf buffers from it, Linux DRM clients).
cc_library(
    name = "fdinfo",
    srcs = ["fdinfo.cc"],
    hdrs = ["fdinfo.h"],
    # Linux and Android only (/proc). The tags keep the other TAP projects from
    # requesting it explicitly, which target_compatible_with alone turns into a
    # build error (b/202306027).
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fake_proc_self",
    testonly = True,
    hdrs = ["fake_proc_self.h"],
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
        "@com_google_googletest//:gtest_for_library",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "fdinfo_test",
    srcs = ["fdinfo_test.cc"],
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
        ":fake_proc_self",
        ":fdinfo",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "proc_mappings",
    srcs = ["proc_mappings.cc"],
    hdrs = ["proc_mappings.h"],
    # Linux and Android only (/proc); see :fdinfo for the tags.
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
    deps = ["@com_google_absl//absl/strings"],
)

cc_test(
    name = "proc_mappings_test",
    srcs = ["proc_mappings_test.cc"],
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
        ":fake_proc_self",
        ":proc_mappings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "accelerator_memory_android_sources_test",
    srcs = ["accelerator_memory_android_sources_test.cc"],
    # The Android combination rule on canned inputs; runs on Linux in TAP.
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
        ":proc_mappings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "accelerator_memory_monitor",
    srcs = ["accelerator_memory_monitor.cc"] + select({
        # The dma-buf reader and the Android combination rule are Android's
        # implementation; both are built on Linux too so that
        # accelerator_memory_dmabuf_test and
        # accelerator_memory_android_sources_test can run in CI.
        "@platforms//os:android": [
            "accelerator_memory_android.cc",
            "accelerator_memory_android_sources.cc",
            "accelerator_memory_dmabuf.cc",
        ],
        "@platforms//os:linux": [
            "accelerator_memory_android_sources.cc",
            "accelerator_memory_dmabuf.cc",
            "accelerator_memory_linux.cc",
            "accelerator_memory_nvml.cc",
        ],
        "@platforms//os:windows": ["accelerator_memory_windows.cc"],
        # macOS and iOS: accelerator_memory_apple.mm, Objective-C++, lives in
        # the objc_library below and comes in through deps.
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["accelerator_memory_unsupported.cc"],
    }),
    hdrs = ["accelerator_memory_monitor.h"] + select({
        "@platforms//os:android": [
            "accelerator_memory_android.h",
            "accelerator_memory_dmabuf.h",
        ],
        "@platforms//os:linux": [
            "accelerator_memory_android.h",
            "accelerator_memory_dmabuf.h",
            "accelerator_memory_linux.h",
            "accelerator_memory_nvml.h",
        ],
        "@platforms//os:windows": ["accelerator_memory_windows.h"],
        "//conditions:default": [],
    }),
    linkopts = select({
        # dxgi: IDXGIAdapter3::QueryVideoMemoryInfo; gdi32: D3DKMTQueryStatistics.
        "@platforms//os:windows": [
            "dxgi.lib",
            "gdi32.lib",
        ],
        # dlopen() of libnvidia-ml.so at run time.
        "@platforms//os:linux": ["-ldl"],
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
            ":fdinfo",
            ":proc_mappings",
            "@com_google_absl//absl/container:flat_hash_map",
        ],
        "@platforms//os:linux": [
            ":fdinfo",
            ":proc_mappings",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:flat_hash_set",
        ],
        "@platforms//os:macos": [":accelerator_memory_apple"],
        "@platforms//os:ios": [":accelerator_memory_apple"],
        "//conditions:default": [],
    }),
)

# The macOS / iOS implementation of the contract, Objective-C++ because it
# reads MTLDevice. Part of :accelerator_memory_monitor (which selects it in on
# Apple platforms), not a library in its own right; its header is public only
# for its test. It takes accelerator_memory_monitor.h as a source, since the
# contract it implements is declared there and the dependency runs the other
# way.
objc_library(
    name = "accelerator_memory_apple",
    srcs = [
        "accelerator_memory_apple.mm",
        "accelerator_memory_monitor.h",
    ],
    hdrs = ["accelerator_memory_apple.h"],
    copts = ["-ObjC++"],
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",

    ],
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
    # The Android reader, tested on canned fdinfo; runs on Linux in TAP.
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
        ":fake_proc_self",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "accelerator_memory_android_test",
    srcs = ["accelerator_memory_android_test.cc"],
    # Runs on an Android device only:
    #   bazel test --config=android_arm64 :accelerator_memory_android_test
    # No TAP project can run it (TAP's Android project builds cc_test targets
    # for the Linux host), and Tricorder's builder uses the default platform.
    # The tags make them skip it; target_compatible_with alone is not enough,
    # as an explicitly requested incompatible target is a build error
    # (b/202306027).
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
        ":proc_mappings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_test(
    name = "accelerator_memory_linux_test",
    srcs = ["accelerator_memory_linux_test.cc"],
    # Canned DRM fdinfo, plus a live check that needs no GPU; runs on Linux in
    # TAP. The tags keep the other TAP projects from requesting it explicitly
    # (b/202306027).
    tags = [
        "noandroid",
        "noios",
        "nomacos",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        ":fake_proc_self",
        ":fdinfo",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "accelerator_memory_nvml_test",
    srcs = ["accelerator_memory_nvml_test.cc"],
    # The GPU-free part of the NVML reader (merging the per-process lists);
    # runs on Linux in TAP. Tags as for accelerator_memory_linux_test.
    tags = [
        "noandroid",
        "noios",
        "nomacos",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "accelerator_memory_nvidia_test",
    srcs = ["accelerator_memory_nvidia_test.cc"],
    # Needs an NVIDIA GPU: allocates framebuffer memory through the CUDA
    # driver (dlopen()ed, no build dependency) and checks that NVML attributes
    # it to this process. requires-gpu-sm90-full picks a whole H100, i.e. no
    # MIG partitioning, which NVML accounts for differently. The no<os> tags
    # keep the other TAP projects from requesting it explicitly (b/202306027).
    # TSan and MSan abort while loading the NVIDIA driver's prebuilt NVML
    # library: TSan reports an incompatible memory layout, and MSan reports
    # uninitialized reads inside NVML, whose writes it cannot see
    # (b/358526563).
    tags = [
        "noandroid",
        "noios",
        "nomacos",
        "nomsan",
        "notsan",
        "nowindows",
        "requires-gpu-sm90-full",
    ],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "accelerator_memory_windows_test",
    srcs = ["accelerator_memory_windows_test.cc"],
    # The test allocates GPU memory through Direct3D 11.
    linkopts = ["d3d11.lib"],
    # Runs in TAP's Windows project. The tags make the other TAP projects and
    # Tricorder's builder skip it; target_compatible_with alone is not enough,
    # as an explicitly requested incompatible target is a build error
    # (b/202306027).
    tags = [
        "noandroid",
        "nobuilder",
        "noios",
        "nolinux",
        "nomacos",
    ],
    target_compatible_with = select({
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_monitor",
        ":host_memory",
        ":memory_snapshot",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# The Apple test: the merge rule on fixed inputs, and live checks with a Metal
# buffer and with weights mapped from a file. Objective-C++, so an
# objc_library wrapped in a cc_test. Runs in TAP's macOS project
# (Forge-on-Mac has a Metal device; the live checks skip without one); the
# tags keep the other TAP projects and Tricorder's builder from requesting it
# explicitly (b/202306027).
# TODO: Remove "manual" and "notap" once the flakiness (seen on GitHub Actions
# macOS runners) is understood and fixed.
objc_library(
    name = "accelerator_memory_apple_test_lib",
    testonly = True,
    srcs = ["accelerator_memory_apple_test.mm"],
    copts = ["-ObjC++"],
    tags = [
        "manual",
        "noandroid",
        "nobuilder",
        "nolinux",
        "notap",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_apple",
        ":accelerator_memory_monitor",
        ":host_memory",
        ":memory_snapshot",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",

    ],
    alwayslink = True,
)

cc_test(
    name = "accelerator_memory_apple_test",
    tags = [
        "manual",
        "noandroid",
        "nobuilder",
        "nolinux",
        "notap",
        "nowindows",
    ],
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":accelerator_memory_apple_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

# Measurement of how Metal allocations show up in the host figures on macOS;
# the data the Apple implementation of the accelerator contract is based on.
# Objective-C++, so it is an objc_library wrapped in a cc_test, like the other
# Metal tests in the tree. Both targets are tagged manual: the test is meant to
# be run by hand (see the file comment) and read, not to gate anything yet.
objc_library(
    name = "accelerator_memory_apple_metal_test_lib",
    testonly = True,
    srcs = ["accelerator_memory_apple_metal_test.mm"],
    copts = ["-ObjC++"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":host_memory",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",

    ],
    alwayslink = True,
)

cc_test(
    name = "accelerator_memory_apple_metal_test",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":accelerator_memory_apple_metal_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_snapshot",
    srcs = ["memory_snapshot.cc"],
    hdrs = ["memory_snapshot.h"],
    deps = [
        ":accelerator_memory_monitor",
        ":host_memory",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "memory_snapshot_test",
    srcs = ["memory_snapshot_test.cc"],
    deps = [
        ":accelerator_memory_monitor",
        ":host_memory",
        ":memory_snapshot",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "memory_monitor",
    srcs = ["memory_monitor.cc"],
    hdrs = ["memory_monitor.h"],
    deps = [
        ":accelerator_memory_monitor",
        ":memory_snapshot",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "memory_monitor_test",
    srcs = ["memory_monitor_test.cc"],
    deps = [
        ":accelerator_memory_monitor",
        ":memory_monitor",
        ":memory_snapshot",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
