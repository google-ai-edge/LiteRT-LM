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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "kokoro_model_config",
    hdrs = ["kokoro_model_config.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "phonemizer",
    srcs = ["phonemizer.cc"],
    hdrs = ["phonemizer.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@espeak_ng//:espeak_ng_base_headers",
        "@espeak_ng//:espeak_ng_speak_lib_header",
        "@espeak_ng//:libespeak_ng_la_sources",  # buildcleaner: keep
    ],
)

cc_test(
    name = "phonemizer_test",
    srcs = ["phonemizer_test.cc"],
    data = [
        "@espeak_ng//:espeak_ng_data",
    ],
    deps = [
        ":common",
        ":phonemizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//support/util:test_utils",
    ],
)
