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
    name = "prompt_source",
    srcs = ["prompt_source.cc"],
    hdrs = ["prompt_source.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "image_decoder",
    hdrs = ["image_decoder.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_test(
    name = "prompt_source_test",
    srcs = ["prompt_source_test.cc"],
    deps = [
        ":image_decoder",
        ":prompt_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//omni/base:io_types",
        "//support/util:test_utils",
    ],
)
