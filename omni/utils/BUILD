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

cc_library(
    name = "audio_resampler",
    srcs = ["audio_resampler.cc"],
    hdrs = ["audio_resampler.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)

cc_test(
    name = "audio_resampler_test",
    srcs = ["audio_resampler_test.cc"],
    deps = [
        ":audio_resampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "audio_utils",
    srcs = ["audio_utils.cc"],
    hdrs = ["audio_utils.h"],
    deps = [
        "//support/preprocessor:audio_preprocessor",
    ],
)

cc_test(
    name = "audio_utils_test",
    srcs = ["audio_utils_test.cc"],
    deps = [
        ":audio_utils",
        "@com_google_googletest//:gtest_main",
        "//support/preprocessor:audio_preprocessor",
    ],
)

cc_library(
    name = "whisper_log_mel_extractor",
    srcs = ["whisper_log_mel_extractor.cc"],
    hdrs = ["whisper_log_mel_extractor.h"],
    deps = [
        ":audio_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//support/preprocessor:audio_preprocessor_miniaudio",
        "//support/util:io_types",
    ],
)

cc_test(
    name = "whisper_log_mel_extractor_test",
    srcs = ["whisper_log_mel_extractor_test.cc"],
    deps = [
        ":whisper_log_mel_extractor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.cc"],
    hdrs = ["tensor_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    deps = [
        ":tensor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "text_utils",
    srcs = ["text_utils.cc"],
    hdrs = ["text_utils.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "text_utils_test",
    srcs = ["text_utils_test.cc"],
    deps = [
        ":text_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)
