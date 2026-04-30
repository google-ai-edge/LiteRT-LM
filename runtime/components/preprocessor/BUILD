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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "audio_preprocessor",
    hdrs = ["audio_preprocessor.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "audio_preprocessor_test",
    srcs = ["audio_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_preprocessor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "audio_preprocessor_miniaudio",
    srcs = ["audio_preprocessor_miniaudio.cc"],
    hdrs = ["audio_preprocessor_miniaudio.h"],
    deps = [
        ":audio_preprocessor",
        ":mel_filterbank",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@kissfft//:kissfftr",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ] + select({
        "@platforms//os:ios": ["@miniaudio//:miniaudio_objc"],
        "@platforms//os:macos": ["@miniaudio//:miniaudio_objc"],
        "//conditions:default": ["@miniaudio"],
    }),
)

cc_test(
    name = "audio_preprocessor_miniaudio_test",
    srcs = ["audio_preprocessor_miniaudio_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_preprocessor",
        ":audio_preprocessor_miniaudio",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "mel_filterbank",
    srcs = [
        "mel_filterbank.cc",
    ],
    hdrs = [
        "mel_filterbank.h",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mel_filterbank_test",
    srcs = ["mel_filterbank_test.cc"],
    deps = [
        ":mel_filterbank",
        ":signal_vector_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/random",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "signal_vector_util",
    hdrs = ["signal_vector_util.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_test(
    name = "signal_vector_util_test",
    srcs = ["signal_vector_util_test.cc"],
    deps = [
        ":signal_vector_util",
        "@com_google_googletest//:gtest_main",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "image_preprocessor",
    hdrs = ["image_preprocessor.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "image_preprocessor_test",
    srcs = ["image_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":image_preprocessor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "by_pass_audio_preprocessor",
    hdrs = ["by_pass_audio_preprocessor.h"],
    deps = [
        ":audio_preprocessor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "by_pass_audio_preprocessor_test",
    srcs = ["by_pass_audio_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":by_pass_audio_preprocessor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "by_pass_image_preprocessor",
    hdrs = ["by_pass_image_preprocessor.h"],
    deps = [
        ":image_preprocessor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "by_pass_image_preprocessor_test",
    srcs = ["by_pass_image_preprocessor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":by_pass_image_preprocessor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "stb_image_preprocessor",
    srcs = ["stb_image_preprocessor.cc"],
    hdrs = ["stb_image_preprocessor.h"],
    deps = [
        ":image_preprocessor",
        ":image_preprocessor_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@stb//:stblib",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "stb_image_preprocessor_test",
    srcs = ["stb_image_preprocessor_test.cc"],
    data = ["//runtime/components/preprocessor/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":image_preprocessor",
        ":stb_image_preprocessor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "image_preprocessor_utils",
    srcs = ["image_preprocessor_utils.cc"],
    hdrs = ["image_preprocessor_utils.h"],
    deps = [
        ":image_preprocessor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "image_preprocessor_utils_test",
    srcs = ["image_preprocessor_utils_test.cc"],
    deps = [
        ":image_preprocessor",
        ":image_preprocessor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
