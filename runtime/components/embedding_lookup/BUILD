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

package(
    default_hdrs_check = "strict",
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "embedding_lookup",
    hdrs = ["embedding_lookup.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "embedding_lookup_text",
    srcs = ["embedding_lookup_text.cc"],
    hdrs = ["embedding_lookup_text.h"],
    deps = [
        ":embedding_lookup",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "embedding_lookup_multi_modal",
    srcs = ["embedding_lookup_multi_modal.cc"],
    hdrs = ["embedding_lookup_multi_modal.h"],
    deps = [
        ":embedding_lookup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@litert//tflite:framework_stable",
        "@litert//tflite/c:c_api_types",
        "@litert//tflite/c:common",
        "@litert//tflite/core:framework",
        "@litert//tflite/kernels:builtin_ops",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "embedding_lookup_end_of_multi_modal",
    srcs = ["embedding_lookup_end_of_multi_modal.cc"],
    hdrs = ["embedding_lookup_end_of_multi_modal.h"],
    deps = [
        ":embedding_lookup",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
        "@litert//tflite:framework_stable",
        "@litert//tflite:model_builder",
        "@litert//tflite/c:c_api_types",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "embedding_lookup_manager",
    srcs = ["embedding_lookup_manager.cc"],
    hdrs = ["embedding_lookup_manager.h"],
    deps = [
        ":embedding_lookup_end_of_multi_modal",
        ":embedding_lookup_multi_modal",
        ":embedding_lookup_text",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/util:litert_status_util",
        "@litert//tflite:framework_stable",
        "@litert//tflite/c:c_api_types",
        "@litert//tflite/c:common",
        "@litert//tflite/core:framework",
        "@litert//tflite/kernels:builtin_ops",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)
