# Copyright 2024 The ODML Authors.
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
    default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//third_party/odml/litert_lm:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "litert_compiled_model_executor_utils",
    srcs = ["litert_compiled_model_executor_utils.cc"],
    hdrs = ["litert_compiled_model_executor_utils.h"],
    deps = [
        "//third_party/absl/container:btree",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:external_file_cc_proto",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
        "//third_party/odml/litert_lm/runtime/util:model_asset_bundle_resources",
    ],
)

cc_library(
    name = "llm_litert_compiled_model_executor",
    srcs = ["llm_litert_compiled_model_executor.cc"],
    hdrs = ["llm_litert_compiled_model_executor.h"],
    deps = [
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_config",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/memory",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/runtime/accelerators/gpu:accelerator_options",
        "@litert//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "//third_party/odml/litert_lm/runtime/components:top_p_cpu_sampler",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_compiled_model_executor_test",
    srcs = ["llm_litert_compiled_model_executor_test.cc"],
    data = ["//third_party/odml/litert_lm/runtime/testdata"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":litert_compiled_model_executor_utils",
        ":llm_executor_config",
        ":llm_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_model",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
        "//third_party/odml/litert_lm/runtime/util:model_asset_bundle_resources",
        "//third_party/tensorflow/lite:model_builder",
    ],
)

cc_library(
    name = "fake_llm_executor",
    srcs = ["fake_llm_executor.cc"],
    hdrs = ["fake_llm_executor.h"],
    deps = [
        ":llm_executor",
        ":llm_executor_config",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "fake_llm_executor_test",
    srcs = ["fake_llm_executor_test.cc"],
    deps = [
        ":fake_llm_executor",
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/status",
        "//third_party/absl/types:span",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor",
    srcs = ["llm_executor.cc"],
    hdrs = ["llm_executor.h"],
    deps = [
        ":llm_executor_base",
        "//third_party/odml/litert_lm/runtime/util:logging_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor_base",
    hdrs = ["llm_executor_base.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "llm_executor_test",
    srcs = ["llm_executor_test.cc"],
    deps = [
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/c:litert_tensor_buffer",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor_config",
    srcs = ["llm_executor_config.cc"],
    hdrs = ["llm_executor_config.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/odml/infra/genai/inference/proto:sampler_params_cc_proto",
        "//third_party/odml/infra/genai/inference/utils/llm_utils:constraint_utils",
        "//third_party/odml/litert_lm/runtime/util:logging",
        "//third_party/odml/litert_lm/runtime/util:scoped_file",
    ],
)

cc_test(
    name = "llm_executor_config_test",
    srcs = ["llm_executor_config_test.cc"],
    deps = [
        ":llm_executor_config",
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/status",
    ],
)
