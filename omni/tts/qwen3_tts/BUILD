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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "qwen3_stage_options",
    hdrs = ["qwen3_stage_options.h"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "qwen3_frontend_stage",
    srcs = ["qwen3_frontend_stage.cc"],
    hdrs = ["qwen3_frontend_stage.h"],
    deps = [
        ":common",
        ":qwen3_stage_options",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/tts:text_frontend",
        "//support/tokenizer:huggingface_tokenizer",
    ],
)

cc_library(
    name = "qwen3_acoustic_predictor_stage",
    srcs = ["qwen3_acoustic_predictor_stage.cc"],
    hdrs = ["qwen3_acoustic_predictor_stage.h"],
    deps = [
        ":common",
        ":qwen3_stage_options",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//omni/tts:acoustic_predictor",
        "//omni/tts:text_frontend",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//support/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "qwen3_latent_decoder_stage",
    srcs = ["qwen3_latent_decoder_stage.cc"],
    hdrs = ["qwen3_latent_decoder_stage.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/tts:acoustic_predictor",
        "//omni/tts:latent_decoder",
    ],
)

cc_library(
    name = "qwen3_vocoder_stage",
    srcs = ["qwen3_vocoder_stage.cc"],
    hdrs = ["qwen3_vocoder_stage.h"],
    deps = [
        ":common",
        ":qwen3_stage_options",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//omni/tts:latent_decoder",
        "//omni/tts:vocoder",
        "//support/util:convert_tensor_buffer",
    ],
)
