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
    name = "omni_session",
    hdrs = ["omni_session.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//omni/base:io_types",
        "//omni/base:stage",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "multi_staged_session",
    srcs = ["multi_staged_session.cc"],
    hdrs = ["multi_staged_session.h"],
    deps = [
        ":omni_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//omni/base:async_stage_scheduler",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "multi_staged_session_test",
    srcs = ["multi_staged_session_test.cc"],
    deps = [
        ":multi_staged_session",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/framework:threadpool",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "omni_engine_header",
    hdrs = ["omni_engine.h"],
    deps = [
        ":omni_session",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "omni_registry",
    srcs = ["omni_registry.cc"],
    hdrs = ["omni_registry.h"],
    deps = [
        ":omni_engine_header",
        ":omni_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//omni/base:model_utils",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/proto:asr_metadata_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:tts_metadata_cc_proto",
        "//runtime/util:file_util",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "omni_registry_test",
    srcs = ["omni_registry_test.cc"],
    deps = [
        ":omni_engine_header",
        ":omni_registry",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",  # build_cleaner: keep
        "@litert//litert/c:litert_common",
        "@litert//litert/c/options:litert_gpu_options",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/components:model_resources",
        "//runtime/proto:asr_metadata_cc_proto",
        "//runtime/proto:asr_session_config_cc_proto",
        "//runtime/proto:tts_session_config_cc_proto",
        "//runtime/util:file_util",
        "//schema/core:litertlm_header_schema",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "omni_engine",
    srcs = ["omni_engine.cc"],
    hdrs = ["omni_engine.h"],
    deps = [
        ":omni_registry",
        ":omni_session",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "omni_engine_test",
    srcs = ["omni_engine_test.cc"],
    deps = [
        ":multi_staged_session",
        ":omni_engine",
        ":omni_registry",
        ":omni_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni/asr:asr_session",
        "//omni/asr:audio_preprocessor",
        "//omni/asr:audio_source",
        "//omni/asr:detokenizer",
        "//omni/asr:levenshtein_text_merger",
        "//omni/asr:moonshine",  # build_cleaner: keep
        "//omni/asr:speech_recognizer",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/text2image:bonsai_flux2",  # build_cleaner: keep
        "//omni/tts:kokoro",  # build_cleaner: keep
        "//omni/tts:stream_text_source",
        "//omni/tts:text_chunk_utils",
        "//omni/tts:text_source",
        "//omni/tts:tts_session",
        "//omni/tts:vocoder",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:asr_metadata_cc_proto",
        "//runtime/proto:asr_session_config_cc_proto",
        "//runtime/proto:tts_session_config_cc_proto",
        "//support/util:test_utils",
    ],
)
