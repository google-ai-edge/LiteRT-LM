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

# [Google-internal load of `cc_binary`]
# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files([
    "capabilities.cc",
    "capabilities.h",
    "conversation.cc",
    "conversation.h",
    "embedding_engine.cc",
    "embedding_engine.h",
    "engine.cc",
    "engine.h",
    "experimental.cc",
    "experimental.h",
])

ENGINE_LITE_COMMON_DEPS = [
    ":engine_internal",
    "@com_google_absl//absl/base:core_headers",
    "@com_google_absl//absl/base:log_severity",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/functional:any_invocable",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:initialize",
    "@com_google_absl//absl/log:log_sink_registry",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time",
    "//support/tokenizer",
    "//runtime/components/constrained_decoding:no_repeat_ngram_config",
    "//runtime/components/constrained_decoding:repetition_penalty_config",
    "//runtime/components/constrained_decoding:suppress_tokens_config",
    "//runtime/engine:engine_factory",
    "//runtime/engine:engine_interface",
    "//runtime/engine:engine_settings",
    "//runtime/engine:io_types",
    "//runtime/executor:executor_settings_base",
    "//runtime/executor:llm_executor_settings",
    "//runtime/proto:engine_cc_proto",
    "//runtime/proto:llm_metadata_cc_proto",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/proto:token_cc_proto",
    "//runtime/util:logging",
    "//runtime/util:scoped_file",
]

CONVERSATION_DEPS = [
    ":conversation_internal",
    "@nlohmann_json//:json",
    "//runtime/components:prompt_template",
    "//runtime/components/constrained_decoding:llg_constraint_config",
    "//runtime/conversation",
    "//runtime/conversation:io_types",
    "//runtime/conversation:thinking_config",
    "//runtime/conversation/model_data_processor:config_registry",
    "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
]

EMBEDDING_ENGINE_DEPS = [
    ":embedding_engine_internal",
    "//runtime/components:model_resources",
    "//runtime/core:embedding_engine_impl",
    "//runtime/engine:embedding_engine",
    "//runtime/engine:embedding_engine_settings",
    "//runtime/executor:embedding_executor_base",
    "//runtime/executor:litert_compiled_model_executor_utils",
    "//support/preprocessor:audio_preprocessor",
    "//support/preprocessor:audio_preprocessor_miniaudio",
]

CAPABILITIES_DEPS = [
    "//schema/capabilities",
]

EXPERIMENTAL_DEPS = [
    ":experimental_internal",
]

ENGINE_COMMON_DEPS = (
    ENGINE_LITE_COMMON_DEPS +
    CONVERSATION_DEPS +
    EMBEDDING_ENGINE_DEPS +
    CAPABILITIES_DEPS +
    EXPERIMENTAL_DEPS
)

# Preprocessor define map for runtime debugger tracing support.
# Evaluates statically against the global `--define LITERT_LM_DEBUGGER_ENABLED=1` build flag,
# and is safely inherited by all `:engine` and `:engine_lite` C-API variant targets.
LITERT_LM_DEBUGGER_DEFINES = select({
    "//runtime/util:litert_lm_debugger_enabled": [
        "LITERT_LM_DEBUGGER_ENABLED=1",
    ],
    "//conditions:default": [],
})

# Header library for engine + session C API.
cc_library(
    name = "engine_headers",
    hdrs = ["engine.h"],
    visibility = ["//visibility:public"],
)

# Header library for engine + session + conversation C API.
cc_library(
    name = "conversation_headers",
    hdrs = ["conversation.h"],
    visibility = ["//visibility:public"],
    deps = [":engine_headers"],
)

# Header library for capabilities C API.
cc_library(
    name = "capabilities_headers",
    hdrs = ["capabilities.h"],
    visibility = ["//visibility:public"],
    deps = [":engine_headers"],
)

# Header library for embedding engine C API.
cc_library(
    name = "embedding_engine_headers",
    hdrs = ["embedding_engine.h"],
    visibility = ["//visibility:public"],
    deps = [":engine_headers"],
)

# Header library for experimental C API.
cc_library(
    name = "experimental_headers",
    hdrs = ["experimental.h"],
    visibility = ["//visibility:public"],
    deps = [":engine_headers"],
)

cc_library(
    name = "engine_internal",
    hdrs = ["engine_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":engine_headers",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:token_cc_proto",
    ],
)

cc_library(
    name = "conversation_internal",
    hdrs = ["conversation_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":conversation_headers",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/conversation",
        "//runtime/conversation:thinking_config",
        "//runtime/engine:engine_settings",
    ],
)

cc_library(
    name = "embedding_engine_internal",
    hdrs = ["embedding_engine_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":embedding_engine_headers",
        "//runtime/engine:embedding_engine",
        "//runtime/engine:embedding_engine_settings",
        "//runtime/engine:io_types",
        "//support/preprocessor:image_preprocessor",
    ],
)

cc_library(
    name = "experimental_internal",
    hdrs = ["experimental_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental_headers",
        "//runtime/engine:io_types",
    ],
)

# LiteRT LM C API target for Engine + Session only.
# Use this target when only low-level engine and session APIs (prefill, decode, scoring)
# are required, avoiding expensive conversation dependencies (prompt templates, JSON handling, etc.).
cc_library(
    name = "engine_lite",
    srcs = ["engine.cc"],
    hdrs = ["engine.h"],
    defines = LITERT_LM_DEBUGGER_DEFINES,
    visibility = ["//visibility:public"],
    deps = ENGINE_LITE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
    alwayslink = True,
)

# CPU-only variant of :engine_lite for Engine + Session functionality.
cc_library(
    name = "engine_lite_cpu",
    srcs = ["engine.cc"],
    hdrs = ["engine.h"],
    defines = LITERT_LM_DEBUGGER_DEFINES,
    visibility = ["//visibility:public"],
    deps = ENGINE_LITE_COMMON_DEPS + [
        "//runtime/core:engine_advanced_impl_cpu_only",
    ],
)

cc_test(
    name = "engine_lite_test",
    srcs = ["engine_lite_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:linux": [],
        "@platforms//os:osx": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":engine_internal",
        ":engine_lite",
        "@com_google_googletest//:gtest_main",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
    ],
)

# LiteRT LM C API target for Engine + Session + Conversation.
# Use this target when full conversation capabilities (message management, system prompts,
# tool calls, constrained decoding, thinking config) are required.
cc_library(
    name = "engine",
    srcs = [
        "capabilities.cc",
        "conversation.cc",
        "embedding_engine.cc",
        "engine.cc",
        "experimental.cc",
    ],
    hdrs = [
        "capabilities.h",
        "conversation.h",
        "embedding_engine.h",
        "engine.h",
        "experimental.h",
    ],
    defines = LITERT_LM_DEBUGGER_DEFINES,
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
    alwayslink = True,
)

# CPU-only variant of :engine for Engine + Session + Conversation functionality.
cc_library(
    name = "engine_cpu",
    srcs = [
        "capabilities.cc",
        "conversation.cc",
        "embedding_engine.cc",
        "engine.cc",
        "experimental.cc",
    ],
    hdrs = [
        "capabilities.h",
        "conversation.h",
        "embedding_engine.h",
        "engine.h",
        "experimental.h",
    ],
    defines = LITERT_LM_DEBUGGER_DEFINES,
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_advanced_impl_cpu_only",
    ],
)

cc_test(
    name = "capabilities_test",
    srcs = ["capabilities_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:linux": [],
        "@platforms//os:osx": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":capabilities_headers",
        ":engine",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "engine_test",
    srcs = ["engine_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:linux": [],
        "@platforms//os:osx": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":conversation_headers",
        ":conversation_internal",
        ":engine",
        ":engine_headers",
        ":engine_internal",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/conversation:thinking_config",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
    ],
)

cc_test(
    name = "embedding_engine_test",
    srcs = ["embedding_engine_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:linux": [],
        "@platforms//os:osx": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":embedding_engine_headers",
        ":engine",
        ":engine_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "litert-lm",
    srcs = [
        "capabilities.cc",
        "conversation.cc",
        "embedding_engine.cc",
        "engine.cc",
        "experimental.cc",
    ],
    copts = ["-fvisibility=hidden"],
    features = ["-legacy_whole_archive"],
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
            # Report unresolved symbol references as errors during linking.
            "-Wl,--no-undefined",
            # Strip all symbols to reduce library size (debug info is removed).
            "-Wl,--strip-all",
            # Remove unused sections/functions to further reduce library size.
            "-Wl,--gc-sections",
            # Ensure symbols are resolved internally first.
            "-Wl,-Bsymbolic",
            # Update the rpath
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = 1,
    deps = ENGINE_COMMON_DEPS + [
        ":capabilities_headers",
        ":conversation_headers",
        ":embedding_engine_headers",
        ":engine_headers",
        ":experimental_headers",
        "//runtime/core:engine_impl",
    ] + select({
        "//conditions:default": [],
    }),
)
