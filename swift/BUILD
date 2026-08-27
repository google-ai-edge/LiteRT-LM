# Copyright 2025 Google LLC
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
load("@build_bazel_rules_apple//apple:apple.bzl", "apple_universal_binary")
load("@build_bazel_rules_apple//apple:apple.bzl", "apple_xcframework")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

# Swift API for LiteRT LM.
swift_library(
    name = "LiteRT_LM_Swift",
    srcs = [
        "Benchmark.swift",
        "Capabilities.swift",
        "Config.swift",
        "Conversation.swift",
        "EmbeddingEngine.swift",
        "EmbeddingEngineConfig.swift",
        "Engine.swift",
        "ExperimentalFlags.swift",
        "LiteRTLMError.swift",
        "Message.swift",
        "ResponseFormat.swift",
        "Tool.swift",
        "ToolManager.swift",
    ],
    copts = ["-enable-testing"],
    features = ["-swift.enable_v6"],
    module_name = "LiteRTLM",
    deps = [


        "//c:engine",
    ],
)

# Swift library containing test sources for Engine.
swift_library(
    name = "LiteRT_LM_EngineTestsLib",
    testonly = True,
    srcs = ["EngineTests.swift"],
    copts = ["-enable-testing"],
    data = ["//runtime/testdata"],
    features = ["-swift.enable_v6"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

# Swift library containing test sources for Conversation.
swift_library(
    name = "LiteRT_LM_ConversationTestsLib",
    testonly = True,
    srcs = ["ConversationTests.swift"],
    copts = ["-enable-testing"],
    data = ["//runtime/testdata"],
    features = ["-swift.enable_v6"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

# Swift library containing test sources for Tool.
swift_library(
    name = "LiteRT_LM_ToolTestsLib",
    testonly = True,
    srcs = ["ToolTests.swift"],
    copts = ["-enable-testing"],
    data = ["//runtime/testdata"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

# Swift library containing test sources for Message.
swift_library(
    name = "LiteRT_LM_MessageTestsLib",
    testonly = True,
    srcs = ["MessageTests.swift"],
    copts = ["-enable-testing"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

swift_interop_hint(
    name = "CLiteRTLM_swift_hint",
    module_name = "CLiteRTLM",
)

cc_library(
    name = "CLiteRTLM_objc",
    hdrs = [
        "//c:capabilities.h",
        "//c:conversation.h",
        "//c:embedding_engine.h",
        "//c:engine.h",
    ],
    aspect_hints = [":CLiteRTLM_swift_hint"],
    deps = [
        "//c:engine",
    ],
)

apple_xcframework(
    name = "CLiteRTLM",
    bundle_id = "com.google.odml.litertlm.CLiteRTLM",
    features = ["exported_symbols"],
    infoplists = ["Info.plist"],
    ios = {
        "simulator": [
            "arm64",
        ],
        "device": [
            "arm64",
        ],
    },
    minimum_os_versions = {
        "ios": "15.0",
    },
    public_hdrs = [
        "//c:engine.h",
        "//c:conversation.h",
        "//c:embedding_engine.h",
        "//c:capabilities.h",
    ],
    deps = [
        ":CLiteRTLM_objc",
    ],
)

macos_dylib(
    name = "CLiteRTLM_dylib",
    features = ["exported_symbols"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libCLiteRTLM_mac.dylib",
    ],
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [":CLiteRTLM_objc"],
)

apple_universal_binary(
    name = "CLiteRTLM_universal",
    binary = ":CLiteRTLM_dylib",
    forced_cpus = [
        "arm64",
        "x86_64",
    ],
    minimum_os_version = "12.0",
    platform_type = "macos",
)

genrule(
    name = "CLiteRTLM_mac",
    srcs = [
        ":CLiteRTLM_universal",
        "//c:engine.h",
        "//c:conversation.h",
        "//c:embedding_engine.h",
        "//c:capabilities.h",
    ],
    outs = ["CLiteRTLM_mac.zip"],
    cmd = """
       mkdir -p CLiteRTLM_mac/Headers
       # Copy the universal dylib
       cp $(location :CLiteRTLM_universal) CLiteRTLM_mac/libCLiteRTLM_mac.dylib
       # Copy the headers
       cp $(location //c:engine.h) CLiteRTLM_mac/Headers/
       cp $(location //c:conversation.h) CLiteRTLM_mac/Headers/
       cp $(location //c:embedding_engine.h) CLiteRTLM_mac/Headers/
       cp $(location //c:capabilities.h) CLiteRTLM_mac/Headers/
       # Create module map
       cat << 'EOF' > CLiteRTLM_mac/Headers/module.modulemap
module CLiteRTLM {
   header "engine.h"
   header "conversation.h"
   header "embedding_engine.h"
   header "capabilities.h"
   export *
}
EOF
       # Zip it all up
       zip -r $@ CLiteRTLM_mac
    """,
    visibility = ["//visibility:public"],
)

ios_unit_test(
    name = "LiteRT_LM_Engine_Tests",
    data = [
        "//runtime/testdata",
    ],
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_EngineTestsLib",
    ],
)

ios_unit_test(
    name = "LiteRT_LM_Conversation_Tests",
    data = [
        "//runtime/testdata",
    ],
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_ConversationTestsLib",
    ],
)

ios_unit_test(
    name = "LiteRT_LM_Tool_Tests",
    data = [
        "//runtime/testdata",
    ],
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_ToolTestsLib",
    ],
)

ios_unit_test(
    name = "LiteRT_LM_Message_Tests",
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_MessageTestsLib",
    ],
)

# Swift library containing test sources for Capabilities.
swift_library(
    name = "LiteRT_LM_CapabilitiesTestsLib",
    testonly = True,
    srcs = ["CapabilitiesTests.swift"],
    copts = ["-enable-testing"],
    data = ["//runtime/testdata"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

ios_unit_test(
    name = "LiteRT_LM_Capabilities_Tests",
    data = [
        "//runtime/testdata",
    ],
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_CapabilitiesTestsLib",
    ],
)

# Swift library containing test sources for EmbeddingEngine.
swift_library(
    name = "LiteRT_LM_EmbeddingEngineTestsLib",
    testonly = True,
    srcs = ["EmbeddingEngineTests.swift"],
    data = ["//runtime/testdata"],
    deps = [
        ":LiteRT_LM_Swift",

    ],
)

ios_unit_test(
    name = "LiteRT_LM_EmbeddingEngine_Tests",
    data = [
        "//runtime/testdata",
    ],
    minimum_os_version = "15.0",
    # The ios_unit_test depends on the swift_library containing the test code.
    deps = [
        ":LiteRT_LM_EmbeddingEngineTestsLib",
    ],
)
