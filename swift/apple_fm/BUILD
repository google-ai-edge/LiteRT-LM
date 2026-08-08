# Apple Foundation Models adapter for LiteRT-LM

package(default_visibility = ["//visibility:public"])

# TODO: The FoundationModels API churned significantly between early Apple betas
# and beta 3 (e.g., GenerationOptions.samplingMode, introduction of LanguageModel).
# Uncomment the targets below once moving to Apple SDK/Xcode toolchain to Beta 3 or newer.
#
# swift_library(
#     name = "LiteRTLMFoundationModels_Swift",
#     srcs = [
#         "LiteRTLanguageModel.swift",
#         "LiteRTSegments.swift",
#         "CGImagePNG.swift",
#     ],
#     copts = ["-enable-testing"],
#     module_name = "LiteRTLMFoundationModels",
#     deps = [


#         "//swift:LiteRT_LM_Swift",
#     ],
# )
#
# # Headless macOS execution test
# swift_library(
#     name = "LiteRT_LM_FMTest_Lib",
#     srcs = ["main.swift"],
#     deps = [
#         ":LiteRTLMFoundationModels_Swift",


#         "//swift:LiteRT_LM_Swift",
#     ],
# )
#
# macos_command_line_application(
#     name = "fmtest",
#     minimum_os_version = "12.0",
#     deps = [
#         ":LiteRT_LM_FMTest_Lib",
#     ],
# )
#
# ios_unit_test(
#     name = "LiteRTLMFoundationModels_Test",
#     minimum_os_version = "27.0",
#     deps = [
#         ":AdapterTests_Lib",
#     ],
# )
#
# swift_library(
#     name = "AdapterTests_Lib",
#     srcs = ["AdapterTests.swift"],
#     deps = [
#         ":LiteRTLMFoundationModels_Swift",



#         "//swift:LiteRT_LM_Swift",
#     ],
# )
