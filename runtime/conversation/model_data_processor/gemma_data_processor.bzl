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

"""Starlark macro for defining standard and lite Gemma model data processor targets."""

# [Google-internal load of `cc_library`]
load("@rules_cc//cc:defs.bzl", "cc_library")

def gemma_data_processor_libraries(
        name,
        srcs,
        hdrs,
        enable_define,
        common_deps,
        visibility = None):
    """Defines `<name>` and `<name>_lite` cc_library targets for a Gemma data processor.

    `<name>` builds tool call constraints with the legacy FST constraint
    provider by default, and with LLGuidance when FST constraints are disabled.
    `<name>_lite` links neither, so it does not support constrained decoding.

    Args:
      name: Base name of the data processor library (e.g., "gemma3_data_processor").
      srcs: Source files for the data processor.
      hdrs: Header files for the data processor.
      enable_define: Preprocessor macro enabling registration (e.g., "ENABLE_GEMMA3_DATA_PROCESSOR").
      common_deps: Dependencies shared by both the standard and `_lite` targets.
      visibility: Optional visibility override.
    """
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        defines = [enable_define] + select({
            "//runtime/conversation/model_data_processor:litert_lm_fst_constraints_disabled": [],
            "//conditions:default": ["LITERT_LM_USE_FST_TOOL_CALL_CONSTRAINTS"],
        }),
        visibility = visibility,
        deps = common_deps + [
            "//runtime/components/tool_use:parser_utils",
        ] + select({
            "//runtime/conversation/model_data_processor:litert_lm_fst_constraints_disabled": [
                "//runtime/components/constrained_decoding:llg_constraint_config",
                "//runtime/components/constrained_decoding:llg_constraint_provider",
                "//runtime/components/constrained_decoding:llguidance_schema_utils",
            ],
            "//conditions:default": [
                "//runtime/components/constrained_decoding:gemma_model_constraint_provider_lib",
            ],
        }),
    )

    cc_library(
        name = name + "_lite",
        srcs = srcs,
        hdrs = hdrs,
        defines = [
            enable_define,
            "LITERT_LM_TOOL_CALL_CONSTRAINTS_DISABLED",
        ],
        visibility = visibility,
        deps = common_deps + [
            "//runtime/components/tool_use:parser_utils_lite",
        ],
    )
