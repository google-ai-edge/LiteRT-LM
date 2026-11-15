# Copyright 2026 Google LLC.
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

set(LITERTLM_LITERT_TARGET_MAP
    "litert::litert_cc_api=${LITERTLM_LITERT_BUILD_DIR}/cc/liblitert_cc_api.a"
    "litert::litert_cc_internal=${LITERTLM_LITERT_BUILD_DIR}/cc/internal/liblitert_cc_internal.a"
    "litert::litert_cc_internal_extra=${LITERTLM_LITERT_BUILD_DIR}/cc/internal/liblitert_cc_internal_extra.a"
    "litert::litert_runtime=${LITERTLM_LITERT_BUILD_DIR}/runtime/liblitert_runtime.a"
    "litert::litert_tool_display=${LITERTLM_LITERT_BUILD_DIR}/tools/liblitert_tool_display.a"
    "litert::litert_dump=${LITERTLM_LITERT_BUILD_DIR}/tools/liblitert_dump.a"
    "litert::litert_tensor_utils=${LITERTLM_LITERT_BUILD_DIR}/tools/liblitert_tensor_utils.a"
    "litert::litert_apply_plugin=${LITERTLM_LITERT_BUILD_DIR}/tools/liblitert_apply_plugin.a"
    "litert::litert_tool_flags_common=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_common.a"
    "litert::litert_tool_flags_apply_plugin=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_apply_plugin.a"
    "litert::litert_tool_flags_mediatek=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_mediatek.a"
    "litert::litert_tool_flags_google_tensor=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_google_tensor.a"
    "litert::litert_tool_flags_samsung=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_samsung.a"
    "litert::litert_tool_flags_intel_openvino=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_intel_openvino.a"
    "litert::litert_tool_flags_qualcomm=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_qualcomm.a"
    "litert::litert_options_parser_registry=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_options_parser_registry.a"
    "litert::litert_tool_flags_types=${LITERTLM_LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_types.a"
    "litert::samsung_soc_model=${LITERTLM_LITERT_BUILD_DIR}/vendors/samsung/libsamsung_soc_model.a"
    "litert::samsung_byte_code=${LITERTLM_LITERT_BUILD_DIR}/vendors/samsung/schema/libsamsung_byte_code.a"
    "litert::samsung_ai_litecore_manager=${LITERTLM_LITERT_BUILD_DIR}/vendors/samsung/libsamsung_ai_litecore_manager.a"
    "litert::qnn_saver_utils=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_saver_utils.a"
    "litert::qnn_manager=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_manager.a"
    "litert::qnn_context_binary_info=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_context_binary_info.a"
    "litert::qnn_tensor_pool=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/libqnn_tensor_pool.a"
    "litert::qnn_backends=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/backends/libqnn_backends.a"
    "litert::qnn_dump_graph=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/dump/libqnn_dump_graph.a"
    "litert::qnn_transformation=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/transformation/libqnn_transformation.a"
    "litert::qnn_wrappers=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/wrappers/libqnn_wrappers.a"
    "litert::qnn_common=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/libqnn_common.a"
    "litert::qnn_builders=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/builders/libqnn_builders.a"
    "litert::qnn_miscs=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_miscs.a"
    "litert::qnn_model=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_model.a"
    "litert::qnn_flexbuffer_helpers=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_flexbuffer_helpers.a"
    "litert::qnn_log=${LITERTLM_LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_log.a"
    "litert::litert_logging=${LITERTLM_LITERT_BUILD_DIR}/c/liblitert_logging.a"
    "litert::litert_c_api=${LITERTLM_LITERT_BUILD_DIR}/c/liblitert_c_api.a"
    "litert::litert_compiler_plugin=${LITERTLM_LITERT_BUILD_DIR}/compiler/liblitert_compiler_plugin.a"
    "litert::litert_core=${LITERTLM_LITERT_BUILD_DIR}/core/liblitert_core.a"
    "litert::litert_core_cache=${LITERTLM_LITERT_BUILD_DIR}/core/cache/liblitert_core_cache.a"
    "litert::litert_core_model=${LITERTLM_LITERT_BUILD_DIR}/core/model/liblitert_core_model.a"
)
