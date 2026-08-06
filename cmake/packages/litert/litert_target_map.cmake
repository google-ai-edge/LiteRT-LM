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

# Updated on 2026-04-06
set(LITERT_TARGET_MAP
    "litert::cc_api=${LITERT_BUILD_DIR}/cc/liblitert_cc_api.a"
    "litert::cc_options=${LITERT_BUILD_DIR}/cc/options/liblitert_cc_options.a"
    "litert::cc_internal=${LITERT_BUILD_DIR}/cc/internal/liblitert_cc_internal.a"
    "litert::cc_internal_extra=${LITERT_BUILD_DIR}/cc/internal/liblitert_cc_internal_extra.a"
    "litert::runtime=${LITERT_BUILD_DIR}/runtime/liblitert_runtime.a"
    "litert::tool_display=${LITERT_BUILD_DIR}/tools/liblitert_tool_display.a"
    "litert::dump=${LITERT_BUILD_DIR}/tools/liblitert_dump.a"
    "litert::npu_numerics_check=${LITERT_BUILD_DIR}/tools/liblitert_npu_numerics_check.a"
    "litert::tensor_utils=${LITERT_BUILD_DIR}/tools/liblitert_tensor_utils.a"
    "litert::apply_plugin=${LITERT_BUILD_DIR}/tools/liblitert_apply_plugin.a"
    "litert::tool_flags_common=${LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_common.a"
    "litert::tool_flags_apply_plugin=${LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_apply_plugin.a"
    "litert::tool_flags_mediatek=${LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_mediatek.a"
    "litert::tool_flags_google_tensor=${LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_google_tensor.a"
    "litert::tool_flags_samsung=${LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_samsung.a"
    "litert::tool_flags_intel_openvino=${LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_intel_openvino.a"
    "litert::tool_flags_qualcomm=${LITERT_BUILD_DIR}/tools/flags/vendors/liblitert_tool_flags_qualcomm.a"
    "litert::tool_flags_types=${LITERT_BUILD_DIR}/tools/flags/liblitert_tool_flags_types.a"
    "litert::qnn_saver_utils=${LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_saver_utils.a"
    "litert::qnn_manager=${LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_manager.a"
    "litert::qnn_context_binary_info=${LITERT_BUILD_DIR}/vendors/qualcomm/libqnn_context_binary_info.a"
    "litert::qnn_tensor_pool=${LITERT_BUILD_DIR}/vendors/qualcomm/core/libqnn_tensor_pool.a"
    "litert::qnn_backends=${LITERT_BUILD_DIR}/vendors/qualcomm/core/backends/libqnn_backends.a"
    "litert::qnn_dump_graph=${LITERT_BUILD_DIR}/vendors/qualcomm/core/dump/libqnn_dump_graph.a"
    "litert::qnn_transformation=${LITERT_BUILD_DIR}/vendors/qualcomm/core/transformation/libqnn_transformation.a"
    "litert::qnn_wrappers=${LITERT_BUILD_DIR}/vendors/qualcomm/core/wrappers/libqnn_wrappers.a"
    "litert::qnn_common=${LITERT_BUILD_DIR}/vendors/qualcomm/core/libqnn_common.a"
    "litert::qnn_builders=${LITERT_BUILD_DIR}/vendors/qualcomm/core/builders/libqnn_builders.a"
    "litert::qnn_miscs=${LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_miscs.a"
    "litert::qnn_model=${LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_model.a"
    "litert::qnn_log=${LITERT_BUILD_DIR}/vendors/qualcomm/core/utils/libqnn_log.a"
    "litert::qnn_soc_table=${LITERT_BUILD_DIR}/vendors/qualcomm/core/schema/libqnn_soc_table.a"
    "litert::logging=${LITERT_BUILD_DIR}/c/liblitert_logging.a"
    "litert::c_api=${LITERT_BUILD_DIR}/c/liblitert_c_api.a"
    "litert::compiler_plugin=${LITERT_BUILD_DIR}/compiler/liblitert_compiler_plugin.a"
    "litert::core=${LITERT_BUILD_DIR}/core/liblitert_core.a"
    "litert::core_cache=${LITERT_BUILD_DIR}/core/cache/liblitert_core_cache.a"
    "litert::core_model=${LITERT_BUILD_DIR}/core/model/liblitert_core_model.a"
)