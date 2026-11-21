// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "c/experimental.h"

#include <cstdint>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "c/c_api_internal.h"
#include "c/conversation.h"
#include "c/conversation_internal.h"  // IWYU pragma: keep
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "c/experimental_internal.h"  // IWYU pragma: keep
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"  // IWYU pragma: keep
#include "runtime/executor/llm_executor_settings.h"

namespace {

using ::litert::lm::c::IsValidConversation;
using ::litert::lm::c::IsValidEngine;
using ::litert::lm::c::IsValidEngineSettings;
using ::litert::lm::c::IsValidSession;
using ::litert::lm::c::RawEnumValue;
using ::litert::lm::c::ReturnError;
using ::litert::lm::c::UpdateMainBackendConfig;

}  // namespace

extern "C" {

LiteRtLmStatusCode
litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(
    LiteRtLmEngine* engine, bool enable_metal_residency_set) {
  if (!IsValidEngine(engine)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_ERROR(engine->engine->UpdateGpuEnableMetalResidencySet(
      enable_metal_residency_set));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_enable_debugger(
    LiteRtLmEngineSettings* settings, bool enable_debugger) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->SetEnableDebugger(enable_debugger);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_gpu_use_ir_model(
    LiteRtLmEngineSettings* settings, bool use_ir_model) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableMainExecutorSettings().SetUseIrModel(
      use_ir_model);
  if (settings->settings->GetMutableVisionExecutorSettings().has_value()) {
    settings->settings->GetMutableVisionExecutorSettings()->SetUseIrModel(
        use_ir_model);
  }
  if (settings->settings->GetMutableAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetUseIrModel(
        use_ir_model);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_experimental_engine_settings_set_head_chunked_attention_mib(
    LiteRtLmEngineSettings* settings, int32_t head_chunked_attention_mib) {
  return UpdateMainBackendConfig<litert::lm::GpuArtisanConfig>(
      settings, "head_chunked_attention_mib",
      [head_chunked_attention_mib](litert::lm::GpuArtisanConfig& config) {
        config.head_chunked_attention_mib = head_chunked_attention_mib;
      });
}

LiteRtLmStatusCode litert_lm_experimental_engine_settings_set_kv_cache_type(
    LiteRtLmEngineSettings* settings, LiteRtLmKvCacheType kv_cache_type) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  litert::lm::KvCacheType cpp_kv_cache_type;
  switch (RawEnumValue(kv_cache_type)) {
    case kLiteRtLmKvCacheTypeDefault:
      cpp_kv_cache_type = litert::lm::KvCacheType::DEFAULT;
      break;
    case kLiteRtLmKvCacheTypeInt8:
      cpp_kv_cache_type = litert::lm::KvCacheType::INT8;
      break;
    default:
      return ReturnError(absl::StatusCode::kInvalidArgument,
                         "Unknown LiteRtLmKvCacheType.");
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  absl::StatusOr<litert::lm::GpuArtisanConfig> config =
      main_settings.MutableBackendConfig<litert::lm::GpuArtisanConfig>();
  if (!config.ok()) {
    if (cpp_kv_cache_type == litert::lm::KvCacheType::INT8) {
      ABSL_LOG(WARNING)
          << "INT8 KV cache is not supported for this model; continuing with "
             "default KV cache.";
    }
    return kLiteRtLmStatusOk;
  }
  config->kv_cache_type = cpp_kv_cache_type;
  main_settings.SetBackendConfig(*config);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_session_get_debug_info(
    LiteRtLmSession* session, LiteRtLmSessionDebugInfo** out_debug_info) {
  LITERT_LM_C_RETURN_IF_NULL(out_debug_info);
  *out_debug_info = nullptr;
  if (!IsValidSession(session)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto debug_info = session->session->GetSessionDebugInfo();
  if (debug_info.has_value()) {
    *out_debug_info = new LiteRtLmSessionDebugInfo{std::move(*debug_info)};
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_conversation_get_session_debug_info(
    LiteRtLmConversation* conversation,
    LiteRtLmSessionDebugInfo** out_debug_info) {
  LITERT_LM_C_RETURN_IF_NULL(out_debug_info);
  *out_debug_info = nullptr;
  if (!IsValidConversation(conversation)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto debug_info = conversation->conversation->GetSessionDebugInfo();
  if (debug_info.has_value()) {
    *out_debug_info = new LiteRtLmSessionDebugInfo{std::move(*debug_info)};
  }
  return kLiteRtLmStatusOk;
}

void litert_lm_experimental_session_debug_info_delete(
    LiteRtLmSessionDebugInfo* debug_info) {
  delete debug_info;
}

LiteRtLmStatusCode litert_lm_experimental_session_debug_info_get_capture_dir(
    const LiteRtLmSessionDebugInfo* debug_info, const char** out_capture_dir) {
  LITERT_LM_C_RETURN_IF_NULL(out_capture_dir);
  *out_capture_dir = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(debug_info);
  *out_capture_dir = debug_info->debug_info.capture_dir.c_str();
  return kLiteRtLmStatusOk;
}

}  // extern "C"
