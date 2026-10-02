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

#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "c/conversation.h"
#include "c/conversation_internal.h"  // IWYU pragma: keep
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "c/experimental_internal.h"  // IWYU pragma: keep
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"

extern "C" {

LiteRtLmStatusCode
litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(
    LiteRtLmEngine* engine, bool enable_metal_residency_set) {
  if (engine == nullptr || engine->engine == nullptr) {
    ABSL_LOG(ERROR) << "Engine is null.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Engine is null.");
  }
  auto status = engine->engine->UpdateGpuEnableMetalResidencySet(
      enable_metal_residency_set);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to update GPU enable metal residency set: "
                    << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

// TODO(b/549220913): Migrate debugger from build-time macro to runtime
// configuration in EngineSettings / SessionConfig.
LiteRtLmStatusCode litert_lm_experimental_is_debugger_enabled(
    bool* out_enabled) {
  LITERT_LM_C_RETURN_IF_NULL(out_enabled);
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  *out_enabled = true;
#else
  *out_enabled = false;
#endif
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_session_get_debug_info(
    LiteRtLmSession* session, LiteRtLmSessionDebugInfo** out_debug_info) {
  LITERT_LM_C_RETURN_IF_NULL(out_debug_info);
  *out_debug_info = nullptr;
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
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
  if (!conversation || !conversation->conversation) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid conversation.");
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
