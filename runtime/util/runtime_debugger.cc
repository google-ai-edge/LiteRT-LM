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

#include "runtime/util/runtime_debugger.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/util/safetensors_util.h"

namespace litert::lm {

namespace {

// Internal sentinel values for in-memory and bypass caching directories.
constexpr absl::string_view kMemoryCacheDir = ":memory";
constexpr absl::string_view kNoCacheDir = ":nocache";

// Target subdirectory name for tensor dumps, relative to the preferred cache
// dir.
constexpr absl::string_view kDebuggerSubdir = "litert_lm_debugger";

// Trace file specifications for generated tokens and tensor files.
constexpr absl::string_view kGeneratedTokensFileName = "generated_tokens.jsonl";
constexpr absl::string_view kPreEventPrefix = "pre_";
constexpr absl::string_view kPostEventPrefix = "post_";

// JSON keys for token-generation and trace-step observability.
constexpr absl::string_view kTokenIdsKey = "token_ids";
constexpr absl::string_view kTextsKey = "texts";
constexpr absl::string_view kScoresKey = "scores";

// HuggingFace Safetensors metadata signature mapping keys.
constexpr absl::string_view kMetadataSignature = "signature";
constexpr absl::string_view kMetadataStep = "step";

// Helper predicate identifying Prefill-phase LiteRT signatures.
bool IsPrefillSignature(absl::string_view signature_name) {
  return absl::StrContains(signature_name, "prefill");
}

// Serializes the content of a LiteRT tensor buffer to disk in HuggingFace
// Safetensors format.
absl::Status SaveTensorBufferToFile(absl::string_view root_directory,
                                    absl::string_view signature_name,
                                    absl::string_view tensor_name,
                                    absl::string_view event_prefix, int step,
                                    const ::litert::TensorBuffer& buffer) {
  const std::string filename =
      absl::StrCat(signature_name, "_", event_prefix, tensor_name, "_step_",
                   step, kSafetensorsExtension);
  const std::string filepath =
      (std::filesystem::path{std::string(root_directory)} / filename).string();

  auto packed_size_or = buffer.PackedSize();
  if (!packed_size_or.HasValue()) {
    return absl::InternalError("Unable to query tensor packed size.");
  }
  const size_t bytes = *packed_size_or;

  // TensorBufferScopedLock::Create requires non-const TensorBuffer& to obtain
  // read-lock handle.
  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *const_cast<::litert::TensorBuffer*>(&buffer),
      ::litert::TensorBuffer::LockMode::kRead);
  if (!lock_and_addr.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to lock tensor ", tensor_name));
  }
  void* addr = lock_and_addr->second;

  SafetensorsDtype dtype = SafetensorsDtype::kDefault;
  size_t element_size = sizeof(float);
  std::vector<int64_t> shape;

  if (auto tensor_type_or = buffer.TensorType(); tensor_type_or.HasValue()) {
    auto dtype_info = GetSafetensorsDtypeInfo(tensor_type_or->ElementType());
    dtype = dtype_info.dtype;
    element_size = dtype_info.element_size;

    const auto& dims = tensor_type_or->Layout().Dimensions();
    if (!dims.empty()) {
      shape.assign(dims.begin(), dims.end());
    }
  }

  if (shape.empty()) {
    ABSL_LOG(WARNING) << "Skipped empty-shape tensor: " << event_prefix
                      << tensor_name;
    return absl::OkStatus();
  }

  const std::string full_tensor_name = absl::StrCat(event_prefix, tensor_name);
  const absl::flat_hash_map<std::string, std::string> metadata = {
      {std::string(kMetadataSignature), std::string(signature_name)},
      {std::string(kMetadataStep), absl::StrCat(step)},
  };
  const std::string json_header =
      BuildSafetensorsHeader(full_tensor_name, dtype, shape, bytes, metadata);
  const uint64_t header_size = json_header.size();

  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for capturing: ", filepath));
  }
  outfile.write(reinterpret_cast<const char*>(&header_size),
                sizeof(header_size));
  outfile.write(json_header.data(), header_size);
  outfile.write(reinterpret_cast<const char*>(addr), bytes);
  outfile.close();

  ABSL_LOG(INFO) << "Captured safetensors tensor " << event_prefix
                 << tensor_name << " to " << filepath;
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<RuntimeDebugger> RuntimeDebugger::Create(
    absl::string_view preferred_cache_dir) {
  if (preferred_cache_dir.empty() || preferred_cache_dir == kMemoryCacheDir ||
      preferred_cache_dir == kNoCacheDir) {
    ABSL_LOG(WARNING)
        << "RuntimeDebugger disabled: No valid cache directory provided "
        << "for tensor capturing (got '" << preferred_cache_dir << "').";
    return nullptr;
  }
  std::string capture_dir =
      (std::filesystem::path{std::string(preferred_cache_dir)} /
       std::string(kDebuggerSubdir))
          .string();
  return std::make_unique<RuntimeDebugger>(capture_dir);
}

RuntimeDebugger::RuntimeDebugger(absl::string_view capture_dir) {
  std::filesystem::path p{std::string(capture_dir)};
  if (p.filename().empty() && p != p.root_path()) {
    p = p.parent_path();
  }
  capture_dir_ = p.string();
}

LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
RuntimeDebugger::CreatePreGraphRunCallback() {
  return [this](absl::string_view signature_name, int current_step,
                absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
                    outputs) {
    CaptureTensors(MonitorPhase::kPre, signature_name, current_step, outputs);
  };
}

LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
RuntimeDebugger::CreatePostGraphRunCallback() {
  return [this](absl::string_view signature_name, int current_step,
                absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
                    outputs) {
    CaptureTensors(MonitorPhase::kPost, signature_name, current_step, outputs);
  };
}

void RuntimeDebugger::CaptureTensors(
    MonitorPhase phase, absl::string_view signature_name,
    const int current_step,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        tensors) {
  std::string target_dir;

  {
    absl::MutexLock lock(mutex_);
    DebugSessionState* session =
        active_debug_session_.load(std::memory_order_acquire);
    if (session != nullptr) {
      target_dir = session->capture_dir;
      session->last_step = current_step;
      if (session->kv_buffers.empty()) {
        for (const auto& [name, buffer] : tensors) {
          if (IsKVCacheTensor(name)) {
            if (auto dup_or = buffer.Duplicate(); dup_or.HasValue()) {
              session->kv_buffers.emplace(std::string(name),
                                          std::move(*dup_or));
            }
          }
        }
      }
    }
  }

  // If the session was unregistered, do not capture any more data.
  if (target_dir.empty()) {
    ABSL_LOG(WARNING)
        << "No session context found for tensor capture. The session may have "
           "been unregistered.";
    return;
  }

  const bool is_prefill = IsPrefillSignature(signature_name);
  const absl::string_view event_prefix =
      (phase == MonitorPhase::kPre) ? kPreEventPrefix : kPostEventPrefix;

  for (const auto& [name, buffer] : tensors) {
    if (!is_prefill && IsKVCacheTensor(name)) {
      // Only log kv cache at the prefill stage.
      continue;
    }

    const absl::Status status = SaveTensorBufferToFile(
        target_dir, signature_name, name, event_prefix, current_step, buffer);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Tensor capture failed for " << name << ": " << status;
    }
  }
}

std::string RuntimeDebugger::GetSessionCaptureDir(int session_id) {
  return absl::StrCat(kDebuggerSubdir, "/", session_id);
}

void RuntimeDebugger::RegisterDebugSession(
    const int session_id, absl::string_view preferred_cache_dir) {
  const absl::string_view base_dir =
      preferred_cache_dir.empty() ? capture_dir_ : preferred_cache_dir;
  if (base_dir == kMemoryCacheDir || base_dir == kNoCacheDir) {
    return;
  }
  const std::string session_dir =
      (std::filesystem::path{std::string(base_dir)} /
       std::to_string(session_id))
          .string();
  std::error_code ec;
  std::filesystem::create_directories(session_dir, ec);
  if (ec && !std::filesystem::exists(session_dir)) {
    ABSL_LOG(ERROR) << "Failed to create debug session directory: "
                    << session_dir << ", error: " << ec.message();
    return;
  }

  absl::MutexLock lock(mutex_);
  auto ctx = std::make_unique<DebugSessionState>();
  ctx->capture_dir = std::move(session_dir);
  sessions_[session_id] = std::move(ctx);
}

void RuntimeDebugger::UnregisterDebugSession(const int session_id) {
  absl::MutexLock lock(mutex_);
  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    if (active_debug_session_.load(std::memory_order_relaxed) ==
        it->second.get()) {
      active_debug_session_.store(nullptr, std::memory_order_release);
    }
    sessions_.erase(it);
  }
}

void RuntimeDebugger::SetActiveDebugSession(const int session_id) {
  absl::MutexLock lock(mutex_);
  const auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    active_debug_session_.store(it->second.get(), std::memory_order_release);
  } else {
    active_debug_session_.store(nullptr, std::memory_order_release);
  }
}

void RuntimeDebugger::ObserveTokens(const int session_id,
                                    const Responses& responses) {
  std::string target_dir;
  std::vector<std::pair<std::string, ::litert::TensorBuffer>> kv_to_flush;
  int last_step = 0;

  {
    absl::MutexLock lock(mutex_);
    const auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
      target_dir = it->second->capture_dir;
      // Conditionally trigger an end-of-step KV-Cache serialization flush
      // solely upon encountering terminal inference execution states.
      if ((responses.GetTaskState() == TaskState::kDone ||
           responses.GetTaskState() == TaskState::kMaxNumTokensReached) &&
          !it->second->kv_buffers.empty()) {
        last_step = it->second->last_step;
        kv_to_flush.reserve(it->second->kv_buffers.size());
        // Employ safe deep-copying for KV-cache TensorBuffer references to
        // avoid mutation race-conditions against concurrent compute threads.
        for (auto& [name, buffer] : it->second->kv_buffers) {
          if (auto dup_or = buffer.Duplicate(); dup_or.HasValue()) {
            kv_to_flush.emplace_back(name, std::move(*dup_or));
          }
        }
      }
    }
  }

  // If the session was unregistered, do not capture any more data.
  if (target_dir.empty()) {
    ABSL_LOG(WARNING)
        << "No session context found for token observation. The session may "
           "have been unregistered.";
    return;
  }

  WriteTokensToFile(target_dir, responses);

  // Safely execute disk-IO bound Safetensors serialization outside of the
  // critical MutexLock boundary to mitigate mutex-contention across
  // parallel inference runs.
  if (!kv_to_flush.empty()) {
    for (const auto& [name, buffer] : kv_to_flush) {
      const absl::Status status = SaveTensorBufferToFile(
          target_dir, "decode", name, kPostEventPrefix, last_step, buffer);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to capture final decode KV tensor " << name
                        << ": " << status;
      }
    }
  }
}

void RuntimeDebugger::WriteTokensToFile(absl::string_view target_dir,
                                        const Responses& responses) {
  if (responses.GetTokenIds().empty() || target_dir.empty()) {
    return;
  }
  const std::string capture_path =
      (std::filesystem::path{std::string(target_dir)} /
       std::string(kGeneratedTokensFileName))
          .string();
  std::ofstream token_file(capture_path, std::ios_base::app);
  if (!token_file.is_open()) {
    return;
  }

  nlohmann::ordered_json record;
  record[std::string(kTokenIdsKey)] = responses.GetTokenIds();
  if (!responses.GetTexts().empty()) {
    record[std::string(kTextsKey)] = responses.GetTexts();
  }
  if (!responses.GetScores().empty()) {
    record[std::string(kScoresKey)] = responses.GetScores();
  }

  const std::string json_line = absl::StrCat(record.dump(), "\n");
  token_file.write(json_line.data(), json_line.size());
  token_file.flush();
}

}  // namespace litert::lm
