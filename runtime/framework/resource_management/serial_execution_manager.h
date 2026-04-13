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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_SERIAL_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_SERIAL_EXECUTION_MANAGER_H_

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <tuple>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/execution_manager_base.h"
#include "runtime/framework/resource_management/resource_manager.h"

namespace litert::lm {

class SerialExecutionManager : public ExecutionManagerBase {
 public:
  static absl::StatusOr<std::unique_ptr<SerialExecutionManager>> Create(
      Tokenizer* absl_nonnull tokenizer,
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutorSettings> absl_nullable
      vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> absl_nullable
      audio_executor_settings,
      ::litert::Environment* absl_nullable litert_env,
      std::unique_ptr<AudioExecutor> absl_nullable audio_executor = nullptr);

  ~SerialExecutionManager() override {
    WaitUntilAllDone(Engine::kDefaultTimeout).IgnoreError();
  };

  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout) override;
  absl::Status WaitUntilSessionDone(SessionId session_id,
                                    absl::Duration timeout) override;
  absl::Status WaitUntilAllDone(absl::Duration timeout) override;

  absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info) override;
  absl::Status ReleaseSession(SessionId session_id) override;
  absl::Status CancelAllTasksInSession(SessionId session_id) override;
  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) override;
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(
      SessionId session_id) override;
  absl::StatusOr<TaskId> GetNewTaskId() override;

  absl::StatusOr<int> GetCurrentStep(const SessionInfo& session_info) override;

  absl::Status SetCurrentStep(const SessionInfo& session_info,
                              int target_step) override;

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return resource_manager_->GetAudioExecutorProperties();
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return resource_manager_->GetVisionExecutorProperties();
  }

 protected:
  absl::Status CreateTask(
      SessionId session_id, TaskId task_id, absl::AnyInvocable<void()> task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::Status QueueTask(TaskId task_id) override;

  absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) override;

  absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  void FinishTaskAndLogErrors(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::Status CloneSession(SessionId session_id,
                            SessionId cloned_session_id) override;

 private:
  SerialExecutionManager(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr)
      : ExecutionManagerBase(tokenizer, std::move(resource_manager),
                             litert_env) {}

  std::deque<TaskId> task_queue_ = {};
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_SERIAL_EXECUTION_MANAGER_H_
