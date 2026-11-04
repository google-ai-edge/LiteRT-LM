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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_CANCELLATION_LATCH_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_CANCELLATION_LATCH_H_

#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert

namespace litert::lm {

// Cancellation of a single cancellable executor call. Allocate it on the stack
// of the call, so that its state can't leak into other calls.
//
// For its lifetime, the LiteRT interpreters of `compiled_models` call
// IsCancelled() after every op. When it is destroyed, their check is replaced
// by one that never cancels, since LiteRT can't unregister a cancellation
// function (it rejects a null one).
//
// Once `should_cancel` returns true, the cancellation is latched: IsCancelled()
// keeps returning true without calling `should_cancel` again, so that a
// one-shot predicate (e.g. one returning true only once) still cancels the
// whole call.
//
// Example:
//   CancellationLatch latch(should_cancel, {&compiled_model});
//   if (latch.IsCancelled()) return absl::CancelledError(...);
//   ... compiled_model.Run(...) ...
class CancellationLatch {
 public:
  // Starts the cancellable call.
  //
  // Args:
  //   should_cancel: Returns true to request the cancellation of the call. Must
  //     outlive the latch.
  //   compiled_models: The compiled models run by the call, whose LiteRT
  //     interpreters check for cancellation after every op. Must outlive the
  //     latch.
  CancellationLatch(absl::FunctionRef<bool()> should_cancel,
                    absl::Span<litert::CompiledModel* const> compiled_models);

  // Ends the cancellable call: the compiled models no longer use the latch.
  ~CancellationLatch();

  CancellationLatch(const CancellationLatch&) = delete;
  CancellationLatch& operator=(const CancellationLatch&) = delete;

  // Checks whether the call is cancelled: calls `should_cancel` until it
  // returns true, then no longer calls it.
  //
  // Returns:
  //   True if `should_cancel` returned true, now or in a previous call.
  bool IsCancelled();

 private:
  absl::FunctionRef<bool()> should_cancel_;
  std::vector<litert::CompiledModel*> compiled_models_;
  bool cancelled_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_CANCELLATION_LATCH_H_
