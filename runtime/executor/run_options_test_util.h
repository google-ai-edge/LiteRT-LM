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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_TEST_UTIL_H_

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_opaque_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/executor/run_options.h"

namespace litert::lm {

// The identifier of the opaque options of CountingRunOptionsFactory(), which
// no accelerator knows, so they don't change the runs.
inline constexpr char kTestRunOptionsIdentifier[] =
    "litert_lm_test_run_options";

// Returns a factory of run options that increments `*num_calls` on every call.
//
// The options carry an opaque payload. The opaque options added to a
// `litert::Options` object are released when the options of its first run are
// destroyed, so reusing the object for another run would be a use-after-free:
// running with these options checks that every run gets its own options. The
// payload is only released by a run, so the options must be used for one.
inline RunOptionsFactory CountingRunOptionsFactory(int* num_calls) {
  return [num_calls]() -> absl::StatusOr<::litert::Options> {
    ++*num_calls;
    ::litert::Options options;
    LITERT_ASSIGN_OR_RETURN(
        ::litert::OpaqueOptions opaque_options,
        ::litert::OpaqueOptions::Create(
            kTestRunOptionsIdentifier, new int(*num_calls),
            [](void* payload) { delete static_cast<int*>(payload); }));
    LITERT_RETURN_IF_ERROR(options.AddOpaqueOptions(std::move(opaque_options)));
    return options;
  };
}

// Returns a factory of run options that fails with `status`.
inline RunOptionsFactory FailingRunOptionsFactory(absl::Status status) {
  return [status]() -> absl::StatusOr<::litert::Options> { return status; };
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_TEST_UTIL_H_
