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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_H_

#include <functional>
#include <optional>
#include <utility>

#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert

namespace litert::lm {

// Creates the LiteRT options of a model run, e.g. with a vendor NPU
// performance mode. See `SetRunOptions()` of the executors.
//
// The executors take a factory rather than a `litert::Options` object because
// a `litert::Options` object can't be used for more than one run: e.g. the
// opaque options added with
// `litert::Options::AddOpaqueOptions(OpaqueOptions&&)` are released when the
// options of the first run are destroyed. The factory is called for every model
// run, so it should be cheap.
using RunOptionsFactory = std::function<absl::StatusOr<::litert::Options>()>;

// Creates the options of a model run with `run_options_factory`, or returns
// std::nullopt if `run_options_factory` is empty, in which case the run uses
// the options the model was compiled with.
inline absl::StatusOr<std::optional<::litert::Options>> CreateRunOptions(
    const RunOptionsFactory& run_options_factory) {
  if (!run_options_factory) {
    return std::nullopt;
  }
  ABSL_ASSIGN_OR_RETURN(::litert::Options run_options, run_options_factory());
  return std::optional<::litert::Options>(std::move(run_options));
}

// Returns the options to pass to `litert::CompiledModel::Run()`: the created
// `run_options`, or nullptr to use the compile-time options.
inline ::litert::Options* GetRunOptionsPtr(
    std::optional<::litert::Options>& run_options) {
  return run_options.has_value() ? &*run_options : nullptr;
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_RUN_OPTIONS_H_
