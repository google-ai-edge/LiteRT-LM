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

// This file checks that all public C API headers compile together as C. On
// non-Windows platforms the BUILD target compiles it with
// `-std=c99 -pedantic-errors -Wstrict-prototypes -Werror`, the oldest language
// mode the headers support.
#ifdef __cplusplus
#error "This file must be compiled as C, not C++."
#endif

// Include every public header in the litert_lm/c directory. Nothing here uses
// them; including them is the point of the test.
#include "c/api_export.h"  // IWYU pragma: keep
#include "c/conversation.h"  // IWYU pragma: keep
#include "c/embedding_engine.h"  // IWYU pragma: keep
#include "c/engine.h"  // IWYU pragma: keep
#include "c/error_reporter.h"  // IWYU pragma: keep
#include "c/experimental.h"  // IWYU pragma: keep
#include "c/model_info.h"  // IWYU pragma: keep

int main(void) { return 0; }
