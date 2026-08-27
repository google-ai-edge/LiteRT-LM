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

#include <iostream>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "schema/capabilities/capabilities.h"

using litert::lm::schema::capabilities::InspectModel;
using litert::lm::schema::capabilities::ModelCapabilities;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_litertlm_model>\n";
    return 1;
  }

  std::string model_path = argv[1];
  absl::StatusOr<ModelCapabilities> info_or = InspectModel(model_path);
  if (!info_or.ok()) {
    std::cerr << "Error inspecting model: " << info_or.status().message()
              << "\n";
    return 1;
  }

  const ModelCapabilities& info = *info_or;

  std::cout << "========================================\n";
  std::cout << " LiteRT-LM Model Inspection Report\n";
  std::cout << "========================================\n";
  std::cout << "File: " << model_path << "\n\n";

  std::cout << info;
  std::cout << "========================================\n" << std::flush;

  return 0;
}
