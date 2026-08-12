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

#include "schema/capabilities/capabilities.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_litertlm_file>\n";
    return 1;
  }
  std::string file_path = argv[1];
  auto info_or = litert::lm::schema::capabilities::InspectModel(file_path);
  if (!info_or.ok()) {
    std::cerr << "Inspection failed: " << info_or.status() << "\n";
    return 1;
  }

  const auto& info = *info_or;
  std::cout << "========================================\n"
            << " LiteRT-LM Model Inspection Report\n"
            << "========================================\n"
            << "File: " << file_path << "\n"
            << "Model Class: "
            << (info.model_class.empty() ? "<none>" : info.model_class) << "\n"
            << "TF Hub ID:   "
            << (info.tf_hub_model_id.empty() ? "<none>" : info.tf_hub_model_id)
            << "\n"
            << "Min Runtime: "
            << (info.min_litertlm_version.empty() ? "<none>"
                                                  : info.min_litertlm_version)
            << "\n";

  if (info.llm_capability.has_value()) {
    const auto& llm = *info.llm_capability;
    std::cout << "\n[LLM Capabilities]\n"
              << "  Max Context Length:     " << llm.max_context_length << "\n"
              << "  Supports Function Call: "
              << (llm.supports_function_calling ? "YES" : "NO") << "\n"
              << "  Supports Thinking:      "
              << (llm.supports_thinking ? "YES" : "NO") << "\n"
              << "  Speculative Decoding:   "
              << (llm.supports_speculative_decoding ? "YES" : "NO") << "\n"
              << "  Input Modalities:       ";
    for (auto m : llm.input_modalities) {
      if (m == litert::lm::schema::capabilities::Modality::kText) {
        std::cout << "Text ";
      }
      if (m == litert::lm::schema::capabilities::Modality::kVision) {
        std::cout << "Vision ";
      }
      if (m == litert::lm::schema::capabilities::Modality::kAudio) {
        std::cout << "Audio ";
      }
    }
    std::cout << "\n  Supported Backends:     ";
    if (llm.supported_backends.empty()) {
      std::cout << "<none>";
    } else {
      for (const auto& b : llm.supported_backends) {
        std::cout << b << " ";
      }
    }
    if (!llm.supported_vision_resolutions.empty()) {
      std::cout << "\n  Vision Resolutions:     ";
      for (auto r : llm.supported_vision_resolutions) {
        std::cout << r << " ";
      }
    }
    std::cout << "\n  Default Sampler:        ";
    if (llm.default_sampler_params.has_value()) {
      std::cout << "temp=" << llm.default_sampler_params->temperature() << ", "
                << "top_k=" << llm.default_sampler_params->k() << ", "
                << "top_p=" << llm.default_sampler_params->p();
    } else {
      std::cout << "<none>";
    }
    std::cout << "\n";
  }
  std::cout << "========================================\n";
  return 0;
}
