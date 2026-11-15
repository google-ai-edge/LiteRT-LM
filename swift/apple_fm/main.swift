// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Headless macOS functional test for the lean Foundation Models adapter.
//
//   swift run fmtest /path/to/model.litertlm [gpu|cpu]
//
// Drives respond / guided generation / tool calling through a real
// `LanguageModelSession` over the LiteRT backend — no device required.

#if canImport(FoundationModels) && compiler(>=6.4)

  import Foundation
  import FoundationModels
  import LiteRTLM
  import LiteRTLMFoundationModels

  @available(macOS 27.0, *)
  @Generable
  struct PrimaryColors {
    @Guide(description: "Exactly three additive primary colors")
    var colors: [String]
  }

  @available(macOS 27.0, *)
  struct TemperatureTool: FoundationModels.Tool {
    let name = "get_temperature"
    let description = "Get the current temperature for a city."
    @Generable struct Arguments {
      @Guide(description: "The city name")
      var city: String
    }
    func call(arguments: Arguments) async throws -> String {
      "The temperature in \(arguments.city) is 21°C and clear."
    }
  }

  @available(macOS 27.0, *)
  func runTest(modelPath: String, backend: Backend) async {
    print("fmtest: model=\(modelPath) backend=\(backend)")
    do {
      let cfg = try EngineConfig(modelPath: modelPath, backend: backend)
      let model = LiteRTLanguageModel(engineConfig: cfg)

      print("\n== respond(to:) ==")
      let a = try await LanguageModelSession(model: model)
        .respond(to: "Explain on-device AI in one sentence.")
      print(a.content)

      print("\n== respond(generating:) ==")
      let g = try await LanguageModelSession(model: model)
        .respond(generating: PrimaryColors.self) { "List the three additive primary colors." }
      print("colors = \(g.content.colors)")

      print("\n== tool calling ==")
      let t = try await LanguageModelSession(model: model, tools: [TemperatureTool()])
        .respond(to: "What is the temperature in Tokyo right now?")
      print(t.content)

      print("\n== DONE ==")
    } catch {
      print("ERROR: \(error)")
    }
  }

  let args = Array(CommandLine.arguments.dropFirst())
  let modelPath = args.first ?? "/tmp/gemma-4-E2B-it.litertlm"
  let backend: Backend = (args.count > 1 && args[1] == "cpu") ? .cpu() : .gpu

  if #available(macOS 27.0, *) {
    await runTest(modelPath: modelPath, backend: backend)
  } else {
    print("fmtest needs macOS 27.")
  }

#else
  print("FoundationModels is not available in this SDK.")
#endif
