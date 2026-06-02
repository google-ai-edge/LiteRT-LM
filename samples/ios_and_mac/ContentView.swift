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

import LiteRTLM  // Import your package!
import SwiftUI

struct ContentView: View {
  @State private var statusMessage = "Initializing Engine..."
  @State private var prompt = "Explain quantum computing in one sentence."
  @State private var responseText = ""
  @State private var conversation: Conversation?

  @State private var engine: Engine?

  var body: some View {
    VStack {
      Text(statusMessage)
        .font(.headline)
        .padding()

      TextField("Enter your prompt here", text: $prompt)
        .textFieldStyle(.roundedBorder)
        .padding()

      Button("Generate Response") {
        Task {
          await generateText()
        }
      }
      .buttonStyle(.bordered)
      .disabled(conversation == nil)  // Disable until engine is ready

      ScrollView {
        Text(responseText)
          .padding()
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      .frame(maxHeight: .infinity)

      Spacer()
    }
    .padding()
    .task {
      await initializeEngine()
    }
  }

  func initializeEngine() async {
    do {
      // 1. Find the file inside the app bundle!
      guard
        let modelPath = Bundle.main.path(
          forResource: "gemma-4-E2B-it", ofType: "litertlm")
      else {
        statusMessage = "Model file not found in app bundle!"
        return
      }

      // 2. Get the path to the writable Caches directory on iOS
      let fileManager = FileManager.default
      guard let cacheDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
      else {
        fatalError("Could not find caches directory")
      }

      let config = try EngineConfig(
        modelPath: modelPath, backend: .gpu, cacheDir: cacheDirectory.path)
      // let config = try EngineConfig(
      //  modelPath: modelPath, backend: .cpu(), cacheDir: cacheDirectory.path)
      let newEngine = Engine(engineConfig: config)

      // 3. Initialize the native engine!
      try await newEngine.initialize()

      // 4. Create a conversation session
      self.engine = newEngine
      self.conversation = try await newEngine.createConversation()

      statusMessage = "Engine Ready! 🎉"
    } catch {
      statusMessage = "Failed to initialize: \(error.localizedDescription)"
    }
  }

  func generateText() async {
    guard let conversation = conversation else { return }

    responseText = ""  // Clear previous response

    do {
      // 5. Use streaming instead of blocking sendMessage!
      for try await chunk in conversation.sendMessageStream(Message(prompt)) {
        if let firstContent = chunk.contents.first {
          switch firstContent {
          case .text(let text):
            responseText += text  // Append the chunk live!
          default:
            break
          }
        }
      }
    } catch {
      responseText = "Inference error: \(error.localizedDescription)"
    }
  }
}
