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
//
// Acknowledgements:
// This implementation was originally authored by @john-rocky and ported
// from the open-source repository: https://github.com/john-rocky/swift-litert-lm/tree/main
//
// Tests that need no `.litertlm` on disk.
//
// `LiteRTLMExecutor.init` only builds a `LazyEngine`; the weights are read on the
// first `respond`. So executor/engine accounting, configuration identity, and
// capability derivation are all observable against a model path that does not
// exist — which makes them safe to run in CI.

#if canImport(FoundationModels) && compiler(>=6.4)

  import FoundationModels
  import LiteRTLM
  import XCTest

  @testable import LiteRTLMFoundationModels

  @available(iOS 27.0, macOS 27.0, *)
  private struct StubTool: FoundationModels.Tool {
    let name = "get_temperature"
    let description = "Get the current temperature for a city."
    @Generable struct Arguments {
      @Guide(description: "The city name")
      var city: String
    }
    func call(arguments: Arguments) async throws -> String { "21°C" }
  }

  @available(iOS 27.0, macOS 27.0, *)
  final class AdapterTests: XCTestCase {
    private static let modelPath = "/tmp/litertlm-tests-nonexistent.litertlm"

    override func setUp() async throws {
      try await super.setUp()
      await LiteRTLanguageModel.releaseCachedEngines()
    }

    override func tearDown() async throws {
      await LiteRTLanguageModel.releaseCachedEngines()
      try await super.tearDown()
    }

    // MARK: - Engine sharing

    /// Foundation Models builds one executor per session — a plain session and a
    /// tool-enabled session over the same model yield two. They must resolve to a
    /// single engine, or the multi-GB weights load twice and the app OOMs.
    func testPlainAndToolSessionsShareOneEngine() throws {
      let model = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      XCTAssertEqual(EngineCache.shared.count, 0)

      let plain = LanguageModelSession(model: model)
      plain.prewarm()
      let tooled = LanguageModelSession(model: model, tools: [StubTool()])
      tooled.prewarm()

      withExtendedLifetime((plain, tooled)) {
        XCTAssertEqual(
          EngineCache.shared.count, 1,
          "two sessions over one model must share a single engine")
      }
    }

    /// Two models over the *same* file but different backends genuinely need
    /// different engines, and must not collide in the cache.
    func testDifferentBackendsDoNotShareAnEngine() throws {
      let cpu = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      let gpu = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))

      let a = LanguageModelSession(model: cpu)
      a.prewarm()
      let b = LanguageModelSession(model: gpu)
      b.prewarm()

      withExtendedLifetime((a, b)) {
        XCTAssertEqual(EngineCache.shared.count, 2)
      }
    }

    /// `visualTokenBudget` is a conversation-level setting, so two models over the
    /// same engine configuration with different budgets share one engine (the
    /// multi-GB weights load once).
    func testDifferentVisualTokenBudgetsShareOneEngine() throws {
      let config = try EngineConfig(modelPath: Self.modelPath, backend: .cpu())
      let small = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 70)
      let large = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 280)

      let a = LanguageModelSession(model: small)
      a.prewarm()
      let b = LanguageModelSession(model: large)
      b.prewarm()

      withExtendedLifetime((a, b)) {
        XCTAssertEqual(EngineCache.shared.count, 1)
      }
    }

    // MARK: - Configuration identity

    /// Regression: `Configuration` once hashed on `modelPath` alone, so a `.gpu`
    /// model silently received a `.cpu` engine.
    func testConfigurationDistinguishesBackend() throws {
      let cpu = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      let gpu = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      XCTAssertNotEqual(cpu, gpu)
    }

    func testConfigurationDistinguishesMaxNumTokens() throws {
      let small = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 512))
      let large = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 4096))
      XCTAssertNotEqual(small, large)
    }

    func testIdenticalConfigurationsCompareEqual() throws {
      let a = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      let b = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      XCTAssertEqual(a, b)
      XCTAssertEqual(a.hashValue, b.hashValue)
    }

    /// Regression: the adapter used to mirror a subset of `EngineConfig`'s fields
    /// into its own `Configuration`, so `cacheDir` / `loraRank` / `audioLoraRank` /
    /// `maxNumImages` were dropped on the way to the engine.
    func testConfigurationCarriesEveryEngineConfigField() throws {
      let engineConfig = try EngineConfig(
        modelPath: Self.modelPath,
        backend: .gpu,
        visionBackend: .gpu,
        audioBackend: .cpu(threadCount: 2),
        maxNumTokens: 4096,
        cacheDir: "/tmp/litertlm-tests-cache",
        loraRank: 8,
        audioLoraRank: 4)

      let carried = LiteRTLanguageModel(engineConfig: engineConfig)
        .executorConfiguration.engineConfig

      XCTAssertEqual(carried, engineConfig)
      XCTAssertEqual(carried.cacheDir, "/tmp/litertlm-tests-cache")
      XCTAssertEqual(carried.loraRank, 8)
      XCTAssertEqual(carried.audioLoraRank, 4)
      XCTAssertEqual(carried.maxNumTokens, 4096)
      XCTAssertEqual(carried.audioBackend, .cpu(threadCount: 2))
    }

    /// `init(engineConfig:)` honours the caller's `cacheDir`; the convenience
    /// initializer supplies the app's Caches directory when none is given.
    func testCacheDirIsHonouredAndDefaulted() throws {
      let explicit = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, cacheDir: "/tmp/explicit"))
      XCTAssertEqual(explicit.executorConfiguration.engineConfig.cacheDir, "/tmp/explicit")

      let sugared = try LiteRTLanguageModel(modelPath: Self.modelPath)
      let defaulted = try XCTUnwrap(sugared.executorConfiguration.engineConfig.cacheDir)
      XCTAssertTrue(defaulted.contains("Caches"), "got \(defaulted)")
    }

    // MARK: - Capabilities

    func testVisionCapabilityTracksVisionBackend() throws {
      let textOnly = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath))
      XCTAssertFalse(textOnly.capabilities.contains(.vision))
      XCTAssertTrue(textOnly.capabilities.contains(.guidedGeneration))
      XCTAssertTrue(textOnly.capabilities.contains(.toolCalling))

      let vision = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, visionBackend: .gpu))
      XCTAssertTrue(vision.capabilities.contains(.vision))
    }

    // MARK: - Guided-generation schema encoding

    /// The encoded schema is embedded verbatim in the prompt, so it must be
    /// canonical: dictionary key order is randomized per process, and without
    /// `.sortedKeys` the prompt — and occasionally the model's behavior — varied
    /// across identical runs.
    func testSchemaEncodingIsDeterministicAndSorted() throws {
      let first = try LiteRTLMExecutor.encodeSchema(SortedKeysProbe.generationSchema)
      for _ in 0..<32 {
        XCTAssertEqual(try LiteRTLMExecutor.encodeSchema(SortedKeysProbe.generationSchema), first)
      }
      let apple = try XCTUnwrap(first.range(of: "\"apple\""))
      let mango = try XCTUnwrap(first.range(of: "\"mango\""))
      let zebra = try XCTUnwrap(first.range(of: "\"zebra\""))
      XCTAssertLessThan(apple.lowerBound, mango.lowerBound)
      XCTAssertLessThan(mango.lowerBound, zebra.lowerBound)
    }

    // MARK: - Tool list encoding

    /// The chat template `tojson`s each entry verbatim into the prompt, so the
    /// string this produces is exactly what the model reads. `.bare` is LFM2's
    /// trained format: no OpenAI envelope, `name` first, schema keys in trained
    /// order, none of FM's bookkeeping left anywhere in it.
    func testBareToolListMatchesTheTrainedFormat() throws {
      let json = LiteRTLMExecutor.toolsJson(try Self.toolDefinitions(), style: .bare)

      XCTAssertFalse(json.contains("\"function\""))
      XCTAssertFalse(json.contains("\"title\""))
      XCTAssertFalse(json.contains("x-order"))
      XCTAssertFalse(json.contains("additionalProperties"))
      XCTAssertTrue(
        json.hasPrefix("[{\"name\": \"get_temperature\", \"description\": "),
        "entries must open with the tool's name, not with whatever key sorts first: \(json)")
      let type = try XCTUnwrap(json.range(of: "{\"type\": \"object\""))
      let properties = try XCTUnwrap(json.range(of: "\"properties\"", range: type.lowerBound..<json.endIndex))
      let required = try XCTUnwrap(json.range(of: "\"required\"", range: type.lowerBound..<json.endIndex))
      XCTAssertLessThan(type.lowerBound, properties.lowerBound)
      XCTAssertLessThan(properties.lowerBound, required.lowerBound)

      // Still one valid JSON array of one object per tool.
      let parsed = try XCTUnwrap(
        try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
      XCTAssertEqual(parsed.count, 2)
      XCTAssertEqual(parsed.map { $0["name"] as? String }, ["get_temperature", "open_url"])
    }

    /// The default stays OpenAI-shaped for models (Qwen, Gemma) trained on the
    /// envelope — but FM's bookkeeping is stripped there too.
    func testDefaultToolListKeepsTheEnvelope() throws {
      let json = LiteRTLMExecutor.toolsJson(try Self.toolDefinitions())
      let parsed = try XCTUnwrap(
        try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
      XCTAssertEqual(parsed.count, 2)
      for entry in parsed {
        XCTAssertEqual(entry["type"] as? String, "function")
        XCTAssertNotNil(entry["function"])
      }
      XCTAssertFalse(json.contains("\"title\""))
      XCTAssertFalse(json.contains("additionalProperties"))
    }

    /// Embedded in the prompt, so it must not vary run to run — dictionary key
    /// order is randomized per process.
    func testToolListEncodingIsDeterministic() throws {
      let tools = try Self.toolDefinitions()
      let first = LiteRTLMExecutor.toolsJson(tools, style: .bare)
      for _ in 0..<32 {
        XCTAssertEqual(LiteRTLMExecutor.toolsJson(tools, style: .bare), first)
      }
    }

    /// A tool without arguments is a name and a description — an empty
    /// `parameters` object invites the model to invent some.
    func testToolWithoutArgumentsCarriesNoParameters() throws {
      let json = LiteRTLMExecutor.toolsJson(
        [Transcript.ToolDefinition(tool: NoArgumentsTool())], style: .bare)
      XCTAssertFalse(json.contains("\"parameters\""))
      XCTAssertFalse(json.contains("\"required\""))
    }

    // MARK: - Thinking leak

    /// The budget force-closes `</think>` mid-thought; the model keeps
    /// reasoning into the visible stream and closes again itself. Only what
    /// follows the last closer is the answer.
    func testVisibleAnswerDropsLeakedThought() {
      XCTAssertEqual(
        LiteRTLMExecutor.visibleAnswer(
          "I am near CAFE LA. Let me answer as requested.</think>You are near CAFE LA."),
        "You are near CAFE LA.")
      XCTAssertEqual(
        LiteRTLMExecutor.visibleAnswer("first</think>middle</think> final answer "),
        "final answer")
    }

    func testVisibleAnswerKeepsPlainRepliesAndEmptiesThoughtOnly() {
      XCTAssertEqual(
        LiteRTLMExecutor.visibleAnswer("  You are in Chuo, Osaka. "), "You are in Chuo, Osaka.")
      XCTAssertEqual(LiteRTLMExecutor.visibleAnswer("all of this was reasoning</think>"), "")
      XCTAssertEqual(LiteRTLMExecutor.visibleAnswer("<think>still open, never closed"),
        "still open, never closed")
    }

    // MARK: - Native tool calls

    /// The model writes its calls Python-style. `True` is a boolean, not the
    /// string "True" (device run 2026-09-04: `set_torch(on=True)` reached FM
    /// as a string and the call failed), and a doubled closer must not leak
    /// into the last value.
    func testNativeToolCallParsesPythonBooleansAndStrayClosers() throws {
      let call = try XCTUnwrap(
        LiteRTLMExecutor.parseNativeToolCall(
          "<|tool_call_start|>[set_torch(on=True)]<|tool_call_end|>"))
      XCTAssertEqual(call.name, "set_torch")
      XCTAssertEqual(call.arguments, #"{"on":true}"#)

      let doubled = try XCTUnwrap(
        LiteRTLMExecutor.parseNativeToolCall(
          "<|tool_call_start|>[set_torch(on=False))]<|tool_call_end|>"))
      XCTAssertEqual(doubled.arguments, #"{"on":false}"#)

      let mixed = try XCTUnwrap(
        LiteRTLMExecutor.parseNativeToolCall(
          "<|tool_call_start|>[translate(source='a, b)', to=\"ja\", n=2)]<|tool_call_end|>"))
      let data = try XCTUnwrap(mixed.arguments.data(using: .utf8))
      let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
      XCTAssertEqual(object["source"] as? String, "a, b)")
      XCTAssertEqual(object["to"] as? String, "ja")
      XCTAssertEqual(object["n"] as? Int, 2)
    }

    // MARK: - Guided generation

    /// The mode is a conversation-level choice, so two models over one engine
    /// configuration in different modes share the engine.
    func testGuidedGenerationModesShareOneEngine() throws {
      let config = try EngineConfig(modelPath: Self.modelPath, backend: .cpu())
      let constrained = LiteRTLanguageModel(engineConfig: config)
      let promptOnly = LiteRTLanguageModel(engineConfig: config, guidedGeneration: .promptOnly)
      XCTAssertEqual(constrained.guidedGeneration, .constrained)
      XCTAssertEqual(promptOnly.guidedGeneration, .promptOnly)

      let a = LanguageModelSession(model: constrained)
      a.prewarm()
      let b = LanguageModelSession(model: promptOnly)
      b.prewarm()

      withExtendedLifetime((a, b)) {
        XCTAssertEqual(EngineCache.shared.count, 1)
      }
    }

    /// The schema goes into the trigger prompt and nowhere else. When the
    /// caller keeps it out of the prompt (`includeSchemaInPrompt: false`) the
    /// trigger is the user's words alone, whatever the engine enforces.
    func testPlanWritesTheSchemaIntoTheTriggerOnly() throws {
      let schema = #"{"type":"object"}"#
      let transcript = Transcript(entries: [
        .instructions(
          Transcript.Instructions(
            segments: [.text(.init(content: "Be brief."))], toolDefinitions: [])),
        .prompt(Transcript.Prompt(segments: [.text(.init(content: "Hi"))])),
        .prompt(Transcript.Prompt(segments: [.text(.init(content: "List the colors"))])),
      ])

      let hinted = try LiteRTLMExecutor.plan(
        from: transcript, schemaJSON: schema, guided: true, tools: [])
      XCTAssertEqual(hinted.systemText, "Be brief.")
      XCTAssertTrue(hinted.prompt.toString.hasPrefix("List the colors"))
      XCTAssertTrue(hinted.prompt.toString.contains(schema))
      XCTAssertEqual(hinted.history.map(\.toString), ["Hi"])
      XCTAssertFalse(hinted.respondingToTool)

      let bare = try LiteRTLMExecutor.plan(
        from: transcript, schemaJSON: nil, guided: true, tools: [])
      XCTAssertEqual(bare.prompt.toString, "List the colors")
    }

    /// On the turn that answers a tool result, a guided request asks for the
    /// structure; the plain turn asks for a sentence and forbids JSON.
    func testPlanToolResultTriggerFollowsTheTurnKind() throws {
      let tools = try Self.toolDefinitions()
      let transcript = Transcript(entries: [
        .prompt(Transcript.Prompt(segments: [.text(.init(content: "Weather?"))])),
        .toolOutput(
          Transcript.ToolOutput(
            id: "1", toolName: "get_temperature", segments: [.text(.init(content: "21°C"))])),
      ])

      let plain = try LiteRTLMExecutor.plan(from: transcript, schemaJSON: nil, tools: tools)
      XCTAssertTrue(plain.respondingToTool)
      XCTAssertEqual(plain.toolResult, "21°C")
      XCTAssertEqual(plain.toolRounds, 1)
      XCTAssertTrue(plain.prompt.toString.contains("Do not output JSON"))

      let schema = #"{"type":"object"}"#
      let guided = try LiteRTLMExecutor.plan(
        from: transcript, schemaJSON: schema, guided: true, tools: tools)
      XCTAssertTrue(guided.respondingToTool)
      XCTAssertFalse(guided.prompt.toString.contains("Do not output JSON"))
      XCTAssertTrue(guided.prompt.toString.contains("Tool \"get_temperature\" returned: 21°C"))
      XCTAssertTrue(guided.prompt.toString.contains("Using that result, respond with ONLY"))
      XCTAssertTrue(guided.prompt.toString.contains(schema))
    }

    /// Under the grammar the prompt names only the keys, in declared order:
    /// the engine holds the types and the enum values, and every further word
    /// in the hint was something the model copied into a value.
    func testSchemaHintUnderTheConstraintNamesOnlyTheKeys() {
      let hint = LiteRTLMExecutor.schemaHint(Self.orderSchema, constrained: true)
      XCTAssertEqual(
        hint,
        "Respond with ONLY a JSON object with exactly these keys, in this order: "
          + "item, size, quantity, extras. Output valid JSON and nothing else.")
    }

    /// A schema as Foundation Models encodes one — `x-order`, `title`,
    /// descriptions, an enum, an optional — rendered for the prompt: one line
    /// per field in declared order, and none of the schema's own vocabulary
    /// for the model to echo.
    func testSchemaHintListsFieldsInDeclaredOrder() throws {
      let hint = LiteRTLMExecutor.schemaHint(Self.orderSchema)
      let item = try XCTUnwrap(hint.range(of: "- item (string): The drink ordered"))
      let size = try XCTUnwrap(hint.range(of: "- size (string, one of: small, medium, large)"))
      let quantity = try XCTUnwrap(hint.range(of: "- quantity (integer)"))
      let extras = try XCTUnwrap(hint.range(of: "- extras (array of string, optional)"))
      XCTAssertLessThan(item.lowerBound, size.lowerBound)
      XCTAssertLessThan(size.lowerBound, quantity.lowerBound)
      XCTAssertLessThan(quantity.lowerBound, extras.lowerBound)
      XCTAssertFalse(hint.contains("x-order"))
      XCTAssertFalse(hint.contains("CoffeeOrder"))
      XCTAssertTrue(hint.hasSuffix("Output valid JSON and nothing else."))
    }

    /// A schema with no properties — a bare type — is shown as it is.
    func testSchemaHintFallsBackToTheRawSchemaWithoutProperties() {
      let hint = LiteRTLMExecutor.schemaHint(#"{"type":"string"}"#)
      XCTAssertTrue(hint.contains(#"{"type":"string"}"#))
      XCTAssertTrue(hint.hasSuffix("Output valid JSON and nothing else."))
    }

    /// The engine sees the properties in `x-order` — the order the model
    /// fills them in under the grammar — with the bookkeeping keys gone, and
    /// the same text every time.
    func testEngineSchemaFollowsDeclaredOrder() throws {
      let engine = LiteRTLMExecutor.engineSchema(Self.orderSchema)
      let item = try XCTUnwrap(engine.range(of: "\"item\""))
      let size = try XCTUnwrap(engine.range(of: "\"size\""))
      let quantity = try XCTUnwrap(engine.range(of: "\"quantity\""))
      let extras = try XCTUnwrap(engine.range(of: "\"extras\""))
      XCTAssertLessThan(item.lowerBound, size.lowerBound)
      XCTAssertLessThan(size.lowerBound, quantity.lowerBound)
      XCTAssertLessThan(quantity.lowerBound, extras.lowerBound)
      XCTAssertFalse(engine.contains("x-order"))
      XCTAssertFalse(engine.contains("\"title\""))
      XCTAssertTrue(engine.contains("\"required\":[\"item\",\"size\",\"quantity\"]"))
      for _ in 0..<16 {
        XCTAssertEqual(LiteRTLMExecutor.engineSchema(Self.orderSchema), engine)
      }
      XCTAssertNoThrow(try ResponseFormat.json(schema: engine))
    }

    /// Sorted-keys encoding of a four-field schema, as `encodeSchema` emits it:
    /// alphabetical, with `x-order` carrying the declared order.
    private static let orderSchema = #"""
      {"additionalProperties":false,"properties":{"extras":{"description":"Extras asked for","items":{"type":"string"},"type":"array"},"item":{"description":"The drink ordered","type":"string"},"quantity":{"description":"How many cups","type":"integer"},"size":{"description":"Cup size","enum":["small","medium","large"],"type":"string"}},"required":["item","size","quantity"],"title":"CoffeeOrder","type":"object","x-order":["item","size","quantity","extras"]}
      """#

    /// `Transcript.ToolDefinition` values as FM itself builds them, taken off a
    /// session rather than constructed by hand.
    private static func toolDefinitions() throws -> [Transcript.ToolDefinition] {
      [
        Transcript.ToolDefinition(tool: StubTool()),
        Transcript.ToolDefinition(tool: OpenURLTool()),
      ]
    }
  }

  @available(iOS 27.0, macOS 27.0, *)
  private struct OpenURLTool: FoundationModels.Tool {
    let name = "open_url"
    let description = "Open a web page."
    @Generable struct Arguments {
      @Guide(description: "The URL to open")
      var url: String
    }
    func call(arguments: Arguments) async throws -> String { "opened" }
  }

  @available(iOS 27.0, macOS 27.0, *)
  private struct NoArgumentsTool: FoundationModels.Tool {
    let name = "get_time"
    let description = "The current time."
    @Generable struct Arguments {}
    func call(arguments: Arguments) async throws -> String { "12:00" }
  }

  /// Properties intentionally declared out of alphabetical order, so the sorted
  /// positions asserted in `testSchemaEncodingIsDeterministicAndSorted` can only
  /// come from canonical encoding.
  @available(iOS 27.0, macOS 27.0, *)
  @Generable
  private struct SortedKeysProbe {
    @Guide(description: "zebra") var zebra: String
    @Guide(description: "apple") var apple: String
    @Guide(description: "mango") var mango: String
  }

#endif
