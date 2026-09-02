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

// LiteRT-LM as an Apple Foundation Models backend.
//
// Acknowledgements:
// This implementation was originally authored by @john-rocky and ported
// from the open-source repository: https://github.com/john-rocky/swift-litert-lm/tree/main
//
// `LiteRTLanguageModel` conforms to the iOS 27 `LanguageModel` protocol, so a
// LiteRT-LM model can drive a stock `LanguageModelSession` — alongside Apple's
// own conformers `SystemLanguageModel` (on-device) and
// `PrivateCloudComputeLanguageModel`:
//
//   let cfg     = try EngineConfig(modelPath: path, backend: .gpu)
//   let model   = LiteRTLanguageModel(engineConfig: cfg)
//   let session = LanguageModelSession(model: model)        // Apple's exact API
//   let answer  = try await session.respond(to: "Hi")        // streaming / tools / @Generable
//
// The FM API is transcript-based (each turn hands the executor the full
// conversation); LiteRT-LM is stateful (a `Conversation` accumulates its own KV
// cache). We bridge by rebuilding a fresh LiteRT `Conversation` from the
// transcript on each turn — correct and simple; an incremental fast-path is a
// later optimization.
//
// Depends on only the LiteRT-LM core Swift API (`LiteRTLM`) and `FoundationModels`.

#if canImport(FoundationModels) && compiler(>=6.4)

  import Foundation
  import FoundationModels
  import LiteRTLM
  import os

  private let logger = Logger()

  // MARK: - Model

  /// How the tool list is written into the conversation config — and therefore
  /// into the prompt: the bundle's chat template `tojson`s each entry verbatim,
  /// so this string is exactly what the model reads, character for character.
  @available(iOS 27.0, macOS 27.0, *)
  public enum ToolListStyle: Sendable, Hashable {
    /// `{"type": "function", "function": {…}}` per tool — the OpenAI envelope
    /// most instruct models (Qwen, Gemma) saw in training. The default.
    case openAIFunctions
    /// `{"name", "description", "parameters"}` per tool — the LFM2 family's
    /// trained format. The envelope is ~30 prompt characters per tool that
    /// these models never saw.
    case bare
  }

  /// How a `respond(generating:)` turn is held to its schema.
  @available(iOS 27.0, macOS 27.0, *)
  public enum GuidedGeneration: Sendable, Hashable {
    /// The engine constrains decoding to the schema (LiteRT-LM `ResponseFormat`,
    /// llguidance): every token is checked against the grammar, so the output
    /// cannot deviate from it. The schema is also written into the prompt
    /// unless the caller's `ContextOptions.includeSchemaInPrompt` is false.
    /// The default.
    case constrained
    /// The schema is only written into the prompt (again subject to
    /// `includeSchemaInPrompt`); the output is whatever the model writes. For a
    /// runtime built without constrained decoding, and for measuring what the
    /// constraint buys.
    case promptOnly
  }

  /// A LiteRT-LM model exposed as an Apple Foundation Models backend.
  @available(iOS 27.0, macOS 27.0, *)
  public struct LiteRTLanguageModel: LanguageModel {
    public typealias Executor = LiteRTLMExecutor

    public let capabilities: LanguageModelCapabilities
    public let executorConfiguration: LiteRTLMExecutor.Configuration
    public let visualTokenBudget: Int32?
    public let toolListStyle: ToolListStyle
    /// Whether a schema is enforced by the engine or only asked for in the
    /// prompt. See `GuidedGeneration`.
    public let guidedGeneration: GuidedGeneration
    /// Cap on invisible reasoning, in tokens. Hybrid-thinking bundles (LFM2.5)
    /// declare a `<think>…</think>` channel in their metadata, and the runtime
    /// diverts everything inside it away from the stream. Without a budget
    /// nothing forces the block closed: a turn where the model decides to think
    /// generates unseen at decode speed — 25–40 s on a phone — and can end with
    /// an empty message, which Foundation Models reports as "Session ended
    /// without producing a response". The budget forces `</think>` after this
    /// many thinking tokens. Zero or negative disables the cap.
    public let thinkingTokenBudget: Int

    /// Build from an `EngineConfig` (the primary initializer). The whole config is
    /// carried through to the engine verbatim — including `cacheDir`, `loraRank`,
    /// and `audioLoraRank`.
    ///
    /// - Parameters:
    ///   - engineConfig: How to build the LiteRT engine.
    ///   - visualTokenBudget: Per-image visual-token cap (an `ExperimentalFlags`
    ///     value, not part of `EngineConfig`); nil = engine default.
    ///   - toolListStyle: How tool definitions are written into the prompt;
    ///     pick to match what the bundle's model saw in training.
    ///   - thinkingTokenBudget: Max invisible reasoning tokens per turn before
    ///     `</think>` is forced; ≤ 0 leaves thinking unbounded.
    ///   - guidedGeneration: Whether a schema is enforced by the engine
    ///     (`.constrained`, the default) or only written into the prompt.
    public init(
      engineConfig: EngineConfig, visualTokenBudget: Int32? = nil,
      toolListStyle: ToolListStyle = .openAIFunctions,
      thinkingTokenBudget: Int = 256,
      guidedGeneration: GuidedGeneration = .constrained
    ) {
      self.visualTokenBudget = visualTokenBudget
      self.toolListStyle = toolListStyle
      self.thinkingTokenBudget = thinkingTokenBudget
      self.guidedGeneration = guidedGeneration
      self.executorConfiguration = LiteRTLMExecutor.Configuration(
        engineConfig: engineConfig)
      let capabilities: [LanguageModelCapabilities.Capability] = {
        var caps: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling]
        if engineConfig.visionBackend != nil { caps.append(.vision) }
        return caps
      }()
      self.capabilities = LanguageModelCapabilities(capabilities)
    }

    /// Build from a model path and explicit settings (sugar over `init(engineConfig:)`).
    ///
    /// Unlike `init(engineConfig:)`, this initializer supplies a `cacheDir` default:
    /// the app's Caches directory, which is writable on every Apple platform.
    ///
    /// - Parameters:
    ///   - modelPath: Absolute path to an on-disk `.litertlm`.
    ///   - backend: Main compute backend (default `.gpu`).
    ///   - visionBackend / audioBackend: Backend per encoder tower, or nil to leave
    ///     that tower off (the safe default for a text-only model).
    ///   - visualTokenBudget: Per-image visual-token cap (nil = engine default).
    ///   - maxTokens: KV/context budget (nil = model/engine default).
    ///   - cacheDir: Where the engine writes its cache files (nil = the app's Caches
    ///     directory).
    ///   - toolListStyle: How tool definitions are written into the prompt;
    ///     pick to match what the bundle's model saw in training.
    ///   - thinkingTokenBudget: Max invisible reasoning tokens per turn before
    ///     `</think>` is forced; ≤ 0 leaves thinking unbounded.
    ///   - guidedGeneration: Whether a schema is enforced by the engine
    ///     (`.constrained`, the default) or only written into the prompt.
    /// - Throws: `LiteRTLMError` if `maxTokens` is less than or equal to 0.
    public init(
      modelPath: String,
      backend: Backend = .gpu,
      visionBackend: Backend? = nil,
      audioBackend: Backend? = nil,
      visualTokenBudget: Int32? = nil,
      maxTokens: Int? = 2048,
      cacheDir: String? = nil,
      toolListStyle: ToolListStyle = .openAIFunctions,
      thinkingTokenBudget: Int = 256,
      guidedGeneration: GuidedGeneration = .constrained
    ) throws {
      let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
      self.init(
        engineConfig: try EngineConfig(
          modelPath: modelPath, backend: backend,
          visionBackend: visionBackend, audioBackend: audioBackend,
          maxNumTokens: maxTokens, cacheDir: cacheDir ?? caches?.path),
        visualTokenBudget: visualTokenBudget,
        toolListStyle: toolListStyle,
        thinkingTokenBudget: thinkingTokenBudget,
        guidedGeneration: guidedGeneration)
    }

    /// Release every cached LiteRT engine built for FM sessions, freeing their
    /// (multi-GB) weights. Any live `LanguageModelSession` over this backend
    /// rebuilds its engine on the next turn.
    public static func releaseCachedEngines() async {
      await EngineCache.shared.purgeAll()
    }
  }

  // MARK: - Executor

  /// Drives generation for `LiteRTLanguageModel` over the FM executor protocol.
  @available(iOS 27.0, macOS 27.0, *)
  public final class LiteRTLMExecutor: LanguageModelExecutor {
    public typealias Model = LiteRTLanguageModel

    /// What engine to build. FM requires this to be a `Hashable` value, so it
    /// cannot hold a live `Engine`; the engine is built lazily and shared across
    /// every executor whose configuration compares equal.
    ///
    /// Wrapping `EngineConfig` whole (rather than mirroring a subset of its fields)
    /// means every engine setting flows through unchanged, and equality covers all
    /// of them.
    public struct Configuration: Hashable, Sendable {
      public let engineConfig: EngineConfig

      public var modelPath: String { engineConfig.modelPath }

      public init(engineConfig: EngineConfig) {
        self.engineConfig = engineConfig
      }
    }

    /// How many user/tool exchanges before the current one are replayed.
    static let historyExchanges = 1

    /// What a cached conversation was built for.
    enum ConversationKind: String {
      /// No tool menu; accepts a response format. Plain chat and guided turns.
      case open
      /// A tool session's conversation, tools in the preface.
      case tooled
    }

    private let engine: LazyEngine
    private let cache = ConversationCache()

    public init(configuration: Configuration) throws {
      // Share one engine per configuration across executors. FM builds a new
      // executor per session (a plain session and a tool-enabled session over the
      // same model yield two executors), and each engine loads multi-GB weights —
      // without sharing, the second session OOMs the app.
      self.engine = EngineCache.shared.engine(for: configuration)
    }

    public func prewarm(model: Model, transcript: Transcript) {
      Task { try? await engine.prewarmed() }
    }

    public func respond(
      to request: LanguageModelExecutorGenerationRequest,
      model: Model,
      streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
      let engine = try await self.engine.ready()
      // Guided generation: a request that carries a schema is a structured-
      // output contract. The schema is written into the prompt unless the
      // caller's `ContextOptions.includeSchemaInPrompt` says not to (FM's own
      // knob, for a schema the instructions already describe), and under
      // `.constrained` the engine also constrains decoding to it
      // (`ResponseFormat`, llguidance), so the output cannot deviate from it.
      // No tool menu on such a turn: the structure is the answer.
      //
      // Tools are different: they go to the engine as definitions and the model
      // calls them in its own trained format, which this executor parses back
      // (`parseNativeToolCall`).
      let tools = request.enabledToolDefinitions
      let schemaJSON = try request.schema.map { try Self.encodeSchema($0) }
      let guided = schemaJSON != nil
      let schemaInPrompt = request.contextOptions.includeSchemaInPrompt ?? true
      let constrained = guided && model.guidedGeneration == .constrained
      let plan = try Self.plan(
        from: request.transcript, schemaJSON: schemaInPrompt ? schemaJSON : nil,
        guided: guided, constrained: constrained, tools: tools)
      let responseFormat =
        try constrained
        ? schemaJSON.map { try ResponseFormat.json(schema: Self.engineSchema($0)) } : nil
      let structured = guided || !tools.isEmpty
      let offerTools = !tools.isEmpty && !plan.respondingToTool && !guided

      // LFM2.5 has its own tool-call format — `<|tool_call_start|>[name(args)]
      // <|tool_call_end|>` — and LiteRT-LM parses it. The JSON router this used
      // to impose fought that: under a grammar for the router's shape the model
      // could not emit the tokens it wanted and produced nothing at all, and
      // without one it emitted its native call, which the router could not read.
      // So the tools go to the engine as tool definitions, and the model answers
      // in the form it was trained for.
      // Reused across turns. Building one costs the tokenisation and KV setup of
      // the whole prefix — measured at 15-19s a turn on a phone, against 0.4-1.4s
      // of actual generation — and it was being paid twice per beat. Tools stay
      // attached for every turn; the tool-output message already tells the model
      // to answer rather than call again.
      // Keyed by the system text and the conversation's kind. A guided turn
      // needs a conversation that accepts a response format and carries no
      // tool menu; a tool session's conversation is neither, so the two kinds
      // never share one.
      let kind: ConversationKind = guided || tools.isEmpty ? .open : .tooled
      let resumeKey = plan.systemText + "\u{1F}" + kind.rawValue
      var reusable: Conversation?
      if let reused = cache.take(for: resumeKey, consumed: plan.triggerIndex) {
        // A conversation is reused only while its KV has room for another
        // turn. Every fix that makes turns *succeed* also makes the KV grow
        // monotonically — the first fully green demo run died on beat 5 with
        // "remaining capacity: 4". A windowed rebuild costs one prefill;
        // an exhausted conversation costs the beat.
        let capacity = model.executorConfiguration.engineConfig.maxNumTokens ?? 2048
        let used = (try? reused.getTokenCount()) ?? 0
        if used > capacity - Self.turnTokenHeadroom {
          LiteRTFMTrace.timing("cache DROP, kv \(used)/\(capacity)")
        } else {
          reusable = reused
        }
      }
      let conversation: Conversation
      if let reused = reusable {
        LiteRTFMTrace.timing(
          "cache HIT, prompt \(plan.promptText.count) chars, history \(plan.history.count)")
        conversation = reused
      } else {
        LiteRTFMTrace.timing(
          "cache MISS, prompt \(plan.promptText.count) chars, history \(plan.history.count)"
            + ", system \(plan.systemText.count) chars")
        conversation = try await Self.build(
        ConversationConfig(
          systemMessage: plan.systemMessage,
          initialMessages: plan.history,
          samplerConfig: Self.sampler(for: request.generationOptions, structured: structured),
          // Cap invisible reasoning. Never `enableThinking: false` here: the
          // runtime only installs the budget constraint when thinking is
          // enabled — disabling it removes the cap and leaves the model free
          // to think, unseen and unbounded (this template ignores the
          // `enable_thinking` flag).
          thinkingConfig: model.thinkingTokenBudget > 0
            ? ThinkingConfig(enableThinking: true, thinkingTokenBudget: model.thinkingTokenBudget)
            : nil,
          // Off: FM runs the tools and re-invokes this executor with the result.
          // Letting the runtime run them too would execute everything twice.
          automaticToolCalling: false,
          // Not offered on the turn that answers a tool result. FM re-invokes
          // the executor after every tool, and a model still holding the menu
          // orders again: translate was called four times in a row off one OCR
          // result. A chained call the model writes anyway is honored below,
          // behind a per-question round limit.
          toolsJsonOverride: offerTools
            ? Self.toolsJson(tools, style: model.toolListStyle) : nil,
          // On for every open conversation, not only the turn that carries a
          // schema: the conversation is reused across turns, and one built
          // without it refuses a later guided turn (`responseFormatNotEnabled`).
          // Tooled conversations keep it off — with it on, the runtime derives a
          // tool-call grammar from the preface where the model's data processor
          // has one, and the native-format tool path here was measured without.
          enableResponseFormat: kind == .open && model.guidedGeneration == .constrained,
          visualTokenBudget: model.visualTokenBudget),
          on: engine)
      }
      defer { cache.keep(conversation, for: resumeKey, consumed: plan.triggerIndex) }

      if guided {
        // Buffered: FM parses the structure out of the text it is handed, and
        // a half-written object is not one. Under the constraint the thinking
        // budget is switched off for this turn: the runtime wraps the caller's
        // constraint in its thinking-budget constraint, and on a hybrid bundle
        // that pairing failed every guided turn (llguidance panic at token 0);
        // the grammar itself keeps `<think>` out, so nothing is lost.
        var full = ""
        let sent = Date()
        var firstToken: Date?
        for try await chunk in conversation.sendMessageStream(
          plan.prompt,
          thinkingConfig: constrained
            ? ThinkingConfig(enableThinking: false, thinkingTokenBudget: 0) : nil,
          responseFormat: responseFormat)
        {
          let piece = chunk.toString
          if firstToken == nil, !piece.isEmpty { firstToken = Date() }
          full += piece
          LiteRTFMTrace.emit(piece)
        }
        let done = Date()
        LiteRTFMTrace.timing(
          "guided turn: ttft \(String(format: "%.1f", (firstToken ?? done).timeIntervalSince(sent)))s"
            + ", decode \(String(format: "%.1f", done.timeIntervalSince(firstToken ?? done)))s"
            + ", \(full.count) chars, \(constrained ? "constrained" : "prompt only")"
            + ", schema in prompt \(schemaInPrompt)")
        // Under the constraint the text is the object. Without one the model
        // may wrap it in prose or a code fence; the first balanced object is
        // taken.
        let json = Self.extractJSONObject(from: full) ?? full
        await channel.send(.response(action: .appendText(json, tokenCount: json.count)))
      } else if offerTools {
        LiteRTFMTrace.emit("\u{00B7} \(tools.count) tools offered\n")
        // Streamed, unconstrained. The non-streaming call took the app down on
        // the first turn; and since this bundle's metadata does not declare the
        // tool-call delimiters, the runtime hands the markers through as text
        // either way — so they are read here.
        var replyText = ""
        var thought = ""
        let sent = Date()
        var firstToken: Date?
        for try await chunk in conversation.sendMessageStream(plan.prompt) {
          let piece = chunk.toString
          if firstToken == nil, !piece.isEmpty { firstToken = Date() }
          replyText += piece
          thought += chunk.channels.values.joined()
          LiteRTFMTrace.emit(piece)
        }
        let done = Date()
        if !thought.isEmpty { LiteRTFMTrace.emit("\n[think] \(thought)\n") }
        LiteRTFMTrace.timing(
          "call turn: ttft \(String(format: "%.1f", (firstToken ?? done).timeIntervalSince(sent)))s"
            + ", decode \(String(format: "%.1f", done.timeIntervalSince(firstToken ?? done)))s"
            + ", \(replyText.count) chars, \(thought.count) thought")
        // The visible text first; a model that spent its whole turn inside the
        // think block sometimes leaves the call — or the answer — in there.
        let native = Self.parseNativeToolCall(replyText) ?? Self.parseNativeToolCall(thought)
        if let call = native {
          let arguments = call.arguments
          LiteRTFMTrace.emit("\(call.name) \(arguments)\n")
          await channel.send(
            .toolCalls(
              action: .toolCall(
                id: UUID().uuidString, name: call.name,
                action: .appendArguments(arguments, tokenCount: arguments.count))))
        } else {
          var answer = Self.visibleAnswer(replyText)
          if answer.isEmpty {
            answer = thought.trimmingCharacters(in: .whitespacesAndNewlines)
          }
          await channel.send(.response(action: .appendText(answer, tokenCount: answer.count)))
        }
      } else if !tools.isEmpty {
        // The turn that answers a tool result. Buffered rather than streamed:
        // the model sometimes answers with another call — "Open CAFE LA in
        // Apple Maps" routed through `search_places` first, then `open_in_maps`
        // — and a call has to be forwarded whole, not scrolled onto the screen
        // as text, which is what a streamed answer turn did to that beat.
        var replyText = ""
        var thought = ""
        let sent = Date()
        var firstToken: Date?
        for try await chunk in conversation.sendMessageStream(plan.prompt) {
          let piece = chunk.toString
          if firstToken == nil, !piece.isEmpty { firstToken = Date() }
          replyText += piece
          thought += chunk.channels.values.joined()
          LiteRTFMTrace.emit(piece)
        }
        let done = Date()
        if !thought.isEmpty { LiteRTFMTrace.emit("\n[think] \(thought)\n") }
        LiteRTFMTrace.timing(
          "answer turn: ttft \(String(format: "%.1f", (firstToken ?? done).timeIntervalSince(sent)))s"
            + ", decode \(String(format: "%.1f", done.timeIntervalSince(firstToken ?? done)))s"
            + ", \(replyText.count) chars, \(thought.count) thought, round \(plan.toolRounds)")
        let native = Self.parseNativeToolCall(replyText) ?? Self.parseNativeToolCall(thought)
        if let call = native, plan.toolRounds < Self.maxToolRoundsPerQuestion {
          LiteRTFMTrace.emit("\(call.name) \(call.arguments)\n")
          await channel.send(
            .toolCalls(
              action: .toolCall(
                id: UUID().uuidString, name: call.name,
                action: .appendArguments(call.arguments, tokenCount: call.arguments.count))))
        } else {
          // A reply that is only call markers (rounds exhausted) or nothing at
          // all (the model never left its think block) falls back to the tool's
          // own words. Something is always sent: an executor that returns
          // without sending is an FM error — "Session ended without producing a
          // response" — that poisons the transcript for every beat after it.
          var answer = Self.visibleAnswer(replyText)
          if native != nil || answer.isEmpty {
            answer = plan.toolResult ?? answer
          }
          await channel.send(.response(action: .appendText(answer, tokenCount: answer.count)))
        }
      } else {
        // Plain chat, no tools anywhere: stream as it generates.
        let sent = Date()
        var firstToken: Date?
        var total = 0
        for try await chunk in conversation.sendMessageStream(plan.prompt) {
          let delta = chunk.toString
          if !delta.isEmpty {
            if firstToken == nil { firstToken = Date() }
            total += delta.count
            LiteRTFMTrace.emit(delta)
            await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
          }
        }
        let done = Date()
        LiteRTFMTrace.timing(
          "chat turn: ttft \(String(format: "%.1f", (firstToken ?? done).timeIntervalSince(sent)))s"
            + ", decode \(String(format: "%.1f", done.timeIntervalSince(firstToken ?? done)))s"
            + ", \(total) chars")
        if total == 0 {
          await channel.send(.response(action: .appendText("", tokenCount: 0)))
        }
      }
    }

    /// Extract the first balanced JSON object from model text (strips prose/fences).
    private static func extractJSONObject(from text: String) -> String? {
      guard let start = text.firstIndex(of: "{") else { return nil }
      var depth = 0
      var inString = false
      var escaped = false
      var idx = start
      while idx < text.endIndex {
        let ch = text[idx]
        if inString {
          if escaped {
            escaped = false
          } else if ch == "\\" {
            escaped = true
          } else if ch == "\"" {
            inString = false
          }
        } else if ch == "\"" {
          inString = true
        } else if ch == "{" {
          depth += 1
        } else if ch == "}" {
          depth -= 1
          if depth == 0 { return String(text[start...idx]) }
        }
        idx = text.index(after: idx)
      }
      return nil
    }

    // MARK: Transcript → LiteRT messages

    struct Plan {
      let systemMessage: Message?
      let history: [Message]
      let prompt: Message
      /// True when the turn was triggered by a tool's output rather than by the
      /// user. FM re-invokes the executor after a tool returns so the model can
      /// answer *from* the result — offering it the tool menu again is how a
      /// small model ends up calling the same tool forever.
      let respondingToTool: Bool
      /// The system message as text, the cache key's first half.
      let systemText: String
      /// Where the trigger sits. A conversation may resume only if it has
      /// already been fed every input before the trigger.
      let triggerIndex: Int
      /// The trigger's text, for the timing trace.
      let promptText: String
      /// The trimmed result text when the trigger is a tool's output — the
      /// fallback answer for a turn whose generation comes back empty.
      let toolResult: String?
      /// How many tools have already run for the current user question. Bounds
      /// chained calls: one follow-up is what "open the shop you found" needs,
      /// unbounded is translate called four times in a row.
      let toolRounds: Int
    }

    /// Chained tool calls allowed per user question before the model is made
    /// to answer with what it has.
    static let maxToolRoundsPerQuestion = 2

    /// KV room a turn needs: its prompt, its generation, and the thinking
    /// budget. A cached conversation closer to the ceiling than this is
    /// dropped and rebuilt from the windowed transcript.
    static let turnTokenHeadroom = 384

    /// Split the FM transcript into a system message, prior turns (history), and
    /// the message to generate from. The generation trigger is the last `.prompt`
    /// OR (in a tool round-trip) the last `.toolOutput`.
    ///
    /// `schemaJSON` is the schema to write into the trigger, nil when none is
    /// wanted there; `guided` says whether the turn expects a structure at all,
    /// a separate question when the caller keeps the schema out of the prompt;
    /// `constrained` says the engine will hold the output to the schema, which
    /// changes how much of it the prompt has to carry.
    static func plan(
      from transcript: Transcript, schemaJSON: String?, guided: Bool = false,
      constrained: Bool = false, tools: [Transcript.ToolDefinition]
    ) throws -> Plan {
      let entries = Array(transcript)
      guard
        let triggerIndex = entries.lastIndex(where: {
          switch $0 {
          case .prompt, .toolOutput: return true
          default: return false
          }
        })
      else {
        throw LiteRTFMError.noPrompt
      }

      let triggeredByTool: Bool
      if case .toolOutput = entries[triggerIndex] { triggeredByTool = true } else {
        triggeredByTool = false
      }

      var triggerText = ""
      var triggerToolResult: String?
      // Tool outputs since the question being answered: every entry after the
      // last user prompt at or before the trigger.
      let lastPromptIndex =
        entries[...triggerIndex].lastIndex(where: {
          if case .prompt = $0 { return true } else { return false }
        }) ?? entries.startIndex
      let toolRounds = entries[lastPromptIndex...triggerIndex].filter {
        if case .toolOutput = $0 { return true } else { return false }
      }.count
      var systemText: [String] = []
      // Only the exchanges near the trigger are sent. A `LanguageModelSession`
      // keeps the whole conversation, which is right for the session and wrong
      // for a phone: by the sixth turn the prefill is mostly history and the
      // model returns nothing at all. Two exchanges is enough for "read that
      // out loud" to know what "that" is.
      let inputPositions = entries.indices.filter { i in
        switch entries[i] {
        case .prompt, .toolOutput: return true
        default: return false
        }
      }
      let keepFrom = inputPositions.suffix(Self.historyExchanges + 1).first ?? 0
      // No tool protocol in the system message any more. The engine is given the
      // tool definitions and the model answers in its own trained format; a
      // second, contradictory convention in the prompt is what made it reply
      // {"tool": "schedule_notification", …} instead of calling anything.
      var history: [Message] = []
      var trigger: Message?

      for (i, entry) in entries.enumerated() {
        let isTrigger = (i == triggerIndex)
        // Instructions always; everything else only inside the window.
        if !isTrigger && i < keepFrom {
          if case .instructions = entry {} else { continue }
        }
        switch entry {
        case .instructions(let instructions):
          systemText.append(text(of: instructions.segments))
        case .prompt(let p):
          var c = contents(of: p.segments)
          if isTrigger, let schemaJSON, !schemaJSON.isEmpty {
            c.append(.text("\n\n" + schemaHint(schemaJSON, constrained: constrained)))
          }
          let message = Message(contents: c, role: .user)
          if isTrigger {
            trigger = message
            triggerText = text(of: p.segments)
          } else {
            history.append(message)
          }
        case .response(let r):
          history.append(Message(contents: [.text(text(of: r.segments))], role: .model))
        case .toolOutput(let output):
          // Trimmed before it goes back to the model. A tool may legitimately
          // return a lot — an OCR pass returned a whole café menu — and every
          // turn after that one carries it in the history until the context is
          // gone and generations come back empty. The host still shows the full
          // text; the model only needs enough to answer from.
          var result = text(of: output.segments)
          if result.count > 240 {
            result = String(result.prefix(240)) + "…"
          }
          // "Do not call another tool" is a bias, not a rule: the executor
          // honors a chained call the model writes anyway, up to
          // `maxToolRoundsPerQuestion`. A guided turn asks for the structure
          // instead — "do not output JSON" would be the one wrong thing to say.
          let instruction: String
          if isTrigger && guided {
            let hint =
              schemaJSON.map { schemaHint($0, constrained: constrained) }
              ?? "Respond with ONLY a JSON object. Output valid JSON and nothing else."
            instruction = "Using that result, " + hint.prefix(1).lowercased() + hint.dropFirst()
          } else {
            instruction =
              "Answer the user in one short sentence using that result. "
              + "Do not output JSON and do not call another tool."
          }
          let message = Message(
            "Tool \"\(output.toolName)\" returned: \(result)\n" + instruction, role: .user)
          if isTrigger {
            trigger = message
            triggerToolResult = result
          } else {
            history.append(message)
          }
        case .toolCalls(let calls):
          // Written back in the model's own call format. A prose placeholder
          // here — "[the assistant called a tool]" — was being imitated: shown
          // that a call looks like a sentence about calling, the model wrote
          // sentences about calling instead of calling.
          let rendered = calls.map { call -> String in
            let arguments = Self.pythonicArguments(call.arguments)
            return "<|tool_call_start|>[\(call.toolName)(\(arguments))]<|tool_call_end|>"
          }.joined()
          history.append(Message(rendered, role: .model))
        case .reasoning:
          break
        @unknown default:
          break
        }
      }

      let system = systemText.joined(separator: "\n").trimmingCharacters(
        in: .whitespacesAndNewlines)
      return Plan(
        systemMessage: system.isEmpty ? nil : Message(system, role: .system),
        history: history,
        prompt: trigger!,  // guaranteed by triggerIndex
        respondingToTool: triggeredByTool,
        systemText: system,
        triggerIndex: triggerIndex,
        promptText: triggerText,
        toolResult: triggerToolResult,
        toolRounds: toolRounds
      )
    }

    /// `<|tool_call_start|>[name(key=value, key='text')]<|tool_call_end|>` —
    /// LFM2.5's own call format, as it comes back when the runtime has not
    /// split it out. The value syntax is Python-ish: bare numbers and booleans,
    /// single or double quoted strings.
    static func parseNativeToolCall(_ text: String) -> (name: String, arguments: String)? {
      guard let open = text.range(of: "<|tool_call_start|>"),
        let close = text.range(of: "<|tool_call_end|>", range: open.upperBound..<text.endIndex)
      else { return nil }
      var body = String(text[open.upperBound..<close.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if body.hasPrefix("[") { body.removeFirst() }
      if body.hasSuffix("]") { body.removeLast() }
      guard let parenthesis = body.firstIndex(of: "(") , body.hasSuffix(")") else {
        let name = body.trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? nil : (name, "{}")
      }
      let name = String(body[body.startIndex..<parenthesis]).trimmingCharacters(in: .whitespaces)
      let inside = String(body[body.index(after: parenthesis)..<body.index(before: body.endIndex)])
      var arguments: [String: Any] = [:]
      for pair in splitTopLevel(inside) {
        guard let equals = pair.firstIndex(of: "=") else { continue }
        let key = String(pair[pair.startIndex..<equals]).trimmingCharacters(in: .whitespaces)
        var value = String(pair[pair.index(after: equals)...]).trimmingCharacters(in: .whitespaces)
        if (value.hasPrefix("'") && value.hasSuffix("'"))
          || (value.hasPrefix("\"") && value.hasSuffix("\"")), value.count >= 2
        {
          value.removeFirst()
          value.removeLast()
          arguments[key] = value
        } else if let number = Int(value) {
          arguments[key] = number
        } else if let number = Double(value) {
          arguments[key] = number
        } else if value == "true" || value == "false" {
          arguments[key] = value == "true"
        } else {
          arguments[key] = value
        }
      }
      return (name, argumentsJSON(arguments))
    }

    /// Split on commas that are not inside quotes, so a quoted value may contain
    /// one.
    private static func splitTopLevel(_ text: String) -> [String] {
      var out: [String] = []
      var current = ""
      var quote: Character?
      for character in text {
        if let open = quote {
          current.append(character)
          if character == open { quote = nil }
        } else if character == "'" || character == "\"" {
          quote = character
          current.append(character)
        } else if character == "," {
          out.append(current)
          current = ""
        } else {
          current.append(character)
        }
      }
      if !current.trimmingCharacters(in: .whitespaces).isEmpty { out.append(current) }
      return out
    }

    /// Arguments back in the form the model writes them: `key='text', n=5`.
    static func pythonicArguments(_ content: GeneratedContent) -> String {
      guard let data = String(describing: content).data(using: .utf8),
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
      else { return "" }
      return object.keys.sorted().compactMap { key -> String? in
        switch object[key] {
        case let value as String: return "\(key)='\(value)'"
        case let value as Bool: return "\(key)=\(value)"
        case let value as NSNumber: return "\(key)=\(value)"
        default: return nil
        }
      }.joined(separator: ", ")
    }

    static func build(_ config: ConversationConfig, on engine: Engine) async throws
      -> Conversation
    {
      try await engine.createConversation(with: config)
    }

    /// The part of a reply that is meant for the user. The thinking budget
    /// force-closes `</think>` after N tokens, but the model does not know
    /// that: it keeps reasoning into the visible stream and closes the block
    /// again itself — so text before the *last* closer is thought that leaked,
    /// not answer ("…short answer as requested.</think>You are near CAFE LA…").
    static func visibleAnswer(_ text: String) -> String {
      let tail = text.range(of: "</think>", options: .backwards).map { text[$0.upperBound...] }
      var answer = String(tail ?? text[...])
      if answer.hasPrefix("<think>") { answer.removeFirst("<think>".count) }
      return answer.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// FM's tool definitions as the string the model will read. The runtime
    /// parses this as a JSON array and hands each entry to the chat template
    /// verbatim (`tojson`), so every character here is prompt tokens — on the
    /// demo phone the tool list alone was 84% of each turn's prefill.
    static func toolsJson(
      _ tools: [Transcript.ToolDefinition], style: ToolListStyle = .openAIFunctions
    ) -> String {
      let entries: [String] = tools.map { tool in
        var function: [String: Any] = ["name": tool.name, "description": tool.description]
        if let encoded = try? encodeSchema(tool.parameters),
          let data = encoded.data(using: .utf8),
          let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let schema = prunedSchema(raw) as? [String: Any],
          let properties = schema["properties"] as? [String: Any], !properties.isEmpty
        {
          function["parameters"] = schema
        }
        switch style {
        case .openAIFunctions:
          return canonicalJSON(["type": "function", "function": function])
        case .bare:
          return canonicalJSON(function)
        }
      }
      return "[" + entries.joined(separator: ", ") + "]"
    }

    /// Strip what the model cannot use: FM's `title`/`x-order` bookkeeping (at
    /// every level, not just the top), `additionalProperties`, and empty
    /// `required` arrays — none of which appear in any model's trained tool
    /// format.
    private static func prunedSchema(_ value: Any) -> Any {
      if let array = value as? [Any] { return array.map(prunedSchema) }
      guard var object = value as? [String: Any] else { return value }
      object.removeValue(forKey: "title")
      object.removeValue(forKey: "x-order")
      object.removeValue(forKey: "additionalProperties")
      if (object["required"] as? [Any])?.isEmpty == true {
        object.removeValue(forKey: "required")
      }
      for (key, inner) in object { object[key] = prunedSchema(inner) }
      return object
    }

    /// The key order trained tool formats use: `name` before `description`
    /// before `parameters`; inside a schema, `type` before `properties` before
    /// `required`. Keys not listed sort alphabetically after these.
    private static let toolKeyOrder = [
      "type", "function", "name", "description", "parameters", "properties", "enum", "required",
      "items",
    ]

    /// Serialize with `toolKeyOrder`. `JSONSerialization` randomizes dictionary
    /// order per process, which would make the prompt differ run to run; sorted
    /// keys would put `description` before `name` and `type` last, which no
    /// model saw in training.
    private static func canonicalJSON(_ value: Any) -> String {
      switch value {
      case let object as [String: Any]:
        let keys = object.keys.sorted { a, b in
          let ia = toolKeyOrder.firstIndex(of: a) ?? toolKeyOrder.count
          let ib = toolKeyOrder.firstIndex(of: b) ?? toolKeyOrder.count
          return ia == ib ? a < b : ia < ib
        }
        return "{"
          + keys.map { "\(quotedJSON($0)): \(canonicalJSON(object[$0]!))" }.joined(separator: ", ")
          + "}"
      case let array as [Any]:
        return "[" + array.map(canonicalJSON).joined(separator: ", ") + "]"
      case let string as String:
        return quotedJSON(string)
      case let number as NSNumber:
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
        return "\(number)"
      default:
        return "null"
      }
    }

    private static func quotedJSON(_ string: String) -> String {
      guard let data = try? JSONSerialization.data(withJSONObject: [string]),
        let text = String(data: data, encoding: .utf8)
      else { return "\"\(string)\"" }
      return String(text.dropFirst().dropLast())
    }

    static func argumentsJSON(_ arguments: [String: Any]) -> String {
      guard JSONSerialization.isValidJSONObject(arguments),
        let data = try? JSONSerialization.data(withJSONObject: arguments),
        let json = String(data: data, encoding: .utf8)
      else { return "{}" }
      return json
    }

    private static func text(of segments: [Transcript.Segment]) -> String {
      segments.compactMap { segment in
        if case .text(let t) = segment { return t.content } else { return nil }
      }.joined(separator: " ")
    }

    /// The schema-in-prompt text for the trigger. The prompt carries what the
    /// engine does not: without a grammar, a field list — name, type, enum
    /// values, description — so the model knows the shape; under a grammar,
    /// the key names alone, in order. Never the raw JSON schema: shown that,
    /// a small model echoes it, and under a grammar the echo comes back as a
    /// well-formed object full of the schema's own words ("title",
    /// "PrimaryColors") — and a field list under a grammar did the same in
    /// miniature, `": "` where a value should be. A schema without properties
    /// (a bare type) is shown as it is.
    static func schemaHint(_ schemaJSON: String, constrained: Bool = false) -> String {
      guard let data = schemaJSON.data(using: .utf8),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
        root["properties"] is [String: Any]
      else {
        return "Respond with ONLY a JSON value that conforms to this JSON schema:\n"
          + "\(schemaJSON)\nOutput valid JSON and nothing else."
      }
      if constrained {
        return "Respond with ONLY a JSON object with exactly these keys, in this order: "
          + propertyOrder(root).joined(separator: ", ") + ". Output valid JSON and nothing else."
      }
      var lines = ["Respond with ONLY a JSON object with these fields, in this order:"]
      lines.append(contentsOf: fieldLines(root, indent: "- ") ?? [])
      lines.append("Output valid JSON and nothing else.")
      return lines.joined(separator: "\n")
    }

    /// One line per property — `- name (type[, one of: …][, optional]):
    /// description` — in Foundation Models' declared order, nested objects
    /// indented beneath their field.
    private static func fieldLines(_ schema: [String: Any], indent: String) -> [String]? {
      guard let properties = schema["properties"] as? [String: Any] else { return nil }
      let required = Set(schema["required"] as? [String] ?? [])
      var lines: [String] = []
      for name in propertyOrder(schema) {
        guard let property = properties[name] as? [String: Any] else { continue }
        var kind = typeName(property)
        if let values = property["enum"] as? [Any] {
          kind += ", one of: " + values.map { "\($0)" }.joined(separator: ", ")
        }
        if !required.contains(name) { kind += ", optional" }
        var line = "\(indent)\(name) (\(kind))"
        if let description = property["description"] as? String, !description.isEmpty {
          line += ": \(description)"
        }
        lines.append(line)
        let nested =
          (property["type"] as? String) == "array" ? property["items"] as? [String: Any] : property
        if let nested, let inner = fieldLines(nested, indent: "  " + indent) {
          lines.append(contentsOf: inner)
        }
      }
      return lines
    }

    private static func typeName(_ property: [String: Any]) -> String {
      if let type = property["type"] as? String {
        if type == "array", let items = property["items"] as? [String: Any] {
          return "array of \(typeName(items))"
        }
        return type
      }
      return property["anyOf"] != nil ? "one of the listed forms" : "value"
    }

    /// Property names in generation order: FM's `x-order` first, anything it
    /// does not name alphabetically after.
    private static func propertyOrder(_ schema: [String: Any]) -> [String] {
      let properties = schema["properties"] as? [String: Any] ?? [:]
      let declared = (schema["x-order"] as? [String] ?? []).filter { properties[$0] != nil }
      return declared + properties.keys.filter { !declared.contains($0) }.sorted()
    }

    /// The schema as the engine's grammar compiler should see it: properties
    /// in Foundation Models' declared generation order (`x-order`), which the
    /// sorted-keys encoding loses and llguidance would replace with
    /// alphabetical — the model was then made to fill `extras` before it had
    /// read what was ordered. `title` and `x-order` are dropped on the way:
    /// they are FM bookkeeping, not schema. Everything else stays sorted, so
    /// the text is still the same run to run.
    static func engineSchema(_ schemaJSON: String) -> String {
      guard let data = schemaJSON.data(using: .utf8),
        let root = try? JSONSerialization.jsonObject(with: data)
      else { return schemaJSON }
      return orderedJSON(root)
    }

    private static func orderedJSON(_ value: Any) -> String {
      switch value {
      case let object as [String: Any]:
        let keys = object.keys.filter { $0 != "title" && $0 != "x-order" }.sorted()
        let members = keys.map { key -> String in
          if key == "properties", let properties = object[key] as? [String: Any] {
            let body = propertyOrder(object).map {
              "\(quotedJSON($0)):\(orderedJSON(properties[$0]!))"
            }.joined(separator: ",")
            return "\(quotedJSON(key)):{\(body)}"
          }
          return "\(quotedJSON(key)):\(orderedJSON(object[key]!))"
        }
        return "{" + members.joined(separator: ",") + "}"
      case let array as [Any]:
        return "[" + array.map(orderedJSON).joined(separator: ",") + "]"
      case let string as String:
        return quotedJSON(string)
      case let number as NSNumber:
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
        return "\(number)"
      default:
        return "null"
      }
    }

    /// Encode a schema to canonical JSON. `.sortedKeys` matters: dictionary key
    /// order is randomized per process, so without it the schema text differs
    /// between runs — and since the schema is embedded in the prompt, the prompt
    /// itself would vary run to run.
    static func encodeSchema(_ schema: GenerationSchema) throws -> String {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.sortedKeys]
      let data = try encoder.encode(schema)
      return String(data: data, encoding: .utf8) ?? ""
    }

    /// Translate the caller's FM `GenerationOptions` into a LiteRT `SamplerConfig`,
    /// so `temperature` / `.greedy` / `.random(top:)` / `.random(probabilityThreshold:)`
    /// are honored instead of overridden. Structured output (guided / tools) is
    /// parsed as JSON, so it's forced near-deterministic regardless.
    private static func sampler(for options: GenerationOptions, structured: Bool) -> SamplerConfig?
    {
      if structured { return try? SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0) }

      var topK = 40
      var topP: Float = 0.95
      var temperature = Float(options.temperature ?? 0.8)

      if let kind = options.samplingMode?.kind {
        switch kind {
        case .greedy:
          topK = 1
          temperature = 0.0
        case .randomTopK(let k, _):
          topK = k
        case .randomProbabilityThreshold(let threshold, _):
          topP = Float(threshold)
        @unknown default:
          break
        }
      }
      return try? SamplerConfig(topK: topK, topP: topP, temperature: temperature)
    }

    /// Map FM segments to LiteRT content: text, image attachments, and audio/video
    /// via the custom segments.
    private static func contents(of segments: [Transcript.Segment]) -> [Content] {
      var out: [Content] = []
      for segment in segments {
        switch segment {
        case .text(let t):
          if !t.content.isEmpty { out.append(.text(t.content)) }
        case .attachment(let attachment):
          guard case .image(let image) = attachment.content else {
            logger.warning(
              "LiteRT-LM: Unsupported attachment type. Only images are supported via .attachment.")
            break
          }
          guard let png = pngData(from: image.cgImage) else {
            logger.warning(
              "LiteRT-LM: Failed to convert CGImage to PNG data. Image attachment ignored.")
            break
          }
          out.append(.imageData(png))
        case .structure:
          break
        @unknown default:
          break
        }
      }
      return out.isEmpty ? [.text("")] : out
    }
  }

  /// Holds a conversation between turns so its prefix is prefilled once.
  ///
  /// Reusable while the prefix is the same string and this turn continues where
  /// the last one stopped. Anything else — a new session, a different tool set,
  /// a branched transcript — misses, and a fresh conversation is built.
  @available(iOS 27.0, macOS 27.0, *)
  final class ConversationCache: @unchecked Sendable {
    private let lock = NSLock()
    private var conversation: Conversation?
    private var key: String?
    private var consumed = -1

    func take(for key: String, consumed trigger: Int) -> Conversation? {
      lock.lock()
      defer { lock.unlock() }
      guard self.key == key, trigger > consumed, let held = conversation else { return nil }
      conversation = nil
      return held
    }

    func keep(_ conversation: Conversation, for key: String, consumed trigger: Int) {
      lock.lock()
      self.conversation = conversation
      self.key = key
      self.consumed = trigger
      lock.unlock()
    }
  }

  /// Raw model output, as it arrives, for a host that wants to show it.
  ///
  /// The tool path cannot stream into the FM channel — a half-written tool call
  /// is not an answer — so without this a caller sees nothing until the whole
  /// turn is done. Debug/demo aid: set it to nil to pay nothing.
  @available(iOS 27.0, macOS 27.0, *)
  public enum LiteRTFMTrace {
    nonisolated(unsafe) public static var onChunk: (@Sendable (String) -> Void)?
    /// Decode tokens per second for the pass that just finished, so a host can
    /// show the real speed rather than a spinner. One stream chunk is one token.
    nonisolated(unsafe) public static var onRate: (@Sendable (Double) -> Void)?

    static func emit(_ piece: String) {
      guard !piece.isEmpty, let onChunk else { return }
      onChunk(piece)
    }

    /// Where a turn's seconds went, for a host that wants to show or log it.
    nonisolated(unsafe) public static var onTiming: (@Sendable (String) -> Void)?

    static func timing(_ line: String) {
      onTiming?(line)
    }

    static func rate(tokens: Int, seconds: Double) {
      guard seconds > 0, tokens > 0, let onRate else { return }
      onRate(Double(tokens) / seconds)
    }
  }

  /// Errors specific to the Foundation Models bridge.
  @available(iOS 27.0, macOS 27.0, *)
  public enum LiteRTFMError: Error, LocalizedError {
    case noPrompt

    public var errorDescription: String? {
      switch self {
      case .noPrompt: return "The transcript contains no prompt to respond to."
      }
    }
  }

  // MARK: - Engine cache + lazy engine

  /// Process-wide cache of one `LazyEngine` per configuration, so multiple FM
  /// executors / sessions sharing a configuration share a single loaded engine.
  @available(iOS 27.0, macOS 27.0, *)
  final class EngineCache: @unchecked Sendable {
    static let shared = EngineCache()
    private let lock = NSLock()
    private var engines: [LiteRTLMExecutor.Configuration: LazyEngine] = [:]

    /// How many distinct engines are currently held. Not part of the public API.
    var count: Int {
      lock.lock()
      defer { lock.unlock() }
      return engines.count
    }

    func engine(for configuration: LiteRTLMExecutor.Configuration) -> LazyEngine {
      lock.lock()
      defer { lock.unlock() }
      if let engine = engines[configuration] { return engine }
      let engine = LazyEngine(configuration: configuration)
      engines[configuration] = engine
      return engine
    }

    func purgeAll() async {
      for engine in drain() { await engine.release() }
    }

    private func drain() -> [LazyEngine] {
      lock.lock()
      defer { lock.unlock() }
      let all = Array(engines.values)
      engines.removeAll()
      return all
    }
  }

  /// Lazily creates and caches the LiteRT engine. The FM executor's `init` is
  /// synchronous but engine initialization is async, so we defer it to the first
  /// `respond` (which is async) and memoize the result.
  @available(iOS 27.0, macOS 27.0, *)
  actor LazyEngine {
    private let configuration: LiteRTLMExecutor.Configuration
    private var engineTask: Task<Engine, Error>?
    private var warmed = false

    init(configuration: LiteRTLMExecutor.Configuration) {
      self.configuration = configuration
    }

    func ready() async throws -> Engine {
      // Memoize the in-flight initialization task as a guard rail against double-loads.
      // Without this, multiple concurrent requests could trigger redundant multi-GB
      // engine initializations.
      if let engineTask { return try await engineTask.value }
      let configuration = self.configuration
      let task = Task {
        let created = Engine(engineConfig: configuration.engineConfig)
        try await created.initialize()
        return created
      }
      engineTask = task
      do {
        return try await task.value
      } catch {
        // A failed initialization stays retryable on the next call.
        if engineTask == task { engineTask = nil }
        throw error
      }
    }

    func prewarmed() async throws {
      let engine = try await ready()
      if warmed { return }
      warmed = true
      let warmup = try await engine.createConversation()
      for try await _ in warmup.sendMessageStream(Message("Hi")) {}
    }

    func release() {
      engineTask = nil
      warmed = false
    }
  }

#endif
