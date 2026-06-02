# Inference Implementation Guidelines

This document provides explicit instructions for implementing LiteRT-LM
inference logic. You MUST strictly adhere to these rules.

## 1. GPU Permissions (Android)

To prevent Adreno GPU crashes on Android, add the following to
`AndroidManifest.xml`:
<!-- linter off -->
`<uses-native-library android:name="libOpenCL.so" android:required="false"/>`
<!-- linter on -->

## 2. Engine Initialization & Strategy

When implementing engine start-up, you MUST structure your Kotlin block
chronologically inside a background worker following this exact structure from
outermost scope to innermost call:

### A. Background Threading (The Outermost Block)

*   **Background Execution Required**: You MUST execute all native engine creation/initialization and file copying operations on a background thread (e.g., `withContext(Dispatchers.IO)`).
*   *(Note: High-level JNI messaging like `sendMessageAsync` handles threading internally and does NOT need to be wrapped in a background Dispatchers block).*

### B. Configuration Syntax (Parameters)

*   **Named Constructor Arguments**: `EngineConfig`, `SessionConfig`, `ConversationConfig`, and `SamplerConfig` are Kotlin data classes. You MUST NOT use `.builder()` builders; instead, instantiate constructors directly using named parameters (e.g., `EngineConfig(modelPath = path, backend = Backend.GPU())`).
*   **Backend Instantiation**: `Backend.CPU` and `Backend.GPU` are classes/data classes. You MUST explicitly instantiate them (e.g., `backend = Backend.CPU()`).

### C. The Cascading Fallback Strategy (Core Try-Catch Sequence)

You MUST attempt engine configuration and start-up using this robust 3-stage
cascading fallback pattern wrapped in try-catch blocks:
1.  **Try Multi-modal on Selected Backend**: Create `EngineConfig` with
    `visionBackend` set matching the general `backend` (e.g., `Backend.GPU()` or
    `Backend.CPU()`) and `audioBackend` set strictly to `Backend.CPU()`. Call
    `engine.initialize()`.
2.  **Fallback to Multi-modal on CPU**: If Step 1 throws an exception (due to
    GPU runtime incompatibilities or NDK issues), catch it, log the warning, and
    immediately attempt to initialize the multi-modal engine strictly on the
    **CPU** backend (re-creating `EngineConfig` with general
    `backend = Backend.CPU()`, `visionBackend = Backend.CPU()`, and
    `audioBackend = Backend.CPU()`). Call `engine.initialize()`. You MUST also
    programmatically toggle the Backend Selector in the UI to 'CPU' to correctly
    represent the active execution backend.
3.  **Fallback to Text-only Engine (CPU-locked)**: If Step 2 also throws an
    exception (due to model capabilities or configuration limits), catch it, log
    a critical alert, and attempt to initialize a
    **text-only engine strictly on the CPU backend** (creating `EngineConfig`
    with `modelPath` and `backend = Backend.CPU()`, completely omitting
    `visionBackend` and `audioBackend`). Call `engine.initialize()`. You MUST
    also ensure the Backend Selector in the UI is programmatically toggled/kept
    to 'CPU' to correctly represent the active execution backend. If this final
    fallback also fails, bubble the error up to the UI.

*   **UI State Machine Mapping**: Upon completing the cascading initialization,
    you MUST map the resulting engine state directly to the UI Loaded states
    defined in `ui_layout_and_state.md`:
    - **If Step 1 or Step 2 Succeeds**: Transition the UI into the **Multi-modal Model** state (fully enabling Image, Audio, Text input, and Send).
    - **If Step 3 Succeeds**: Transition the UI into the **Text-only Model** state (enabling only Text input and Send, keeping the "Image" and "Audio" buttons strictly disabled).

### D. Explicit Initialization Requirements

*   After creating any `Engine` instance, you MUST explicitly call`engine.initialize()` inside the configurations before attempting to create conversations or sessions.

## 3. Conversation vs Session Usage

*   **API Routing**: The high-level `sendMessageAsync` method is located in the
    `Conversation` class, not `Session`. You MUST create a conversation using
    `engine.createConversation()` and call `sendMessageAsync` on the resulting
    object.
    *   *Example*: `val conversation = engine.createConversation()`

## 4. Multi-modal Support

*   **Compile Full Pipelines**: You MUST implement complete multi-modal
    pipelines in the codebase (including the UI layouts for image/audio picker
    panels, media loading handlers, and Content list parsing modules).
*   **No Build-time Model Shortcuts**: Even if the model selected at build time
    is text-only, you are strictly forbidden from omitting or modifying these
    paths. All multimedia views and API integrations MUST compile successfully.
*   **Dynamic Run-time Fallback**: The compiled application is a versatile demo
    system. Fallbacks to text-only configurations MUST be handled strictly at
    runtime through the cascading configuration try-catch sequences rather than
    by omitting codebase structures.

## 5. Multi-modal Input Assembly

*   **Explicit Content Passing**: For multi-modal input, you MUST ensure you
    pass a constructed list of `Content` objects (text + media) to
    `sendMessageAsync`, and not just a single text string.
    *   **Content Order Check**: You MUST add `Content.Text` to the contents
        list BEFORE any media content (like `Content.Image` or `Content.Audio`)
        to match model expectations.
*   **Real Inference**: You MUST perform real inference using the LiteRT-LM
    engine. Do NOT use mock responses.
