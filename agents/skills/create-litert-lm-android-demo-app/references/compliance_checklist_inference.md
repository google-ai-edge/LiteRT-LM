# Inference Compliance Checklist

You MUST perform a comprehensive compliance audit against this file before
declaring completion. You MUST use this table to report compliance. If any
checks fail, you must fix the code and fully re-run the audit.

**Audit Status Reset Constraint**: Whenever any change is made to the codebase
after a compliance audit starts, you MUST reset all statuses in the compliance
report to 'Pending' (or empty) to ensure a full, clean re-verification of all
items.

**Evidence Quality Constraint**: For all active checks, the "Evidence" column
**MUST** contain specific code references (file names and line numbers) or exact
command outputs. Vague or descriptive summaries (e.g., "verified", "supported",
"manual scroll") are NOT acceptable as evidence.

<!-- mdformat off -->
**Status:** Pass / Fail / Skipped
**Evidence:** Command output / Code snippet

| Requirement                                                  | Status | Evidence |
| :----------------------------------------------------------- | :----- | :------- |
| **Setup & Build**                                            |        |          |
| GPU OpenCL Permissions (`AndroidManifest.xml`)               |        |          |
| APK Verification (`zipinfo`)                                 |        |          |
| **Inference API & Optimization**                             |        |          |
| Data Classes vs Builders Used                                |        |          |
| Explicit `initialize()` Called                               |        |          |
| Conversation Class Used (not Session)                        |        |          |
| MTP Enabled for GPU                                          |        |          |
| **Multi-modal API**                                          |        |          |
| Initialize Fallback Try-Catch Block                          |        |          |
| Multi-modal Executors (CPU Audio)                            |        |          |
| Content Order (Text before Media)                            |        |          |
| Real Inference Used                                          |        |          |
<!-- mdformat on -->

### Detailed Verification Checklist

To complete the compliance report above, verify the following UI and
implementation details strictly:

**Setup & Build**

-   [ ] **GPU Support**: Verify that
    <!-- linter off -->
    `<uses-native-library android:name="libOpenCL.so" android:required="false" />`
    <!-- linter on -->
    is in the `AndroidManifest.xml` `<application>` tag for GPU support.
-   [ ] **APK Verification**: Always run
    `zipinfo bazel-bin/<app_name>.apk 'lib/*'` to verify `.so` files are
    correctly located in the compiled APK.

**Inference API & Optimization**

-   [ ] **Data Classes**: Verify that `EngineConfig`, `SessionConfig`, `ConversationConfig`, and `SamplerConfig` use direct constructors with named arguments, NOT `.builder()` patterns.
-   [ ] **Explicit Initialization**: Verify that `engine.initialize()` is explicitly called before creating conversations or sessions.
-   [ ] **Conversation Routing**: Verify that `sendMessageAsync` is called on a `Conversation` instance created via `engine.createConversation()`.

**Multi-modal API**

-   [ ] **Initialize Fallback Try-Catch Block**: Verify that the engine
    initialization code implements a 3-stage cascading try-catch block:
    1. **Try Multi-modal on Selected Backend**: Attempts to configure and
       initialize a multi-modal engine on the selected general backend
       (e.g., GPU), setting CPU-locked audio and unified vision backends.
    2. **Fallback to Multi-modal on CPU**: If Step 1 throws an exception,
       catches it, attempts to initialize the multi-modal engine strictly on a
       CPU-only configuration (CPU general, CPU vision, CPU audio), and
       programmatically updates/toggles the UI Backend Selector to "CPU".
    3. **Fallback to Text-only Engine (CPU-locked)**: If Step 2 also throws an
       exception, catches it, falls back to an `EngineConfig` strictly
       configured on the CPU backend (`backend = Backend.CPU()`) completely
       omitting `visionBackend` and `audioBackend`, and programmatically
       updates/keeps the Backend Selector displayed as "CPU" in the UI.
-   [ ] **Multi-modal Executors**: Verify that `visionBackend` and
    `audioBackend` are explicitly initialized in `EngineConfig`, and that
    `audioBackend` is strictly set to `Backend.CPU()`.
-   [ ] **Multi-modal Backend**:
    -   [ ] Verify that selected media data are wrapped in `Content` objects and
        passed to the inference call.
    -   [ ] **Explicit Argument Check**: Verify that `sendMessageAsync` (or
        equivalent) actually includes the constructed list of `Content` objects
        (text + media), and is not just passing a single text string.
    -   [ ] **Content Order Check**: Ensure that `Content.Text` is added to the
        contents list **BEFORE** any media content to match model expectations.
-   [ ] **Real Inference**: Verify that the app performs real inference using
    the LiteRT-LM engine and does not use mock responses.
