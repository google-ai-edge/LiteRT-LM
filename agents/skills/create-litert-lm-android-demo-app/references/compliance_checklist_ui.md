# UI Compliance Checklist

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
| Workspace Directory Name                                     |        |          |
| Strictly Bazel 9 (No Gradle files generated)                 |        |          |
| SDK Version (`minSdkVersion` >= 24)                          |        |          |
| SDK Version (`targetSdkVersion` >= 35)                       |        |          |
| Bzlmod Config (Bazel 9, no WORKSPACE)                        |        |          |
| Bazel Platform Flag Setup (ARM64 Only)                       |        |          |
| AppCompat Avoidance (Extends `Activity`)                     |        |          |
| Bzlmod SDK Configuration Fallback                            |        |          |
| Source Compilation Target                                    |        |          |
| Binary Dependency Verification                               |        |          |
| Edge-to-Edge Handling                                        |        |          |
| **UI State & Flow**                                          |        |          |
| Initial State (inputs disabled except picker)                |        |          |
| Model Picked State                                           |        |          |
| Backend Selected State                                       |        |          |
| Backend Selector Options                                     |        |          |
| Backend Label Alignment                                      |        |          |
| Backend Default Selection                                    |        |          |
| Inference State (disable all during inference)               |        |          |
| Post-Inference State                                         |        |          |
| Loading Animation: File to Cache                             |        |          |
| Loading Animation: Model in Engine                           |        |          |
| Loading Animation: Waiting for Response                      |        |          |
| Shared Animation Helper Used                                 |        |          |
| Active Animation Tracking Verified                           |        |          |
| **Layout & Components**                                      |        |          |
| Model Picker UI                                              |        |          |
| Model Label Alignment                                        |        |          |
| Model Filename Display                                       |        |          |
| Strict Layout Order                                          |        |          |
| Conversation Display Layout Weight                           |        |          |
| Conversation Display Layout                                  |        |          |
| Auto-Scroll                                                  |        |          |
| Markdown Support (Markwon)                                   |        |          |
| Explicit UI Contrast & ForceDark Checked                      |        |          |
| **Multi-modal UI & File Handling**                           |        |          |
| Text-only Fallback Buttons Disabling                         |        |          |
| Image Picker (Storage Select)                                |        |          |
| Audio Picker (Storage Select)                                |        |          |
| Images Rendered Visually                                     |        |          |
| Audio Clearly Indicated (Filename/Icon)                      |        |          |
| Audio Filename Resolution (`OpenableColumns`)                |        |          |
| Background File Operations                                   |        |          |
| File Extension & Caching Preservation                        |        |          |
<!-- mdformat on -->

### Detailed Verification Checklist

To complete the compliance report above, verify the following UI and
implementation details strictly:

**Setup & Build**

-   [ ] **Workspace Directory Name**: Verify that the project directory is named `litert_lm_android_sample_app` and serves as the Bazel workspace root.
-   [ ] **SDK Version (minSdkVersion)**: Verify that `AndroidManifest.xml` specifies `minSdkVersion` (at least 24).
-   [ ] **SDK Version (targetSdkVersion)**: Verify that `AndroidManifest.xml` specifies `targetSdkVersion` (at least 35).
-   [ ] **Bzlmod Config (Bazel 9)**: Verify that the project uses Bzlmod (no
    `WORKSPACE` file) and `MODULE.bazel` handles dependencies according to Bazel
    9 requirements.
-   [ ] **Bazel Platform Flag Setup (ARM64 Only)**: Verify that the platform
    targeting flag `--android_platforms=@rules_android//:arm64-v8a` is
    persistently logged inside `task.md` under the
    **Environment, Tools & Reference Paths** section, and correctly appended to
    all compile executions to enforce ARM64 targets by default.
-   [ ] **AppCompat Avoidance**: Verify that the main activity extends
    `android.app.Activity` directly (not `AppCompatActivity`).
-   [ ] **Bzlmod SDK Configuration Fallback**: If automatic SDK detection fails,
    verify explicit configuration via
    `android_sdk_repository_extension.configure(...)` is present.
-   [ ] **Source Compilation Target**: Verify that the `BUILD` file contains a
    `kt_android_library` (or similar) target compiling the local source files
    (e.g., `MainActivity.kt`).
-   [ ] **Binary Dependency**: Verify that the **Source Compilation Target** is included in the `android_binary` dependencies.
-   [ ] **Edge-to-Edge Handling**: Verify that the app targets API 35+ and
    handles window insets in code to avoid layout overlap with system bars.

**UI State & Flow**

-   [ ] **Initial State**: Verify that all inputs except the model picker are
    disabled on startup.
-   [ ] **Model Picked State**: Verify the Model Picker becomes disabled after a
    model is picked.
-   [ ] **Backend Selected State**: Verify the Backend Selector becomes disabled
    after a backend is selected.
-   [ ] **Backend Selector Options**: Ensure **BOTH** CPU and GPU options are
    available in the Backend Selector.
-   [ ] **Backend Label Alignment**: Verify "Backend:" label is left-aligned.
-   [ ] **Backend Default Selection**: Verify *no option is selected by
    default*.
-   [ ] **Inference State**: Verify all message inputs (prompt, send, image,
    audio) are disabled during inference to prevent concurrent requests.
-   [ ] **Post-Inference State**: Verify the UI transitions back to a state
    where inputs are enabled after inference completes (success or error).
-   [ ] **Loading State Verification**:
    -   [ ] Verify animation is shown in the conversation display when loading
        model file to cache (in background thread).
    -   [ ] Verify animation is shown in the conversation display when loading
        model in Engine.
    -   [ ] Verify animation is shown in the conversation display when waiting
        for conversation response.
-   [ ] **Shared Animation Helper**: Verify that all loading states use a shared
    helper function to ensure UI consistency.
-   [ ] **Active Animation Tracking**: Verify that the animation actively tracks
    the background operation and stops immediately upon completion, rather than
    running for a simulated, hardcoded duration.

**Layout & Components**

-   [ ] **Model Picker UI**: Verify that the "Pick Model" click listener triggers a real system file picking `Intent(Intent.ACTION_GET_CONTENT)` with MIME type `*/*` via `startActivityForResult`.
-   [ ] **Model Label Alignment**: Verify the selected model name label is
    horizontally aligned with the "Pick Model" button.
-   [ ] **Model Filename Display**: Verify it shows the real filename after
    selection.
-   [ ] **Strict Layout Order**: Verify the vertical order strictly matches the
    ASCII specification diagram.
-   [ ] **Conversation Display Layout Weight**: Ensure the conversation
    container uses layout parameters with `weight` to fill available vertical
    space.
-   [ ] **Conversation Display Layout**: Verify it uses a `ScrollView` containing a vertical layout, and uses `adjustViewBounds = true` for images.
-   [ ] **Markdown Support**: Verify it renders Markdown for text messages (e.g., using `io.noties.markwon:core:4.6.2`).
-   [ ] **Auto-Scroll**: Verify the conversation automatically scrolls to the
    bottom when new content arrives. **Evidence MUST cite the line number
    calling `fullScroll` on the ScrollView.**
-   [ ] **Explicit UI Contrast & ForceDark**: Verify that
    `android:forceDarkAllowed="false"` is explicitly declared on the activity
    root layout. Furthermore, verify that contrasting text and background colors
    are explicitly set for all key UI components (including buttons,
    conversation displays, text inputs, labels, and radio selectors) to prevent
    low-visibility text.

**Multi-modal UI & File Handling**

-   [ ] **Multi-modal UI**:
    -   [ ] **UI Layout Presence**: Verify that the Image and Audio buttons are
        always fully declared and present in the layout XML and Kotlin/Java
        views, never dynamically hidden or removed.
    -   [ ] **Inference Fallback Controls**: If engine initialization falls back
        to text-only mode, verify that the Image and Audio buttons are
        explicitly set to disabled in code, blocking media inputs on text-only
        model loads.
    -   [ ] Verify that the image picker successfully resolves and loads local photos from storage (Gallery).
    -   [ ] Verify that the audio picker successfully resolves and loads local audio files from storage.
    -   [ ] Verify that picked **images are rendered visually** in the conversation display immediately upon selection.
    -   [ ] Verify that picked **audio is clearly indicated** (e.g., filename or icon) in the conversation display immediately upon selection.
    -   [ ] **Audio Filename Resolution**: Verify that
        `OpenableColumns.DISPLAY_NAME` is used to resolve the filename of picked
        audio files for display.
-   [ ] **Background File Operations**: Verify that all file copying and reading
    operations for models and media are performed on a background thread and do
    not block the UI thread.
-   [ ] **File Extension & Caching**: Verify that the app always copies picked
    files (model, video, or audio streams) to `context.cacheDir` (resolving
    streams instead of passing raw `content://` URIs) and that the original file
    extensions are fully preserved.
