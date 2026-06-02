# App UI and State Configuration

This document defines the rules for setting up the Android application
workspace, configuring the Bazel 9 build, and defining the App UI and State.
Agents MUST follow these rules to ensure platform compatibility and prevent
crashes.

## Workspace Setup

1. **Create the App Directory**: Create a new folder named
   `litert_lm_android_sample_app` directly under the designated root directory.
   * All Android application source code, `MODULE.bazel`, and `BUILD` files MUST
     reside inside this `litert_lm_android_sample_app` directory, which serves
     as the Bazel workspace root.

## App Build Configuration (Example using Bazel 9)

> [!NOTE]
> The following rules apply to building the sample application using **Bazel 9
> and Bzlmod**, which is the environment tested to be working. Other build tools
> (e.g., Gradle or Bazel 7) or versions may work but are not covered here.

### Target Toolchain: Bazel 9
While the broader repository code may support other build systems like Gradle,
this guide focuses on **Bazel 9 using Bzlmod**. It is recommended to use this
setup unless you are familiar with adapting it to other build systems.

### Environment Steps (Bazel 9 Configuration)
Configure the sample app's dependency mapping via `MODULE.bazel` using these
compatible version boundaries:

*   **Target Versions**:
    *   `rules_android` ~0.7.1
    *   `rules_kotlin` ~2.3.20
    *   `rules_java` ~7.12.2
    *   `rules_jvm_external` ~6.2
*   **Target SDK Version**: For the sample app build, you MUST target API 35+ in
    `AndroidManifest.xml` to ensure compatibility with Android 15's enforced
    Edge-to-Edge behavior. You MUST handle window insets in code (e.g., using
    `ViewCompat.setOnApplyWindowInsetsListener`) to prevent content from drawing
    behind system bars.

### Bazel 9 Structural Requirements

- **Bazel Versioning**: Always run `bazelisk` from within the specific target
  directory to ensure the correct local `.bazelversion` is used. Avoid using dry
  `bazel` commands directly.
- **No WORKSPACE File**: For Bazel 9 sample apps, you **MUST NOT** create a
  `WORKSPACE` file. Use Bzlmod and configure dependencies exclusively via
  `MODULE.bazel`.
- **Targeting ARM64**: Required when targeting an ARM64 device (like Pixel 8+).
  - **DO NOT** use legacy flags like `--fat_apk_cpu`.
  - **DO** configure the platform flag `--android_platforms=@rules_android//:arm64-v8a`.
  - **Mandatory `task.md` Flag Setup**: You MUST persistently record the
    platform flag `--android_platforms=@rules_android//:arm64-v8a` under the
    **Environment, Tools & Reference Paths** section in `task.md` and append
    this flag to every `bazel build`/`bazel test` command to ensure correct
    target architectures.
  - *Option A*: Ensures native libraries are correctly extracted from fat AARs.
  - *Option B*: Ensures C++ code is compiled for the correct architecture.
- **Java Rules**: If `java_import` fails to resolve in `BUILD`, explicitly load
  it: `load("@rules_java//java:defs.bzl", "java_import")`.
- **C++17**: Pass `--cxxopt=-std=c++17 --host_cxxopt=-std=c++17` during build
  execution.
- **Load Paths**: In `BUILD` files, load Android rules from
  `@rules_android//rules:rules.bzl` instead of
  `@rules_android//android:rules.bzl`.
- **AppCompat Avoidance**: Extend `Activity` rather than `AppCompatActivity`.
  Avoid `androidx.appcompat` to prevent missing style crashes. Because you are
  avoiding AppCompat, you MUST use `startActivityForResult` for file picking
  instead of the modern `registerForActivityResult` (which often fails on bare
  Activity classes).
- **AAR Dependencies via Bzlmod**: When using `rules_jvm_external` to pull in
  AAR dependencies (like `Markwon`), you MUST configure `maven.install` to use
  Starlark Android rules to avoid `name 'aar_import' is not defined` errors:
    ```starlark
    maven.install(
        artifacts = [
            "androidx.core:core-ktx:1.12.0", # Required to handle Edge-to-Edge system window insets in Kotlin
            # Add other application UI dependencies here (e.g. "io.noties.markwon:core:4.6.2")
        ],
        version_conflict_policy = "pinned",
        use_starlark_android_rules = True,
        aar_import_bzl_label = "@rules_android//rules:rules.bzl",
    )
    ```
- **SDK Extension**: You MUST configure the SDK in `MODULE.bazel` as follows:
    ```starlark
    android_sdk_repository_extension = use_extension(
        "@rules_android//rules/android_sdk_repository:rule.bzl",
        "android_sdk_repository_extension"
    )
    android_sdk_repository_extension.configure(
        path = "/path/to/Android/Sdk",
        api_level = 35,
        build_tools_version = "35.0.0",
    )
    use_repo(android_sdk_repository_extension, "androidsdk")

    # Add "@androidsdk//:sdk-toolchain" and "@androidsdk//:all" to register_toolchains in MODULE.bazel
    ```

## UI Specification

### UI State Machine

1.  **Initial State**: Only the Model Picker is enabled. All other inputs
    (Backend Selector, Message Inputs) are disabled. *Note*: Disable all
    individual `RadioButtons`, not just the `RadioGroup`.
2.  **Model Picked**: Disable the Model Picker. Show a loading animation while
    copying the file to cache. Once copied, enable the Backend Selector.
3.  **Backend Selected**: Disable the Backend Selector. Show loading
    animation while initializing the mock engine.
4.  **Model Loaded**: Stop loading animation. Support 2 UI states depending on the loaded model:
    - **Multi-modal Model**: Enable all inputs (Image, Audio, Text input, and Send button).
    - **Text-only Model**: Enable only the Text input and Send button.
      **The "Image" and "Audio" buttons MUST remain disabled.**
5.  **Inference State**: Show loading animation. Disable all message inputs
    to prevent concurrent requests. Re-enable upon completion/error.

- **Loading Animation Helper**: You **MUST** use a shared loading animation helper function for all loading states to ensure UI consistency.

### UI Component Details

- **Buttons**: The "Pick Model", "Image", and "Audio" buttons should share the
  same visual style for UI consistency.
- **Model Picker**:
    - Triggering the "Pick Model" button MUST invoke a real Android system file
      picker using `Intent(Intent.ACTION_GET_CONTENT)` with MIME type set to
      `*/*` to support varying model file extensions.
    - Configure this file picker using `startActivityForResult` inside the button click listener.
    - The name of the selected model MUST be displayed horizontally to the right
      of the "Pick Model" button. It MUST show the actual resolved filename
      (see **Filename Resolution** below), not a hardcoded alias like
      `model.bin`.
- **Backend Selector**: Radio buttons for "CPU" and "GPU". The label "Backend:"
  MUST be placed to the left.
- **Conversation Display**:
    - Use a vertical `LinearLayout` inside a `ScrollView`.
    - The `ScrollView` **MUST** have an ID assigned (e.g., `@+id/sv_conversation`) to allow programmatic scrolling.
    - Set `adjustViewBounds = true` on Images.
    - You **MUST** support rendering Markdown in text messages (e.g., using `io.noties.markwon:core:4.6.2`).
- **Input Fields**:
    - The Text Input field should be scrollable to handle long user queries.
    - A send button must be placed on the right-hand side of the text input
      field.
- **Media Inputs**:
    - **Image Picker**: Clicking the "Image" button MUST trigger a real system
      file picker using `Intent(Intent.ACTION_GET_CONTENT)` with MIME type set
      to `"image/*"` to load a local photo from storage.
    - **Audio Picker**: Clicking the "Audio" button MUST trigger a real system
      file picker using `Intent(Intent.ACTION_GET_CONTENT)` with MIME type set
      to `"audio/*"` to load a local audio file from storage.
- **Filename Resolution**: When handling `content://` URIs (e.g., from file
  pickers), you MUST NOT rely on `Uri.lastPathSegment`. You MUST query
  `ContentResolver` for `OpenableColumns.DISPLAY_NAME` to get the actual
  filename.
- **Preserve File Extensions & Caching**: You MUST always copy all selected
  files (model, video, or audio streams) to `context.cacheDir` and resolve and
  preserve their original file extensions. Do NOT pass raw `content://` URIs
  directly to native API calls, as native engines and codecs inspect extensions
  to verify format compatibility.

### UI Layout Diagram
The relative vertical positioning of components **MUST** strictly follow the
order shown below:

```text
|--------------|
| Pick Model   |   Model Name
|--------------|

Backend: ( ) CPU    ( ) GPU

|-----------------------------|
|                             |
|        Conversation         |
|           Display           |
|        (ScrollView)         |
|                             |
|         [loading...]        |
|-----------------------------|

|-------|     |-------|
| Image |     | Audio |
|-------|     |-------|

|---------------------|
|                     |   |-------|
|      Text input     |   | Send  |
|                     |   |-------|
|---------------------|
```

## Constraints and Implementation Rules

- **Consistent Loading Animation**: You **MUST** use a shared helper function
  (e.g., `startLoadingAnimation`) to ensure all loading states (model picking,
  engine init, waiting for response) use the exact same animation style and
  speed (`loading.`, `loading..`, `loading...` with a 500ms delay per dot)
  **and display it in the conversation display**.
- **Main Thread UI Updates**: Explicitly dispatch UI updates and animation
  states to the Main thread (e.g., using `withContext(Dispatchers.Main)`).
- **Auto-Scroll Conversation**: Use `fullScroll(View.FOCUS_DOWN)` wrapped in
  `.post {}` when adding new content and when updating text during streaming
  responses to keep the latest content visible.
- **Explicit UI Text Contrast**: To prevent dynamic theme contrast bugs:
  1. **Disable Force Dark**: You MUST set `android:forceDarkAllowed="false"` on activity root layouts.
  2. **Contrast Pairings**: You MUST explicitly set contrasting text and background/container colors for all components—including buttons, conversation displays, text inputs, labels, and radio selectors—to guarantee complete readability.

## Troubleshooting

### Duplicate Artifact Versions in Coursier
If you see warnings about duplicate artifact versions in Coursier (e.g.,
`com.google.code.gson:gson has multiple versions`), check if different modules
are bringing in different versions.

- **Solution**: Ensure `version_conflict_policy = "pinned"` is set in
  `maven.install`.
