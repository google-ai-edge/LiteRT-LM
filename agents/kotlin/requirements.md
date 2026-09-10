# LiteRT-LM Kotlin Domain Requirements

This document defines the strict build, dependency, and API rules for developing
Kotlin/JNI (Android) applications using LiteRT-LM. Agents MUST follow these
rules to ensure platform compatibility and prevent crashes.

* **No Core Modification**: Never modify the core C++ engine files or `WORKSPACE` unless explicitly requested by the user.

## Part 1: Build & Dependencies

You can integrate LiteRT-LM via a prebuilt Maven AAR or by building from source
using Bazel.

### Option A: Maven AAR (Prebuilt)

1.  **Dependencies**: Use `aar_import` in `BUILD`. You MUST manually add
    transitive dependencies to your `BUILD` file to prevent `ImportDepsChecker`
    failures. For completeness, explicitly add:
    -   `@maven//:com_google_code_gson_gson`
    -   `@maven//:org_jetbrains_annotations`
    -   `@maven//:org_jetbrains_kotlin_kotlin_stdlib`
    -   `@maven//:org_jetbrains_kotlin_kotlin_reflect`
    -   `@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_core_jvm`
    -   `@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android`
2.  **GPU Permissions**: To prevent Adreno GPU crashes, add
    <!-- linter off -->
    `<uses-native-library android:name="libOpenCL.so" android:required="false"/>`
    <!-- linter on -->
    to `AndroidManifest.xml`.

### Option B: Build from Source

1.  **Repository Path**: If building from source, the agent must know the path
    to the LiteRT-LM repository. If the user specifies an existing repository
    path, that path should be used. Otherwise, the repository should be cloned
    from `https://github.com/google-ai-edge/LiteRT-LM` to a subdirectory under
    the root directory designated for the task (e.g., `[ROOT]/LiteRT-LM`).
2.  **Bazel Version**: You must build the `LiteRT-LM` repository using the Bazel
    version defined in its `.bazelversion` file.
3.  **Dual JAR Generation**: Building from source MUST yield TWO distinct JAR
    files. Ensure you target the correct architecture (e.g., use
    `--config=android_arm64` for ARM64 devices):
    *   **Kotlin Class JAR**: Compile the Kotlin bindings (e.g.,
        `//kotlin/java/com/google/ai/edge/litertlm:litertlm-android`). Locate
        the resulting output `.jar` that contains the compiled `.class` files.
    *   **Native JNI JAR**: Compile the native JNI libraries. Zip all required
        `.so` files into a directory structure `lib/<target_abi>/` inside a
        newly created JAR named `litertlm_native.jar`.
4.  **16KB Page Alignment**: For Android 15+, native `.so` libraries MUST be
    16KB aligned. Add
    <!-- linter off -->
    `linkopts = select({"@platforms//os:android": ["-Wl,-z,max-page-size=16384"], "//conditions:default": []})`
    <!-- linter on -->
    to your cc_binary. Verify using: `readelf -l path/to/lib.so` (ensure align
    is `0x4000`).
5.  **Packaging & Imports**:
    *   Store both the `litertlm_native.jar` (Native JNI JAR) and the Kotlin
        Class JAR in the `litert_lm_prebuilt` directory at the root level.
        Import them into your app using `java_import`.
    *   **Strict Constraint**: **DO NOT use `cc_import` or direct file
        references for prebuilt `.so` files in `android_binary` deps.** Bazel
        may place them in incorrect paths inside the APK.
    *   **Acceleration Libraries**: Explicitly check if the source repository
        contains a prebuilt directory (e.g., `prebuilt/android_arm64/`) and copy
        any relevant acceleration libraries (like `libLiteRtGpuAccelerator.so`)
        into your packaged Native JNI JAR alongside the built engine `.so`.

## Part 2: LiteRT-LM Kotlin API Invariants

#### 1. Engine Initialization

*   **Data Classes vs Builders**: `EngineConfig`, `SessionConfig`,
    `ConversationConfig`, and `SamplerConfig` are Kotlin data classes. Do NOT
    use `.builder()` patterns. Use direct constructors with named arguments.
*   **Backend Instantiation**: `Backend.CPU` and `Backend.GPU` are classes/data
    classes. You MUST instantiate them (e.g., `Backend.CPU()`, `Backend.GPU()`).
*   **Explicit Initialization**: After creating an `Engine` instance, you MUST
    call `engine.initialize()` before creating conversations or sessions.
*   **Multi-modal Executors**: Explicitly initialize `visionBackend` and
    `audioBackend` in `EngineConfig`.
*   **CPU Audio Constraint**: `audioBackend` MUST be set to `Backend.CPU()`.
    Passing GPU will cause immediate initialization errors.

#### 2. Conversation vs Session Usage

*   **API Routing**: The high-level `sendMessage` method is located in the
    `Conversation` class, not `Session`. Create a conversation using
    `engine.createConversation()`.

#### 3. Input Assembly

*   **Preserve File Extensions**: When copying picked files to the cache, you
MUST preserve the original file extension. When handling `content://` URIs, do
not rely on `Uri.lastPathSegment`. Query `ContentResolver` for
`OpenableColumns.DISPLAY_NAME` to get the actual filename.
*   **Explicit Content Passing**: The `sendMessage` method (or equivalent)
expects a list of `Content` objects for multi-modal input. Ensure you pass the
constructed list (text + media), not just a single text string.

#### 4. Threading and I/O

*   **Background Execution Required**: All engine initialization
    (`initialize()`), file copying, and inference (`sendMessage`) MUST execute
    on a background thread (e.g., `Dispatchers.IO`). Calling these on the Main
    thread causes an ANR crash.

#### 5. JNI Library Loading

*   **Ordered Loading**: If manually calling `System.loadLibrary()`, load
    dependencies first to prevent crashes.
*   **Custom Library Loaders**: Check if the library uses a custom loader (like
    `NativeLibraryLoader`). These loaders often expect libraries to be at
    specific package paths inside the JAR or APK, rather than the standard
    `lib/<abi>/` directory.

#### 6. Performance Optimization

*   **Multi-Token Prediction (MTP)**: For GPU backends, universally recommend
    enabling MTP. Set `ExperimentalFlags.enableSpeculativeDecoding = true`
    before initializing the engine.

## Part 3: App Build Configuration (Bazel 9 Only)

> [!IMPORTANT]
> The following rules apply SPECIFICALLY to building the sample application (or
> user applications) using **Bazel 9 and Bzlmod**. They may not be applicable
> for Bazel 7 or Gradle builds.

### Strict Target Toolchain: Bazel 9 Only
While the broader repository code allows Gradle integration elsewhere, this
specific sample app target **MUST be built exclusively with Bazel 9 using
Bzlmod**. Do NOT generate `build.gradle`, `build.gradle.kts`, or use Gradle
wrappers for this sample app.

### Environment Steps (Bazel 9 Configuration)
Configure the sample app's dependency mapping via `MODULE.bazel` using these
compatible version boundaries:

*   **Target Versions**: `rules_android` ~0.7.1, `rules_kotlin` ~2.3.20,
    `rules_java` ~7.12.2, `rules_jvm_external` ~6.2.
*   **Target SDK Version**: For the sample app build, you MUST target API 35+ in
    `AndroidManifest.xml` to ensure compatibility with Android 15's enforced
    Edge-to-Edge behavior. You MUST handle window insets in code (e.g., using
    `ViewCompat.setOnApplyWindowInsetsListener`) to prevent content from drawing
    behind system bars.

### Bazel 9 Structural Requirements

- **Bazel Versioning**: Different components (like `LiteRT-LM` repo and
  `sample_app`) may require different Bazel versions. Always run `bazelisk` from
  within the specific target directory to ensure the correct local
  `.bazelversion` is used. Avoid using dry `bazel` commands directly.
- **NDK Version**: NDK version 27 or higher is required.
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
        artifacts = [ ... ],
        version_conflict_policy = "pinned",
        use_starlark_android_rules = True,
        aar_import_bzl_label = "@rules_android//rules:rules.bzl",
    )
    ```
- **SDK Extension**: If auto-detection fails, configure it in `MODULE.bazel` as
  follows:
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

    register_toolchains(
        "@androidsdk//:sdk-toolchain",
        "@androidsdk//:all",
    )
    ```

## Troubleshooting

### Duplicate Artifact Versions in Coursier
If you see warnings about duplicate artifact versions in Coursier (e.g.,
`com.google.code.gson:gson has multiple versions`), check if different modules
are bringing in different versions.

- **Solution**: Ensure `version_conflict_policy = "pinned"` is set in
  `maven.install`.
