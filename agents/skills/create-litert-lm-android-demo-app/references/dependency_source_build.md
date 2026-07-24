# LiteRT-LM Source Build Guide

This document provides instructions for obtaining the LiteRT-LM shared library
and Kotlin interface by building the repository from source, and how to
integrate them. Agents MUST follow these steps when the user chooses the
**Source Build** scenario.

* **No Core Modification**: Never modify the core C++ engine files unless explicitly requested by the user.

## Build Instructions

1.  **Repository Path Identification**: Ask the user if they have already cloned
    the `LiteRT-LM` repository locally and to provide the path if so.
    *   **If the path is provided and valid**: Use this existing repository path
        to build the library.
    *   **If the repository does not exist locally**:
        1. First, verify that `git-lfs` is installed on the system (if it is
           missing, halt and ask the user to install it).
        2. Once verified, clone the repository from
           `https://github.com/google-ai-edge/LiteRT-LM` into a new subdirectory
           under the task's root directory (e.g., `[ROOT]/LiteRT-LM`).
2.  **Bazel Version**: You must build the `LiteRT-LM` repository using the Bazel
    version defined in its `.bazelversion` file.
3.  **Dual JAR Generation**: Building from source MUST yield TWO distinct JAR
    files. Ensure you target the correct architecture by using the corresponding
    pre-configured flag in your Bazel commands (e.g., `--config=android_arm64`
    for `arm64-v8a`, `--config=android_arm` for `armeabi-v7a`,
    `--config=android_x86` for `x86`, or `--config=android_x86_64` for
    `x86_64`):
    *   **Kotlin Class JAR**: Compile the Kotlin bindings (e.g.,
        `//kotlin/kotlin/com/google/ai/edge/litertlm:litertlm-android`). Locate
        the resulting output `.jar` that contains the compiled `.class` files.
    *   **Native JNI JAR**: Compile the native JNI libraries. Zip all required
        `.so` files into a directory structure `lib/<target_abi>/` inside a
        newly created JAR named `litertlm_native.jar`. **CRITICAL**: The
        `<target_abi>` directory MUST be exactly the standard Android ABI name
        (`arm64-v8a`, `armeabi-v7a`, `x86`, or `x86_64`). Do NOT use the Bazel
        config name (like `android_arm64`) as the folder name.
4.  **Packaging & Imports**:
    *   Store both the `litertlm_native.jar` (Native JNI JAR) and the Kotlin
        Class JAR in the `litert_lm_prebuilt` directory under the
        `litert_lm_android_sample_app` folder (which serves as the Bazel
        workspace root).
        Import them into your app by creating a `BUILD` file in that directory
        with `java_import` rules.
    *   **Strict Constraint**: **DO NOT use `cc_import` or direct file
        references for prebuilt `.so` files in `android_binary` deps.** Bazel
        may place them in incorrect paths inside the APK.
    *   **Acceleration Libraries**: Explicitly check if the source repository
        contains a prebuilt directory (e.g., `prebuilt/android_arm64/`) and copy
        any relevant acceleration libraries (like `libLiteRtGpuAccelerator.so`)
        into your packaged Native JNI JAR alongside the built engine `.so`.

## Integration

After generating the JAR files and placing them in
`litert_lm_android_sample_app/litert_lm_prebuilt/`, configure your project as
follows.

### `litert_lm_prebuilt/BUILD` Configuration

Create a `BUILD` file in the `litert_lm_prebuilt` directory to register the
prebuilt targets.

**Discovered Dependency Exports**: When configuring the `litertlm_kotlin`
target, you MUST inspect the original `BUILD` file of the bindings library
inside your cloned LiteRT-LM source tree (e.g. read the `deps` attribute of the
`//kotlin/kotlin/com/google/ai/edge/litertlm:litertlm-android` build target).
Register all third-party external libraries listed there inside the `exports`
attribute of the prebuilt `java_import`. This transitively propagates compiling
and runtime classpaths to the sample app, maintaining clean modular
encapsulation.

```starlark
load("@rules_java//java:defs.bzl", "java_import")

package(default_visibility = ["//visibility:public"])

java_import(
    name = "litertlm_kotlin",
    jars = ["litertlm-android.jar"],
    exports = [
        # Export external libraries used inside the Kotlin binding prebuilts
        # so they are automatically added to down-stream classpaths depending on this rule.
        "@maven//:com_example_library_transitive_dependency",
    ],
)

java_import(
    name = "litertlm_native",
    jars = ["litertlm_native.jar"],
)
```

### `MODULE.bazel` Configuration

Even inside the Source Build, you MUST configure your `MODULE.bazel` to fetch
all external transitive dependency libraries utilized by the LiteRT-LM bindings
(such as JSON packages or core Android targets) using `rules_jvm_external`:

```starlark
maven.install(
    artifacts = [
        "androidx.core:core-ktx:1.12.0", # Required to handle Edge-to-Edge system window insets in Kotlin

        # CRITICAL: You MUST explicitly register external dependencies required by the
        # prebuilt Kotlin bindings (e.g., "com.example.library:transitive-dependency:1.0.0").
        "com.example.library:transitive-dependency:1.0.0",
    ],
    ...
)
```

### Transitive Dependency Analysis (Log/Bytecode Verification)

Once prebuilt targets and Bzlmod configurations are complete, you MUST perform a
validation check to statically verify that no transitive bytecode imports were
missed during target configuration. Run the standard JDK `jdeps -summary` tool
against the generated bindings JAR as a checkpoint to confirm all package
dependencies are satisfied on the active classpath:

*   **Verification command**:
    ```bash
    jdeps -summary litert_lm_prebuilt/litertlm-android.jar
    ```

### Application `BUILD` File Configuration

In your application's `BUILD` file (e.g., `app/BUILD`), depend directly on the
prebuilt bindings. Because the transitive dependencies are cleanly exported by
`litertlm_kotlin`, no redundant target declarations are required in the app
target:

```starlark
# In your android_binary or kt_android_library target:
deps = [
    "//litert_lm_prebuilt:litertlm_kotlin",
    "//litert_lm_prebuilt:litertlm_native",
    # ... other dependencies
]
```
