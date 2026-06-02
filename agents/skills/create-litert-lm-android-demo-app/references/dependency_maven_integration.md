# LiteRT-LM Maven Integration Guide

This document defines the rules for integrating LiteRT-LM via a prebuilt Maven
AAR for developing Kotlin/JNI (Android) applications.

* **No Core Modification**: Never modify the core C++ engine files unless explicitly requested by the user.

## Build & Dependencies

The prebuilt library can be resolved via Maven using `rules_jvm_external` in
`MODULE.bazel` with the coordinate:

* `com.google.ai.edge.litertlm:litertlm-android:latest.release`

You MUST add all required direct and transitive coordinates to your Bzlmod setup
if Bzlmod fails to resolve them automatically, preventing compile-time
`ImportDepsChecker` errors.

### `MODULE.bazel` Configuration

To configure Bzlmod dependency resolution for the prebuilt LiteRT-LM Maven
package, add the following to `MODULE.bazel`:

```starlark
bazel_dep(name = "rules_jvm_external", version = "6.2")

maven = use_extension("@rules_jvm_external//:extension.bzl", "maven")
maven.install(
    artifacts = [
        "androidx.core:core-ktx:1.12.0", # Required to handle Edge-to-Edge system window insets in Kotlin
        "com.google.ai.edge.litertlm:litertlm-android:latest.release",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
    version_conflict_policy = "pinned",
    use_starlark_android_rules = True,
    aar_import_bzl_label = "@rules_android//rules:rules.bzl",
)
use_repo(maven, "maven")
```

If compiling for a target device architecture different from your host system,
you MUST configure the Android NDK extension to provide the native C++
compilation toolchain:

*   **NDK Toolchain Version**:
    *   `rules_android_ndk` ~0.1.5

```starlark
android_ndk_repository_extension = use_extension(
    "@rules_android_ndk//:extension.bzl",
    "android_ndk_repository_extension",
)
android_ndk_repository_extension.configure(
    path = "/path/to/android-ndk",
)
use_repo(android_ndk_repository_extension, "androidndk")

# Add "@androidndk//:all" to register_toolchains in MODULE.bazel
```

### `BUILD` File Configuration

In your application's `BUILD` file, add the Maven target directly to your
dependencies:

```starlark
deps = [
    "@maven//:com_google_ai_edge_litertlm_litertlm_android",
]
```
