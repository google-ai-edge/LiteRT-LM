# Dependency Compliance Checklist

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
| **Common Setup & Build**                                     |        |          |
| No Source Copy (No C++ files in app dir)                     |        |          |
| Library Dependency Verification                              |        |          |
| **Source Build Specifics**                                   |        |          |
| Repo Check                                                   |        |          |
| Target Arch Verification                                     |        |          |
| Transitive Deps (Source Build)                               |        |          |
| Dual JAR Import Verification                                 |        |          |
| Prebuilt Folder Check                                        |        |          |
| **Maven Integration Specifics**                              |        |          |
| Maven Coordinate Verification                                |        |          |
| Maven Transitive Deps                                        |        |          |
<!-- mdformat on -->

### Detailed Verification Checklist

To complete the compliance report above, verify the following UI and
implementation details strictly:

**Common**

-   [ ] **No Source Copy**: Verify that you did NOT copy LiteRT-LM source files directly into the sample app project.
-   [ ] **Library Dependency**: Verify that the LiteRT-LM library is included in the sample app's `kt_android_library` dependencies.

**Source Build Specifics**

-   [ ] **Repo Check**: Verify that you checked for the LiteRT-LM repository and pulled or cloned it as needed before generating the sample app code.
-   [ ] **Target Arch Verification**: Do NOT rely on the directory structure in
    the APK or JAR. Verify Target Architecture by running `file` or `readelf -h`
    on the actual `.so` file.
-   [ ] **Transitive Deps (Source Build)**: Verify that you extracted all
    transitive dependencies by inspecting the LiteRT-LM bindings target's
    original source `BUILD` file, registered them inside `MODULE.bazel`, and
    added them to the `exports` list of `litert_lm_prebuilt/BUILD`.
-   [ ] **Dual JAR Import Verification**: Verify that **BOTH** the Kotlin Class
    JAR and the Native JNI JAR are stored in the `litert_lm_prebuilt` folder
    under `litert_lm_android_sample_app` and explicitly imported via
    `java_import` in `litert_lm_prebuilt/BUILD`.
-   [ ] **Prebuilt Folder Check**: Verify if any libraries in the source repo's
    `prebuilt/` directory are successfully included in your native packaging.

**Maven Integration Specifics**

-   [ ] **Maven Coordinate**: Verify that `com.google.ai.edge.litertlm:litertlm-android:latest.release` is used in `MODULE.bazel`.
-   [ ] **Maven Transitive Deps**: Verify that Maven automatically resolved transitive dependencies successfully, and handle any version locking conflicts if they arise.
