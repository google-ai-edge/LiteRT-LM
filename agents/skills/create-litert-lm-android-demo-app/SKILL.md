---
name: create-litert-lm-android-demo-app
description: >-
  Guides the agent to orchestrate and implement a standalone LiteRT-LM Android
  demo app with backend selection and multi-modality support using Bazel 9.
  Use when asked to create a demo app for LiteRT-LM on Android.
  Don't use for general Android app development without LiteRT-LM, or for
  building the core library alone without an app.
---

# Create a LiteRT-LM Android Demo App

This skill ensures that the generated LiteRT-LM Android demo app captures all
requirements on the first pass, strictly follows platform constraints, and
implements the user interface, state machine, and feature specifications.

## Core Constraints & Technical Rules

- **Monitoring Subagent**: You MUST maintain a dedicated, active monitoring
  subagent. This subagent is responsible for monitoring `task.md` to detect any
  loss of context or deviation from plan, and for ensuring strict adherence to
  all listed compliance items. Do not create duplicate subagents during context
  refreshes.
- **Airtight Compliance & Gating**: To successfully declare a task complete, you
  MUST satisfy both of these strict verification gates:
  - **No Heuristic Pass**: You are STRICTLY FORBIDDEN from using recollection or
    memory to pass items. Every `Pass` item MUST list a direct markdown link to
    the live workspace file with exact starting and ending line numbers in the
    `Evidence` column, or print the exact terminal log block. Blank or generic
    claims will be flagged as an immediate compliance failure.
  - **No Pending/Fail Items**: You MUST NOT notify of task completion if any
    checklist item remains in `Pending` or `Fail` status, or if any detailed
    checkbox `[ ]` is left unchecked. Every single row and checkbox MUST be
    verified and marked `Pass` or checked `[x]` with active evidence.
- **No Custom Planning**: You are STRICTLY FORBIDDEN from proposing, drafting,
  or generating any custom implementation plans or alternative planning files.
  The verbatim execution steps copied directly into `task.md` during **Step 1**
  serve as your SOLE, absolute plan. You MUST follow the tasks inside `task.md`
  sequentially step-by-step and track progress exclusively by checking off its
  checkboxes.
- **Just-In-Time (JIT) Reference Reading**: To prevent context pollution and
  early phase-leakage, you MUST NOT read reference guides (like
  `references/*.md`) ahead of time. Load and study each guide sheet strictly
  just-in-time as required by the active execution step. When loading or copying
  these files, resolve their path dynamically relative to where this `SKILL.md`
  is stored to prevent directory drift errors.
- **Strict Sequential Execution**: You MUST follow the execution steps in the
  exact numerical order listed in `SKILL.md` or `task.md`. You are NOT allowed
  to skip steps, jump ahead, or reorder operations (such as pulling dependencies
  early) based on active developer preference, unless you first ask for and
  receive explicit approval from the user.
- **Stick to the Chosen Build Path**: Once the build path (Source vs. Maven) is
  decided, stick strictly to it. Do NOT swap paths mid-task even upon
  compilation failures.
- **WORKSPACE Modification Ban**: Modifying `WORKSPACE` is banned. Compile
  exclusively via Bzlmod using `MODULE.bazel` for all extensions.
- **No Mock Inference**: Ensure the app starts in a clean, uninitialized state.
  Do not use hardcoded paths for the model picker. Real inference is required.
- **Verify Before Declaring Failure**: Always inspect active stdout/stderr logs
  before declaring a background command failed; ensure it isn't simply
  processing slowly.
- **Strict Asynchronous Verification**: For all background/async operations
  (cloning, Bazel builds), do NOT assume task completion based on partial or
  progress logs. You MUST actively check command status and verify successful
  completion before beginning any dependent steps.
- **Headless Environment**: Always assume there is no device or emulator
  available for execution. Inspect the implementation statically to ensure
  compliance requirements are met, and mark runtime tests in the compliance
  reports as "Pass (Static/Build Only)".
- **Tool Path & Reference Persistence**: You MUST resolve and persistently
  record the absolute paths of all local/external tools (Android SDK/NDK,
  bazelisk) and all relative skill `references/*.md` files under an
  **Environment, Tools & Reference Paths** section in the top of `task.md`
  during **Step 1** (before your execution Cwd changes). For all subsequent
  steps, you MUST use these recorded absolute paths for all view, read, and
  copy operations.

---

## Context Scenarios

### Scenario: Source Build

- **Prerequisites**: You have a local clone of the `LiteRT-LM` repository, or
  intend to build the project from source.
- **Path Context**: The source code is available. All paths are relative to the
  cloned repository root. You will build dependencies from source using Bazel.
- **Dependency Action**: You MUST read `references/dependency_source_build.md`
  for instructions on how to obtain the LiteRT-LM shared library and Kotlin
  interface.

### Scenario: Maven Integration

- **Prerequisites**: You are creating a standalone Android application without the LiteRT-LM source code.
- **Path Context**: Only the requirement and skill markdowns are available. You will resolve the LiteRT-LM dependencies via Maven.
- **Dependency Action**: You MUST read
  `references/dependency_maven_integration.md` for instructions on how to obtain
  the LiteRT-LM shared library and Kotlin interface.

---

## Execution Steps

1. **Create `task.md` Checklist**: Create a `task.md` file to track progress and constraints.
   - **1.1 Copy Execution Steps**: You MUST copy **ALL** execution steps,
     including all numbered sub-steps and **Context Refresh** instructions,
     verbatim into `task.md` as a checklist. Do not summarize or omit any steps.
   - **1.2 Copy Constraints & Rules**: You MUST copy the complete
     **Core Constraints & Technical Rules** section from `SKILL.md` verbatim
     into the top of `task.md` to ensure they remain persistently active
     constraints in file system memory.
   - **1.3 Context Refresh Safeguard**: You MUST copy instructions into the top
     of `task.md` directing you to perform the following actions before
     beginning any next step (`x.`) or sub-step (`x.y`), or if the active
     history is compacted:
     - 1.3.1 Reload and read the ENTIRE `SKILL.md` file from line 1 to the end.
     - 1.3.2 Fully reload and read any referenced markdown documents associated
       with that step (e.g., `references/ui_layout_and_state.md` before
       initiating `3.2`).
     - 1.3.3 Check if a monitoring subagent is active for this task, and if not,
       instantiate one as required by the **Monitoring Subagent Constraint**.
   - **1.4 Instantiate Monitoring Subagent**: Instantiate and launch the
     dedicated monitoring subagent immediately as required by the
     **Monitoring Subagent Constraint**.
   - **1.5 Record Absolute Reference Paths**: You MUST immediately resolve the
     absolute path onto each relative `references/*.md` file listed in this
     skill from your starting trigger directory, and record them under an
     **Environment, Tools & Reference Paths** section in the top of `task.md`.

2. **Gather Inputs & Initial Environment Setup**:
   - **2.1 Project Root**: Ask the user for the root path to store the sample code and all dependencies.
   - **2.2 Target Scenario**: Ask the user to choose their target scenario (from
     the **Context Scenarios** above) to determine the build path, but
     **only record the choice** for use in step
     **Obtain LiteRT-LM Shared Library and Kotlin Interface**.
   - **2.3 Android SDK/NDK Setup**: Ask the user for their absolute path to an existing Android SDK.
     - *If provided*: Record the path to configure the Bazel build. Verify NDK version 27.3.13750724 is installed and used.
     - *If not provided*: Download the Android SDK command-line tools and
       extract them under the task's root directory
       (e.g., `[ROOT]/android_sdk`). Use the `sdkmanager` to install required
       platforms, build tools, and strictly NDK version 27.3.13750724, ensuring
       all licenses are accepted.
   - **2.4 Target Device & Model**: Ask for the target device (to determine architecture) and the target model.
     - Let the user know that the default target device is Pixel 10.
     - Let the user know that the default model is
       `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main`,
       and ask if they want to change it.
   - **2.5 Environment Verification**: Download the specified model to the root path defined by the user.

3. **Implement App UI and State (Excluding Inference)**:
   - **3.1 Load UI Specifications**: Load `references/ui_layout_and_state.md` and follow the instructions inside to create the app UI and state machine (everything except actual inference calls).
   - **3.2 Initialize UI Checklist**: Create `compliance_review_ui.md` by copying `references/compliance_checklist_ui.md`, and initialize all statuses to 'Pending'.
   - **3.3 Verify Layout & State**: Compile and verify that the UI layouts, insets, and other components build perfectly without JNI/native code references.
   - **3.4 Local UI Compliance Audit**: Read and audit `compliance_review_ui.md` in place and save the file with detailed results and evidence (citing line-numbered links).

4. **Obtain LiteRT-LM Shared Library and Kotlin Interface**:
   - **4.1 Load Scenario Reference**: Load the appropriate scenario guide identified in step **2.2** (e.g., `references/dependency_source_build.md` or `references/dependency_maven_integration.md`).
   - **4.2 Initialize Dependency Checklist**: Create `compliance_review_dependency.md` by copying `references/compliance_checklist_dependency.md`, and initialize all statuses to 'Pending'.
   - **4.3 Integrate Library & Interface**: Follow the scenario instructions to obtain the required LiteRT-LM shared library and Kotlin interface.
   - **4.4 Local Dependency Compliance Audit**: Read and audit `compliance_review_dependency.md` in place and save the file with detailed results and evidence (citing line-numbered links).

5. **Full Implementation with LiteRT-LM Integration**:
   - **5.1 Load Inference Rules**: You MUST read `references/inference_implementation.md` to understand the specific rules for integrating LiteRT-LM.
   - **5.2 Initialize Inference Checklist**: Create `compliance_review_inference.md` by copying `references/compliance_checklist_inference.md`, and initialize all statuses to 'Pending'.
   - **5.3 Implement Inference**: You MUST immediately integrate LiteRT-LM
     inference into the application once the App UI and State builds
     successfully. Do NOT pause for user feedback unless blocked by unsolvable
     errors.
   - **5.4 Local Inference Compliance Audit**: Read and audit `compliance_review_inference.md` in place and save the file with detailed results and evidence (citing line-numbered links).

6. **Verification and Compliance Audit**:
   - **6.1 Re-evaluate Compliance Reports**: Reload and read all three existing
     compliance reports (`compliance_review_ui.md`,
     `compliance_review_dependency.md`, and
     `compliance_review_inference.md`) sequentially one by one to fully
     rehydrate them in your active context, re-running full verification checks
     to ensure subsequent changes in later steps did not break earlier passes.
     Update your evidence blocks with new integrated code files and line
     numbers.
   - **6.2 Verify Compliance Gates (No Pending/Fail Items)**: Verify that
     **absolutely no** checklist items are left in `Pending` or `Fail` status in
     any of the reports. All applicable blocks MUST be signed off as `Pass` and
     checkboxes ticked `[x]`, as required by **No Pending Items Constraint**.
   - **6.3 Final Comprehensive Audit Sign-Off**: Once all compliance reviews
     have successfully passed, perform a final overall audit of the codebase and
     build artifacts (such as the compiled APK, logs, or generated binaries) to
     guarantee that the compiled task is 100% robust and complete.

7. **Deploy**:
   - **7.1 Installation Instructions**: Provide instructions to the user to install the APK and push the model file to their device.
   - **7.2 Device Requirements**: Remind the user to verify that the target
     Android phone allows custom APK installation. Some devices may require a
     `userdebug` build of Android to allow installation of debug APKs or
     disabling package verification.
