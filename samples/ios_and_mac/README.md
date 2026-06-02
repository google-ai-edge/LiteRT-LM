<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LiteRT-LM Swift API for iOS and macOS

This directory provides examples and instructions for integrating large language
models natively into iOS and macOS applications using the LiteRT-LM Swift API.

## Prerequisites

- iOS 15.0 or later
- macOS 12.0 or later
- Xcode 15.0 or later
- A `.litertlm` model file (e.g., Gemma)

## Integration Steps

### 1. Add Dependency

Follow these steps to add the LiteRTLM Swift package to your Xcode project using
Swift Package Manager (SPM):

1. In Xcode, select **File** > **Add Package Dependencies...**
2. In the search bar at the top right, enter the GitHub repository URL:
   `https://github.com/google-ai-edge/LiteRT-LM.git`
3. Select the package from the list and click **Add Package**.
4. Select the target app you want to add the dependency to and click **Finish**.

> [!NOTE]
> If you see an error like `no such module LiteRTLM` after adding the package:
> 1. Click on your project in the project navigator.
> 2. Select your app target.
> 3. Go to the **General** tab.
> 4. Scroll down to **Frameworks, Libraries, and Embedded Content**.
> 5. Click the **`+`** button.
> 6. Select **LiteRTLM Package** -> **LiteRTLM**.
> 7. Click **Add**.

### 2. Add a Model File

1. Obtain a compatible `.litertlm` model file (e.g., **Gemma 4 E2B** from
   [Hugging Face](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)).
2. Drag and drop the model file into your Xcode project navigator. In the
   dialog that appears, ensure your app target is checked.
3. *(Alternative)* If the file is not found at runtime, go to your project's
   **Build Phases** > **Copy Bundle Resources** and add the file there
   manually.

### 3. Usage Example

A simple example of a SwiftUI chat application using streaming responses is
available in the [devsite Swift documentation][devsite-url].

[devsite-url]: https://ai.google.dev/edge/litert-lm/swift

It demonstrates how to:

- Find the model in the app bundle.
- Configure and initialize the `Engine`.
- Create a `Conversation` session.
- Send a message and handle the streaming response live in the UI.

## Running the App

### On iOS
1. Connect your physical iPhone to your Mac using a cable.
2. In Xcode, select your iPhone's name from the run destination menu at the
   top center of the window.
3. Click the **Run** button (or press `Cmd + R`) to build and run the app on
   your device!

### On macOS
1. In Xcode, select **My Mac** from the run destination menu at the top center
   of the window.
2. Click the **Run** button (or press `Cmd + R`) to build and run the app on
   your Mac!

> [!IMPORTANT]
> If you are using a locally built, unsigned version of the library on Mac,
  macOS may block it with a "Malware" warning. To bypass this for local testing,
  run this command in your Mac terminal:
> ```bash
> xattr -rd com.apple.quarantine path/to/CLiteRTLM.xcframework
> ```

> [!TIP]
> If your physical device's iOS version is lower than the deployment target set
> in Xcode, you can change it by going to the **General** tab, and under
> **Minimum Deployments** (or **Deployment Info**), change the iOS version to
> match your device's version.
