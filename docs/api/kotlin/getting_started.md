# LiteRT-LM Kotlin API

The Kotlin API of LiteRT-LM for **Android** and **JVM (Linux, MacOS, Windows)**
with features like **GPU and NPU acceleration**, **multi-modality**, and
**tools use**.

## Introduction

Here is a sample terminal chat app built with the Kotlin API:

```kotlin
import com.google.ai.edge.litertlm.*

suspend fun main() {
  Engine.setNativeMinLogSeverity(LogSeverity.ERROR) // Hide log for TUI app

  val engineConfig = EngineConfig(modelPath = "/path/to/model.litertlm")
  Engine(engineConfig).use { engine ->
    engine.initialize()

    engine.createConversation().use { conversation ->
      while (true) {
        print("\n>>> ")
        conversation.sendMessageAsync(readln()).collect { print(it) }
      }
    }
  }
}
```

![](demo.gif)

To try out the above sample, clone the repo and run with
[example/Main.kt](../../../kotlin/kotlin/com/google/ai/edge/litertlm/example/Main.kt):

```bazel
bazel run -c opt //kotlin/kotlin/com/google/ai/edge/litertlm/example:main -- <abs_model_path>
```

Available `.litertlm` models are on the
[HuggingFace LiteRT Community](https://huggingface.co/litert-community). The
above animation was using the
[Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT).

For Android sample, check out the
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app.

## Getting Started with Gradle

While LiteRT-LM is developed with Bazel, we provide the Maven packages for
Gradle/Maven users.

### 1. Add the Gradle dependency

#### Standard

```
dependencies {
    // For Android source set
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")

    // OR

    // For JVM (Linux, macOS, Windows) source set
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:latest.release")
}
```

#### Multiplatform

```
commonMain {
    dependencies {
        // For common source set
        implementation("com.google.ai.edge.litertlm:litertlm-core:latest.release")
    }
}
androidMain {
    dependencies {
        // For Android source set
        implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")
    }
}
jvmMain {
    dependencies {
        // For JVM (Linux, macOS, Windows) source set
        implementation("com.google.ai.edge.litertlm:litertlm-jvm:latest.release")
    }
}
```

You can find the available versions on Google Maven in
[litertlm-core](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-core),
[litertlm-android](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-android),
and
[litertlm-jvm](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-jvm).

`latest.release` can be used to get the latest release.

### 2. Initialize the Engine

The `Engine` is the entry point to the API. Initialize it with the model path
and configuration. Remember to close the engine to release resources.

**Note:** The `engine.initialize()` method can take a significant amount of time
(e.g., up to 10 seconds) to load the model. It is strongly recommended to call
this on a background thread or coroutine to avoid blocking the UI thread.

```kotlin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig

val engineConfig = EngineConfig(
    modelPath = "/path/to/your/model.litertlm", // Replace with your model path
    backend = Backend.CPU(), // Or Backend.GPU() and Backend.NPU("...")
    // Optional: Pick a writable dir. This can improve 2nd load time.
    // cacheDir = "/tmp/" or context.cacheDir.path (for Android)
)

val engine = Engine(engineConfig)
engine.initialize()
// ... Use the engine to create a conversation ...

// Close the engine when done
engine.close()
```

On Android, to use the GPU backend, the app needs to request the depending
native libraries explicitly by adding the following to your
`AndroidManifest.xml` inside the `<application>` tag:

```xml
  <application>
    <uses-native-library android:name="libvndksupport.so" android:required="false"/>
    <uses-native-library android:name="libOpenCL.so" android:required="false"/>
  </application>
```

To use the **NPU** backend, you might need to specify the directory containing
the NPU libraries. On Android, if the libraries are bundled with your app, set
it to `context.applicationInfo.nativeLibraryDir`. See [LiteRT-LM
NPU](https://ai.google.dev/edge/litert/next/litert_lm_npu#NPU) for more details
about the NPU native libraries.

```kotlin
val engineConfig = EngineConfig(
    modelPath = modelPath,
    backend = Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
)
```

### 3. Create a Conversation

Once the engine is initialized, create a `Conversation` instance. You can
provide a `ConversationConfig` to customize its behavior.

```kotlin
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig

// Optional: Configure the system instruction, initial messages, sampling
// parameters, etc.
val conversationConfig = ConversationConfig(
    systemInstruction = Contents.of("You are a helpful assistant."),
    initialMessages = listOf(
        Message.user("What is the capital city of the United States?"),
        Message.model("Washington, D.C."),
    ),
    samplerConfig = SamplerConfig(topK = 10, topP = 0.95, temperature = 0.8),
)

val conversation = engine.createConversation(conversationConfig)
// Or with default config:
// val conversation = engine.createConversation()

// ... Use the conversation ...

// Close the conversation when done
conversation.close()
```

`Conversation` implements `AutoCloseable`, so you can use the `use` block for
automatic resource management for one-shot or short-lived conversations:

```kotlin
engine.createConversation(conversationConfig).use { conversation ->
    // Interact with the conversation
}
```

### 4. Sending Messages

There are three ways to send messages:

-   **`sendMessage(contents, extraContext): Message`**: Synchronous call that
    blocks until the model returns a complete response. This is simpler for
    basic request/response interactions.
-   **`sendMessageAsync(contents, callback, extraContext)`**: Asynchronous call
    for streaming responses. This is better for long-running requests or when
    you want to display the response as it's being generated.
-   **`sendMessageAsync(contents, extraContext): Flow<Message>`**: Asynchronous
    call that returns a Kotlin Flow for streaming responses. This is the
    recommended approach for Coroutine users.

**Synchronous Example:**

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message

print(conversation.sendMessage("What is the capital of France?"))
```

**Asynchronous Example with callback:**

Use `sendMessageAsync` to send a message to the model and receive responses
through a callback.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback

val callback = object : MessageCallback {
    override fun onMessage(message: Message) {
        print(message)
    }

    override fun onDone() {
        // Streaming completed
    }

    override fun onError(throwable: Throwable) {
        // Error during streaming
    }
}

conversation.sendMessageAsync("What is the capital of France?", callback)
```

**Asynchronous Example with Flow:**

Use `sendMessageAsync` (without the callback arg) to send a message to the model
and receive responses through a Kotlin Flow.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

// Within a coroutine scope
conversation.sendMessageAsync("What is the capital of France?")
    .catch { ... } // Error during streaming
    .collect { print(it.toString()) }
```

### 5. Multi-Modality

Note: This only works with models with multi-modality support, e.g., the
[Gemma3n](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).

`Message` objects can contain different types of `Content`, including `Text`,
`ImageBytes`, `ImageFile`, and `AudioBytes`, `AudioFile`.

```kotlin
// Initialize the `visionBackend` and/or the `audioBackend`
val engineConfig = EngineConfig(
    modelPath = "/path/to/your/model.litertlm", // Replace with your model path
    backend = Backend.CPU(), // Or Backend.GPU() or Backend.NPU(...)
    visionBackend = Backend.GPU(), // Or Backend.NPU(...)
    audioBackend = Backend.CPU(), // Or Backend.NPU(...)
)

// Sends a message with multi-modality.
// See the Content class for other variants.
conversation.sendMessage(Contents.of(
    Content.ImageFile("/path/to/image"),
    Content.AudioBytes(audioBytes), // ByteArray of the audio
    Content.Text("Describe this image and audio."),
))
```

### 6. Defining and Using Tools

Note: This only works with models with tool support, e.g., the
[FunctionGemma](https://huggingface.co/google/functiongemma-270m-it).

There are two ways to define tools:

1.  With Kotlin functions (recommended for most cases)
2.  With Open API specification (full control of the tool spec and execution)

#### Defining Tools with Kotlin Functions

You can define custom Kotlin functions as tools that the model can call to
perform actions or fetch information.

Create a class implementing `ToolSet` and annotate methods with `@Tool` and
parameters with `@ToolParam`.

```kotlin
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam

class SampleToolSet: ToolSet {
    @Tool(description = "Get the current weather for a city")
    fun getCurrentWeather(
        @ToolParam(description = "The city name, e.g., San Francisco") city: String,
        @ToolParam(description = "Optional country code, e.g., US") country: String? = null,
        @ToolParam(description = "Temperature unit (celsius or fahrenheit). Default: celsius") unit: String = "celsius"
    ): Map<String, Any> {
        // In a real application, you would call a weather API here
        return mapOf("temperature" to 25, "unit" to  unit, "condition" to "Sunny")
    }

    @Tool(description = "Get the sum of a list of numbers.")
    fun sum(
        @ToolParam(description = "The numbers, could be floating point.") numbers: List<Double>,
    ): Double {
        return numbers.sum()
    }
}
```

Behind the scenes, the API inspects these annotations and the function signature
to generate an OpenAPI-style schema. This schema describes the tool's
functionality, parameters (including their types and descriptions from
`@ToolParam`), and return type to the language model.

##### Parameter Types

The types for parameters annotated with `@ToolParam` can be `String`, `Int`,
`Boolean`, `Float`, `Double`, or a `List` of these types (e.g., `List<String>`).
Use nullable types (e.g., `String?`) to indicate nullable parameters. Set a
default value to indicate that the parameter is optional, and mention the
default value in the description in `@ToolParam`.

##### Return Type

The return type of your tool function can be any Kotlin type. The result will be
converted to a JSON element before being sent back to the model.

-   `List` types are converted to JSON arrays.
-   `Map` types are converted to JSON objects.
-   Primitive types (`String`, `Number`, `Boolean`) are converted to the
    corresponding JSON primitive.
-   Other types are converted to strings with the `toString()` method.

For structured data, returning `Map` or a data class that will be converted to a
JSON object is recommended.

#### Defining Tools with OpenAPI Specification

Alternatively, you can define a tool by implementing the `OpenApiTool` class and
providing the tool's description as a JSON string conforming to the Open API
specification. This method is useful if you already have an OpenAPI schema for
your tool or if you need fine-grained control over the tool's definition.

```kotlin
import com.google.ai.edge.litertlm.OpenApiTool

class SampleOpenApiTool : OpenApiTool {

    override fun getToolDescriptionJsonString(): String {
        return """
        {
          "name": "addition",
          "description": "Add all numbers.",
          "parameters": {
            "type": "object",
            "properties": {
              "numbers": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              },
              "description": "The list of numbers to sum."
            },
            "required": [
              "numbers"
            ]
          }
        }
        """.trimIndent() // Tip: trim to save tokens
    }

    override fun execute(paramsJsonString: String): String {
        // Parse paramsJsonString with your choice of parser/deserializer and
        // execute the tool.

        // Return the result as a JSON string
        return """{"result": 1.4142}"""
    }
}
```

#### Registering Tools

Include instances of your tools in the `ConversationConfig`.

```kotlin
val conversation = engine.createConversation(
    ConversationConfig(
        tools = listOf(
            tool(SampleToolSet()),
            tool(SampleOpenApiTool()),
        ),
        // ... other configs
    )
)

// Send messages that might trigger the tool
conversation.sendMessageAsync("What's the weather like in London?", callback)
```

The model will decide when to call the tool based on the conversation. The
results from the tool execution are automatically sent back to the model to
generate the final response.

#### Manual Tool Calling

By default, tool calls generated by the model are automatically executed by
LiteRT-LM and the results from the tool execution are automatically sent back to
the model to generate the next response.

If you want to manually execute tools and send results back to the model, you
can set `automaticToolCalling` in `ConversationConfig` to `false`.

```kotlin
val conversation = engine.createConversation(
    ConversationConfig(
        tools = listOf(
            tool(SampleOpenApiTool()),
        ),
        automaticToolCalling = false,
    )
)
```

If you disable automatic tool calling, you will need to manually execute tools
and send results back to the model in your application code. The `execute`
method of `OpenApiTool` will **not** be called automatically when
`automaticToolCalling` is set to `false`.

```kotlin
// Send a message that triggers a tool call.
val responseMessage = conversation.sendMessage("What's the weather like in London?")

// The model returns a Message with `toolCalls` populated.
if (responseMessage.toolCalls.isNotEmpty()) {
    val toolResponses = mutableListOf<Content.ToolResponse>()
    // There can be multiple tool calls in a single response.
    for (toolCall in responseMessage.toolCalls) {
        println("Model wants to call: ${toolCall.name} with arguments: ${toolCall.arguments}")

        // Execute the tool manually with your own logic. `executeTool` is just an example here.
        val toolResponseJson = executeTool(toolCall.name, toolCall.arguments)

        // Collect tool responses.
        toolResponses.add(Content.ToolResponse(toolCall.name, toolResponseJson))
    }

    // Use Message.tool to create the tool response message.
    val toolResponseMessage = Message.tool(Contents.of(toolResponses))

    // Send the tool response message to the model.
    val finalMessage = conversation.sendMessage(toolResponseMessage)
    println("Final answer: ${finalMessage.text}") // e.g., "The weather in London is 25c."
}
```

#### Example

To try out tool use, clone the repo and run with
[example/ToolMain.kt](../../../kotlin/kotlin/com/google/ai/edge/litertlm/example/ToolMain.kt):

```bazel
bazel run -c opt //kotlin/kotlin/com/google/ai/edge/litertlm/example:tool -- <abs_model_path>
```

### 7. Extra Template Context Variables

You can pass extra context variables to the prompt template for rendering.
This allows you to customize the model's behavior based on dynamic values.

`extraContext` is an optional `Map<String, Any>` that can be passed to
`sendMessage` and `sendMessageAsync`. These variables are merged with the extra
context provided in the `Preface` (if any), with keys in the message-level
context overwriting those in the `Preface`.

```kotlin
val extraContext = mapOf(
    "user_name" to "Alice",
    "enable_thinking" to true
)

// Synchronous
val response = conversation.sendMessage("Hello!", extraContext = extraContext)

// Asynchronous with Flow
conversation.sendMessageAsync("Hello!", extraContext = extraContext)
    .collect { ... }
```

These variables are used within the Jinja-style prompt templates, e.g.,
`{{ user_name }}` or `{% if enable_thinking %}`.

## Error Handling

API methods can throw `LiteRtLmNativeException` for errors from the native layer or
standard Kotlin exceptions like `IllegalStateException` for lifecycle issues.
Always wrap API calls in try-catch blocks. The `onError` callback in
`MessageCallback` will also report errors during asynchronous operations.
