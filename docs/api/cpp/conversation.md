# LiteRT-LM Conversation API

[**`Conversation`**][Conversation] is a high-level API, representing a single,
stateful conversation with the LLM and is the recommended entry point for most
users. It internally manages a [`Session`][Session] and handles complex data
processing tasks. These tasks include maintaining the initial context, managing
tool definitions, preprocessing multimodal data, and applying Jinja prompt
templates with role-based message formatting.

## Conversation API Workflow

The typical lifecycle for using the Conversation API is:

1.  **Create an [`Engine`][Engine]**: Initialize a single [`Engine`][Engine]
    with the model path and configuration. This is a heavyweight object that
    holds the model weights.
2.  **Create a [`Conversation`][Conversation]**: Use the [`Engine`][Engine] to
    create one or more lightweight [`Conversation`][Conversation] objects.
3.  **Send Message**: Utilize the [`Conversation`][Conversation] object's
    methods to send messages to the LLM and receive responses, effectively
    enabling a chat-like interaction.

Below is the simplest way to send message and get model response. It is
recommended for most use cases. It mirrors
[Gemini Chat APIs](https://ai.google.dev/gemini-api/docs/text-generation#multi-turn-conversations).

-   [`SendMessage`][SendMessage]: A blocking call that takes user input and
    returns the complete model response.
-   [`SendMessageAsync`][SendMessageAsync]: A non-blocking call that streams the
    model's response back token-by-token through callbacks.

Here are example code snippet:

### Text only content

```cpp
#include "runtime/engine/engine.h"

// ...

// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create(model_path);
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    model_assets,
    /*backend=*/litert::lm::Backend::CPU);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
CHECK_OK(engine);

// 3. Create a Conversation
auto conversation_config = ConversationConfig::CreateDefault(**engine);
CHECK_OK(conversation_config)
absl::StatusOr<std::unique_ptr<Conversation>> conversation = Conversation::Create(**engine, *conversation_config);
CHECK_OK(conversation);

// 4. Send message to the LLM with blocking call.
absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
    Message{
        {"role", "user"},
        {"content", "What is the tallest building in the world?"}
    });
CHECK_OK(model_message);

// 5. Print the model message.
std::cout << *model_message << std::endl;

// 6. Send message to the LLM with asynchronous call
// where CreatePrintMessageCallback is a users implemented callback that would
// process the message once a chunk of message output is received.
std::stringstream captured_output;
(*conversation)->SendMessageAsync(
    Message{
        {"role", "user"},
        {"content", "What is the tallest building in the world?"}
    },
    CreatePrintMessageCallback(std::stringstream& captured_output)
);
// Wait until asynchronous finish or timeout.
*engine->WaitUntilDone(absl::Seconds(10));
```

<details>
<summary>

Example `CreatePrintMessageCallback`

</summary>

```cpp
absl::AnyInvocable<void(absl::StatusOr<Message>)> CreatePrintMessageCallback(
    std::stringstream& captured_output) {
  return [&captured_output](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      std::cout << message.status().message() << std::endl;
      return;
    }
    if (message->empty()) {
      std::cout << std::endl << std::flush;
      return;
    }
    ABSL_CHECK_OK(PrintMessage(*message, captured_output,
                               /*streaming=*/true));
  };
}

absl::Status PrintMessage(const Message& message,
                              std::stringstream& captured_output,
                              bool streaming = false) {
  if (message["content"].is_array()) {
    for (const auto& content : message["content"]) {
      if (content["type"] == "text") {
        captured_output << content["text"].get<std::string>();
        std::cout << content["text"].get<std::string>();
      }
    }
    if (!streaming) {
      captured_output << std::endl << std::flush;
      std::cout << std::endl << std::flush;
    } else {
      captured_output << std::flush;
      std::cout << std::flush;
    }
  } else if (message["content"]["text"].is_string()) {
    if (!streaming) {
      captured_output << message["content"]["text"].get<std::string>()
                      << std::endl
                      << std::flush;
      std::cout << message["content"]["text"].get<std::string>() << std::endl
                << std::flush;
    } else {
      captured_output << message["content"]["text"].get<std::string>()
                      << std::flush;
      std::cout << message["content"]["text"].get<std::string>() << std::flush;
    }
  } else {
    return absl::InvalidArgumentError("Invalid message: " + message.dump());
  }
  return absl::OkStatus();
}
```

</details>

### Multimodal data content

```cpp
// To use multimodality, the engine must be created with vision and audio
// backend depending on the multimodality to be used
auto engine_settings = EngineSettings::CreateDefault(
    model_assets,
    /*backend=*/litert::lm::Backend::CPU,
    /*vision_backend*/litert::lm::Backend::GPU,
    /*audio_backend*/litert::lm::Backend::CPU,
);

// The same steps to create Engine and Conversation as above...

// Send message to the LLM with image data.
absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
    Message{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "Describe the following image: "}},
          {{"type", "image"}, {"path", "/file/path/to/image.jpg"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

// Send message to the LLM with audio data.
model_message = (*conversation)->SendMessage(
    Message{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "Transcribe the audio: "}},
          {{"type", "audio"}, {"path", "/file/path/to/audio.wav"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

// The content can include multiple image or audio data.
model_message = (*conversation)->SendMessage(
    Message{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "First briefly describe the two images "}},
          {{"type", "image"}, {"path", "/file/path/to/image1.jpg"}},
          {{"type", "text"}, {"text", "and "}},
          {{"type", "image"}, {"path", "/file/path/to/image2.jpg"}},
          {{"type", "text"}, {"text", " then transcribe the content in the audio"}},
          {{"type", "audio"}, {"path", "/file/path/to/audio.wav"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

```

### Use Conversation with Tools

Please refer to [Tool Use](./tool_use.md) for detailed Tool Usage with
Conversation API

## Components in Conversation

[`Conversation`][Conversation] could be regarded as a delegate for users to
maintain [`Session`][Session] and complicated data processing before sending the
data to Session.

### I/O Types

The core input and output format for the Conversation API is
[`Message`][Message]. Currently, this is implemented as
[`Message`][Message], which is a type alias for
[`ordered_json`][ordered_json], a flexible nested key-value data structure.

The [`Conversation`][Conversation] API operates on a message-in-message-out
basis, mimicking a typical chat experience. The flexibility of
[`Message`][Message] allows users to include arbitrary fields as needed by
specific prompt templates or LLM models, enabling LiteRT-LM to support a wide
variety of models.

While there isn't a single rigid standard, most prompt templates and models
expect [`Message`][Message] to follow conventions similar to those used in the
[Gemini API Content](https://ai.google.dev/api/caching#Content) or the
[OpenAI Message structure](https://platform.openai.com/docs/api-reference/chat/create).

[`Message`][Message] must contain `role`, representing who the message is sent
from. `content` can be as simple as a text string.

```json
{
  "role": "model", // Represent who the message is sent from.
  "content": "Hello World!" // Naive text only content.
}
```

For multimodal data input, `content` is a list of `part`. Again `part` is not a
predefined data structure but a
[ordered key-value pair data type][ordered_json]. The specific fields depend on
what the prompt template and the model expect.

```json
{
  "role": "user",
  "content": [  // Multimodal content.
    // Now the content is composed of parts
    {
      "type": "text",
      "text": "Describe the image in details: "
    },
    {
      "type": "image",
      "path": "/path/to/image.jpg"
    }
  ]
}
```

For multimodal `part`, we support the following format handled by
[`data_utils.h`](https://github.com/google-ai-edge/LiteRT-LM/blob/e5ead3aa409b8e631111d487aaa5e54e91ead747/runtime/conversation/model_data_processor/data_utils.h#L26-L57)

```json
{
  "type": "text",
  "text": "this is a text"
}

{
  "type": "image",
  "path": "/path/to/image.jpg"
}

{
  "type": "image",
  "blob": "base64 encoded image bytes as string",
}

{
  "type": "audio",
  "path": "/path/to/audio.wav"
}

{
  "type": "audio",
  "blob": "base64 encoded audio bytes as string",
}
```

### Model Data Processor

The [`ModelDataProcessor`][ModelDataProcessor] is a crucial model-specific
component responsible for converting the generic [`Message`][Message] format
into the [`InputData`][InputData] type required by the [`Session`][Session], and
convert from [`Session`][Session]'s [`Responses`][Responses] to
[`Message`][Message]. It functions similarly to
[`Hugging Face Multi-modal processors`](https://huggingface.co/docs/transformers/en/main_classes/processors#transformers.ProcessorMixin),
handling all necessary data preprocessing before the main LLM model execution.

The specific responsibilities of a [`ModelDataProcessor`][ModelDataProcessor]
vary based on the model's capabilities and the expected prompt template format.
For instance:

*   **Multimodal Models (e.g., `Gemma3N`):** The
    [`Gemma3DataProcessor`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/model_data_processor/gemma3_data_processor.h)
    includes image and audio preprocessors.
*   **Function Calling Models (e.g., `Qwen3`):** The
    [`Qwen3DataProcessor`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/model_data_processor/qwen3_data_processor.h)
    handle `tool_calls` and `tool_response` content.

Each [`ModelDataProcessor`][ModelDataProcessor] is associated with a
[`DataProcessorConfig`][DataProcessorConfig] for initialization and can accept
[`DataProcessorArguments`][DataProcessorArguments] during message sending.

-   **[`DataProcessorConfig`][DataProcessorConfig]:** This configuration is used
    to initialize the [`ModelDataProcessor`][ModelDataProcessor]. It corresponds
    to the [`LlmModelType`][LlmModelType] protobuf stored in the
    [model file metadata](https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/proto/llm_metadata.proto#L91).
    The [`DataProcessorConfig`][DataProcessorConfig] provides default values for
    any fields not specified in the metadata's [`LlmModelType`][LlmModelType].

-   **[`DataProcessorArguments`][DataProcessorArguments]:** This allows passing
    turn-specific arguments to the [`ModelDataProcessor`][ModelDataProcessor]
    via [`SendMessage`][SendMessage] or [`SendMessageAsync`][SendMessageAsync].
    These arguments can alter the processor's behavior for a single turn.
    However, most models do not require such turn-specific arguments, so this is
    often an empty struct.

To support a new LLM model type, you will typically need to implement a new
[`ModelDataProcessor`][ModelDataProcessor] to encapsulate its unique data
handling logic. See
[Add ModelDataProcessor for a new LLM model type](#add-modeldataprocessor-for-new-llm-model-type)
for detailed instructions.

### Prompt Template

To maintain flexibility for variant models, [PromptTemplate][PromptTemplate] is
implemented as a thin wrapper around [Minja](https://github.com/google/minja).
Minja is a C++ implementation of the [Jinja template engine][Jinja], which
processes JSON input to generate formatted prompts.

The [Jinja template engine][Jinja] is a widely adopted format for LLM prompt
templates. Here are a few examples:

-   [`google-gemma-3n-e2b-it.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/google-gemma-3n-e2b-it.jinja)
-   [`HuggingFaceTB-SmolLM3-3B.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/HuggingFaceTB-SmolLM3-3B.jinja)
-   [`Qwen-Qwen3-0.6B.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/Qwen-Qwen3-0.6B.jinja)

The [Jinja template engine][Jinja] format should strictly match the structure
expected by the instruction-tuned model. Typically, model releases include the
standard Jinja template to ensure proper model usage.

The Jinja template used by the model will be provided by the model file
metadata.

> [!NOTE] A subtle change in prompt because of incorrect formatting can lead to
> significant model degradation. As reported in [Quantifying Language Models'
> Sensitivity to Spurious Features in Prompt Design or: How I learned to start
> worrying about prompt formatting](https://arxiv.org/abs/2310.11324)

### Preface

[`Preface`][Preface] sets the initial context for the conversation. It can
include initial messages, tool definitions, and any other background information
the LLM needs to start the interaction. This achieves functionality similar to
the
[`Gemini API system instruction`](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions)
and [`Gemini API Tools`](https://ai.google.dev/api/caching#Tool)

[Preface][Preface] contains the following fields

-   `messages` The messages in the preface. The messages provided the initial
    background for the conversation. For example, the messages can be the
    conversation history, prompt engineering system instructions, few-shot
    examples, etc.

-   `tools` The tools the model can use in the conversation. The format of tools
    is again not fixed, but mostly follows
    [`Gemini API FunctionDeclaration`](https://ai.google.dev/api/caching#FunctionDeclaration).

-   `extra_context` The extra context that keeps the extensibility for models to
    customize its required context information to start a conversation. For
    examples,

    -   `enable_thinking` for models with thinking mode, e.g.
        [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) or
        [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

Example preface to provide initial system instruction, tools and disable
thinking mode.

```cpp
Preface preface = JsonPreface({
  .messages = {
      {"role", "system"},
      {"content", {"You are a model that can do function calling."}}
    },
  .tools = {
    {
      {"name", "get_weather"},
      {"description", "Returns the weather for a given location."},
      {"parameters", {
        {"type", "object"},
        {"properties", {
          {"location", {
            {"type", "string"},
            {"description", "The location to get the weather for."}
          }}
        }},
        {"required", {"location"}}
      }}
    },
    {
      {"name", "get_stock_price"},
      {"description", "Returns the stock price for a given stock symbol."},
      {"parameters", {
        {"type", "object"},
        {"properties", {
          {"stock_symbol", {
            {"type", "string"},
            {"description", "The stock symbol to get the price for."}
          }}
        }},
        {"required", {"stock_symbol"}}
      }}
    }
  },
  .extra_context = {
    {"enable_thinking": false}
  }
});
```

### History

[Conversation][Conversation] maintains a list of all [Message][Message]
exchanges within the session. This history is crucial for prompt template
rendering, as the jinja prompt template typically requires the entire
conversation history to generate the correct prompt for the LLM.

However, the LiteRT-LM [Session][Session] is stateful, meaning it processes
inputs incrementally. To bridge this gap, [Conversation][Conversation] generates
the necessary incremental prompt by rendering the prompt template twice: once
with the history up to the previous turn, and once including the current
message. By comparing these two rendered prompts, it extracts the new portion to
be sent to the [Session][Session].

### ConversationConfig

[`ConversationConfig`][ConversationConfig] is used to initialize a
[`Conversation`][Conversation] instance. You can create this configuration in a
couple of ways:

1.  **From an [`Engine`][Engine]:** This method uses the default
    [`SessionConfig`][SessionConfig] associated with the engine.
2.  **From a specific [`SessionConfig`][SessionConfig]:** This allows for more
    fine-grained control over the session settings.

Beyond session settings, you can further customize the
[`Conversation`][Conversation] behavior within the
[`ConversationConfig`][ConversationConfig]. This includes:

*   Providing a [`Preface`][Preface].
*   Overwriting the default [`PromptTemplate`][PromptTemplate].
*   Overwriting the default [`DataProcessorConfig`][DataProcessorConfig].

These overwrites are particularly useful for fine-tuned models, which might
require different configurations or prompt templates than the base model they
were derived from.

### MessageCallback

[`MessageCallback`][SendMessageAsync] is the callback function that users should
implement when they use asynchronous [`SendMessageAsync`][SendMessageAsync]
method.

The callback signature is `absl::AnyInvocable<void(absl::StatusOr<Message>)>`.
This function is triggered under the following conditions:

*   When a new chunk of the [`Message`][Message] is received from the Model.
*   If an error occurs during LiteRT-LM's message processing.
*   Upon completion of the LLM's inference, the callback is triggered with an
    empty [`Message`][Message] (e.g., `Message()`) to signal the end of the
    response.

Refer to the [Step 6 asynchronous call](#text-only-content) for an example
implementation.

> [!IMPORTANT] The [`Message`][Message] received by the callback contains only
> the latest chunk of the model's output, not the entire message history.

For example, if the complete model response expected from a blocking
[`SendMessage`][SendMessage] call would be:

```json
{
  "role": "model",
  "content": [
    "type": "text",
    "text": "Hello World!"
  ]
}
```

The callback in [`SendMessageAsync`][SendMessageAsync] might be invoked multiple
times, each time with a subsequent piece of the text:

```json
// 1st Message
{
  "role": "model",
  "content": [
    "type": "text",
    "text": "He"
  ]
}

// 2nd Message
{
  "role": "model",
  "content": [
    "type": "text",
    "text": "llo"
  ]
}

// 3rd Message
{
  "role": "model",
  "content": [
    "type": "text",
    "text": " Wo"
  ]
}

// 4th Message
{
  "role": "model",
  "content": [
    "type": "text",
    "text": "rl"
  ]
}

// 5th Message
{
  "role": "model",
  "content": [
    "type": "text",
    "text": "d!"
  ]
}
```

The implementer is responsible for accumulating these chunks if the complete
response is needed during the asynchronous stream. Alternatively, the full
response will be available as the last entry in the [`History`](#history) once
the asynchronous call is complete.

## Add ModelDataProcessor for new LLM model type

[WIP] 🚧

[conversation]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/conversation.h#L160 "litert::lm::Conversation"
[session]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/engine/engine.h#L71 "litert::lm:Session"
[engine]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/engine/engine.h#L65 "litert::lm::Engine"
[ModelDataProcessor]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/conversation.h#L205 "litert::lm::ModelDataProcessor"
[InputData]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/engine/io_types.h#L154 "litert::lm::InputData"
[Responses]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/engine/io_types.h#L170 "litert::lm::Responses"
[sendmessage]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/conversation.h#L180 "Conversation::SendMessage"
[sendmessageasync]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/conversation.h#L205 "Conversation::SendMessageAsync"
[Jinja]: https://jinja.palletsprojects.com/en/stable/ "jinja prompt template"
[PromptTemplate]: https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/prompt_template.h "litert::lm::PromptTemplate"
[message]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/io_types.h#L28 "litert::lm::Message"

[ordered_json]: https://json.nlohmann.me/api/ordered_json/ "ordered_json"
[preface]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/io_types.h#L48 "litert::lm::Preface"
[ConversationConfig]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/conversation.h#L44 "litert::lm::ConversationConfig"
[SessionConfig]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/engine/engine_settings.h#L145 "litert::lm::SessionConfig"
[DataProcessorConfig]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/model_data_processor/config_registry.h#L29 "litert::lm::DataProcessorConfig"
[DataProcessorArguments]: https://github.com/google-ai-edge/LiteRT-LM/blob/63f7dec93ac85560e64194a00b5d7c407de40846/runtime/conversation/model_data_processor/config_registry.h#L38 "litert::lm::DataProcessorArguments"
[LlmModelType]: https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/proto/llm_model_type.proto "litert.lm.proto.LlmModelType"
