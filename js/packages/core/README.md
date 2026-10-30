# LiteRT-LM JS API

The JavaScript API of LiteRT-LM for running LLMs in the browser. This is an
early preview that supports text-in / text-out running in WebGPU.
[LiteRT-LM](https://www.google.com/url?sa=D&q=https%3A%2F%2Fai.google.dev%2Fedge%2Flitert-lm%2F)
is the **production-ready** orchestration layer to run LLMs with LiteRT,
engineered for **high-performance**, **cross-platform** execution.

-   **Documentation**: [https://ai.google.dev/edge/litert-lm/js](https://ai.google.dev/edge/litert-lm/js)
-   **GitHub**: [https://github.com/google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)

## Supported Models

The LiteRT-LM JS API supports the models in the
[Web LLM Models](https://huggingface.co/collections/litert-community/web-llm-models)
collection on Hugging Face, including Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.
Load one by the URL of its model file, or by its repo with a `hf://` URI, which
chooses the repo's web model file (see [Load Models from Hugging Face](#load-models-from-hugging-face)):

```ts
const engine = await Engine.create({
  // The model's Hugging Face repo
  model: 'hf://litert-community/gemma-4-E2B-it-litert-lm',
  // Or the URL of a model file, on Hugging Face or anywhere else
  // model: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm',
});
```

Some of these models are gated: Hugging Face only lets users who have accepted
the model's license download them. To load one, pass the access token of such a
user (see [Gated Models](#gated-models)).

## Introduction

Here is a sample REPL chat app built with the JavaScript API:

```ts
<div id="out" style="white-space: pre-wrap; font-family: monospace;"></div>
<input id="in" onkeydown="if(event.key === 'Enter') repl(this)">

<script type="module">
  import { Engine } from 'https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm';
  const engine = await Engine.create({
    // Load the Gemma 4 E2B model
    model: 'hf://litert-community/gemma-4-E2B-it-litert-lm',
    // Or use the E4B model by swapping in this line
    // model: 'hf://litert-community/gemma-4-E4B-it-litert-lm',
  });
  const chat = await engine.createConversation();

  window.repl = async (el) => {
    const text = el.value;
    el.value = ''; // Clear immediately
    out.append(`\n>>> ${text}\nAI: `);

    for await (const chunk of chat.sendMessageStreaming(text)) {
      out.append(chunk.content[0].text);
    }
  };
</script>
```

## Getting Started

LiteRT-LM is available as an npm package. You can install the latest version
from npm or directly import it from a CDN:

```shell
# From npm
npm i --save @litert-lm/core

# From a CDN (in your JavaScript file)
import * as litertlm from 'https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm';
```

### Initialize the Engine

The `Engine` is the entry point to the API. It handles model loading, session
creation, and resource management. Remember to `delete` the engine to release
resources when the model is no longer needed.

**Note:** Initializing the engine can take several seconds to load the model.

```ts
import {Engine, EngineSettings} from '@litert-lm/core';

const engineSettings = {
  // A URL, a hf:// URI, a ReadableStream, or a Blob
  model: 'url/path/to/model.litertlm',

  // You can configure context length and other settings here
  mainExecutorSettings: {
    maxNumTokens: 8192,
  },
} satisfies EngineSettings;

const engine = await Engine.create(engineSettings);

// ... Use the engine to create a conversation ...

// Delete the engine when done.
await engine.delete();
```

### Load Models from Hugging Face

A `hf://<org>/<repo>[@<revision>][/<path>]` URI loads a model from the Hugging
Face Hub. Without a path, LiteRT-LM chooses the model file in the repo for the
engine's backend. If several files match, it loads one and logs a warning that
lists the others, which you can load by giving their path:

```ts
// The repo's web model file.
model: 'hf://litert-community/gemma-4-E4B-it-litert-lm'

// A specific file at a specific revision (branch, tag, or commit).
model: 'hf://litert-community/gemma-4-E4B-it-litert-lm@main/gemma-4-E4B-it-gpu.litertlm'
```

The revision is resolved to its current commit before anything is downloaded,
so the file that's downloaded always matches its SHA-256.

### Gated Models

Hugging Face only lets users who have accepted a gated model's license download
it. To load a gated model, pass the access token of such a user, for example
from [Sign in with Hugging Face](https://huggingface.co/docs/hub/oauth), in a
`ModelSource`:

```ts
const engine = await Engine.create({
  model: {
    url: 'hf://litert-community/Gemma3-1B-IT',
    accessToken: userAccessToken,
  },
});
```

**Caution:** Don't embed your own access token in a web page: everyone who
loads the page can read it.

### Create a Conversation

Once the engine is initialized, create a `Conversation` instance. You can
provide a `ConversationConfig` to customize its behavior.

```ts
const conversation = await engine.createConversation({
  preface: {
    messages: [
      {role: 'system', content: 'You are a helpful assistant'}
    ]
  }
});

conversation.sendMessage({
  role: 'user',
  content: 'Write a poem',
});
```

### Send Messages

You can send messages with or without streaming.

#### Non-Streaming Example

```ts
// Simple string input
let response = await conversation.sendMessage("What is the capital of France?");
console.log(response.content[0].text);

// Or with full message structure
response = await conversation.sendMessage({role: 'user', content: '...'});
```

#### Streaming Example

```ts
// sendMessageStreaming returns a ReadableStream of response chunks
const stream = conversation.sendMessageStreaming('Tell me a long story.');

for await (const chunk of stream) {
  // Chunks are Records containing pieces of the response
  for (const item of chunk.content) {
    if (item.type === 'text') {
      console.log(item.text);
    }
  }
}
```

#### Cancel Generation

You can cancel an ongoing generation explicitly by calling `cancel()` on the
`Conversation` instance:

```ts
// Cancel any ongoing generation
conversation.cancel();
```

If you are streaming the response, exiting the `for await...of` loop early (such
as with `break`) will also automatically cancel the ongoing generation:

```ts
for await (const chunk of stream) {
  if (shouldStop()) {
    break; // Cancels the stream and underlying generation
  }
}
```
