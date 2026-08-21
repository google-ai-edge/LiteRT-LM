# Constrained Decoding in LiteRT-LM

LiteRT-LM supports constrained decoding, allowing you to enforce specific
structures on the model's output. This is particularly useful for tasks like:

-   **Function Calling**: Ensuring the model outputs a valid function call
    matching a specific schema.
-   **Structured Data Extraction**: Forcing the model to adhere to a specific
    format (e.g., specific regex patterns).
-   **Grammar Enforcement**: Using context-free grammars (via Lark) to guide
    generation.

This document explains how to enable, configure, and use constrained decoding in
your application.

There are two ways to use constrained decoding in LiteRT-LM:

1.  **Constrained Decoding for Tool Calling**: LiteRT-LM will read the tool
    declarations and apply constrained decoding to guarantee correct function
    calling syntax.
2.  **Custom Constrained Decoding**: Use LLGuidance or your own `Constraint`
    class to mask the vocabulary during sampling.

**Note**: You can only choose *one* of these options when constructing a
`Conversation` object.

## Constrained Decoding for Tool Calling

To enable *constrained decoding for tool calling*, enable it in the
`ConversationConfig` when creating your `Conversation` instance, which can be
done in the `ConversationConfig::Builder` by calling
`SetEnableConstrainedDecoding(true)`:

```cpp
#include "runtime/conversation/conversation.h"

ConversationConfig::Builder builder;
builder.SetEnableConstrainedDecoding(true);
auto config = builder.Build(*engine).value();
```

When the model generates function calls as part of its output, the function call
string is constrained to follow the function-calling syntax of the model.

## Custom Constrained Decoding

To use [LLGuidance](https://github.com/guidance-ai/llguidance/tree/main/) or an
external `Constraint` object, set the appropriate *Constraint Provider* in the
`ConversationConfig` when creating your `Conversation` instance.

Example for using LLGuidance:

```cpp
#include "runtime/conversation/conversation.h"

// Set a ConstraintProviderConfig in the ConversationConfig::Builder.
// This line sets the ConstraintProvider to LLGuidance with default settings.
ConversationConfig::Builder builder;
builder.SetConstraintProviderConfig(LlGuidanceConfig());
auto config = builder.Build(*engine).value();
```

Example for using your own `Constraint` objects:

```cpp
#include "runtime/conversation/conversation.h"

ConversationConfig::Builder builder;
builder.SetConstraintProviderConfig(ExternalConstraintConfig());
auto config = builder.Build(*engine).value();
```

### Constraint Providers

LiteRT-LM supports two different backends for constrained decoding, configured
via `ConstraintProviderConfig`:

1.  **LLGuidance (`LlGuidanceConfig`)**: Uses the
    [LLGuidance](https://github.com/guidance-ai/llguidance) library. Supports
    Regex, JSON Schema, and Lark grammars.
2.  **External (`ExternalConstraintConfig`)**: Allows passing a pre-constructed
    `Constraint` object per-request. Useful for custom C++ constraint
    implementations.

### Using Constraints in `SendMessage`

Once enabled, you can apply constraints to individual messages using the
`decoding_constraint` field in the `OptionalArgs` struct passed to `SendMessage`
or `SendMessageAsync`. This field is of type `std::optional<ConstraintArg>`.

#### LLGuidance Constraints

LLGuidance constraints can be specified as Regex, JSON Schema, or Lark grammars.

##### Regex Constraint

Constrain the output to match a regular expression.

```cpp
#include "runtime/components/constrained_decoding/llg_constraint_config.h"

// ...

LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kRegex;
// Example: Force output to be a sequence of 'a's followed by 'b's
constraint_arg.constraint_string = "a+b+";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

##### JSON Schema Constraint

Constrain the output to be a valid JSON object matching a schema.

```cpp
LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kJsonSchema;
// Example: Simple JSON object with a "name" field
constraint_arg.constraint_string = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string"}
  },
  "required": ["name"]
})";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

##### Lark Grammar Constraint

Constrain the output to follow a Lark grammar.

```cpp
LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kLark;
// Example: A simple calculator grammar
constraint_arg.constraint_string = R"(
    start: expr
    expr: atom
        | expr "+" atom
        | expr "-" atom
        | expr "*" atom
        | expr "/" atom
        | "(" expr ")"
    atom: /[0-9]+/
    WS: /[ \t\n\f]+/
    %ignore WS
)";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

#### External Constraints

If you have a custom implementation of the `Constraint` interface (e.g., a
highly specialized C++ state machine), you can use `ExternalConstraintArg`.

Prerequisite: You must have initialized `Conversation` with
`ExternalConstraintConfig`.

```cpp
// 1. Initialize with ExternalConstraintConfig
auto config = ConversationConfig::Builder()
                  .SetConstraintProviderConfig(ExternalConstraintConfig())
                  .Build(*engine)
                  .value();
auto conversation = Conversation::Create(*engine, config).value();

// 2. Create your custom constraint (must implement litert::lm::Constraint)
class MyCustomConstraint : public litert::lm::Constraint {
 public:
  // A simple custom state that tracks the current step.
  class MyState : public litert::lm::Constraint::State {
   public:
    explicit MyState(int step) : step_(step) {}
    int step() const { return step_; }

   private:
    int step_;
  };

  // A custom bitmap that allows only a single specified token.
  class SingleAllowedTokenBitmap : public litert::lm::Bitmap {
   public:
    explicit SingleAllowedTokenBitmap(int allowed_token)
        : allowed_token_(allowed_token) {}
    bool Get(int index) const override { return index == allowed_token_; }

   private:
    int allowed_token_;
  };

  std::unique_ptr<State> Start() const override {
    return std::make_unique<MyState>(0);
  }

  bool IsEnded(const State& state) const override {
    const auto& my_state = static_cast<const MyState&>(state);
    // Ends after generating 2 constrained tokens.
    return my_state.step() >= 2;
  }

  int GetVocabularySize() const override { return 32000; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(
      const State& state, int token) const override {
    const auto& my_state = static_cast<const MyState&>(state);
    return std::make_unique<MyState>(my_state.step() + 1);
  }

  absl::StatusOr<std::unique_ptr<litert::lm::Bitmap>> ComputeBitmap(
      const State& state) const override {
    const auto& my_state = static_cast<const MyState&>(state);
    if (my_state.step() == 0) {
      // In the first step, only allow token ID 42.
      return std::make_unique<SingleAllowedTokenBitmap>(42);
    } else {
      // In the second step, only allow token ID 99.
      return std::make_unique<SingleAllowedTokenBitmap>(99);
    }
  }
};
auto my_constraint = std::make_unique<MyCustomConstraint>();

// 3. Pass it to SendMessage
ExternalConstraintArg external_constraint;
external_constraint.constraint = std::move(my_constraint);

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = std::move(external_constraint)}
);
```

## API Reference

### `ConstraintProviderConfig`

A variant configuration passed to `ConversationConfig`.

-   `LlGuidanceConfig`: Configures LLGuidance.
    -   `eos_id`: Optional override for the End-of-Sequence token ID.
-   `ExternalConstraintConfig`: Empty struct (marker) to enable external
    constraints.

### `ConstraintArg`

A variant argument passed via `OptionalArgs` to `SendMessage`.

-   `LlGuidanceConstraintArg`:
    -   `constraint_type`: `kRegex`, `kJsonSchema`, or `kLark`.
    -   `constraint_string`: The pattern/schema/grammar string.
-   `ExternalConstraintArg`:
    -   `constraint`: `std::unique_ptr<Constraint>`. Ownership is transferred to
        the conversation object for that request.
