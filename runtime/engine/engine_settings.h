#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_

#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Settings used for initializing LiteRT LM.
// This class encapsulates the model-specific settings that are used for
// initializing the LiteRT LM. These settings are typically fixed for a given
// model and are not expected to change during the inference process.
// The model assets are required to initialize the LiteRT LM.
// TODO(b/397975034) Add overloading << operator for debugging.
class EngineSettings {
 public:
  explicit EngineSettings(const LlmExecutorSettings& executor_settings)
      : main_executor_settings_(executor_settings) {}

  const LlmExecutorSettings& GetMainExecutorSettings() const {
    return main_executor_settings_;
  }

 private:
  // Settings for the main executor.
  LlmExecutorSettings main_executor_settings_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
