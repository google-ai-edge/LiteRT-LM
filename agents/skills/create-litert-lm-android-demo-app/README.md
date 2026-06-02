# LiteRT-LM Android Demo App Skill

This directory contains a specialized agent skill that guides AI developers to
successfully implement, build, and package a standalone
**LiteRT-LM Android demo application** from scratch on the first pass.

## Installation

To register and automatically load this skill in your agent sessions:

1.  **Locate your agent's skills directory** (the folder where your agent loads
    custom instructions or skills).
2.  **Copy the entire `create-litert-lm-android-demo-app` folder** directly into
    that skills directory:

    ```bash
    # Replace the destination path with your active agent skills directory
    cp -r agents/skills/create-litert-lm-android-demo-app /path/to/your/agent/skills/
    ```

3.  Once copied, the agent will automatically detect and load the skill when
    resolving related user requests.

## Usage

To trigger a skill, instruct the agent under your active session with a prompt
specifying the parameters of the target task. The agent will recognize the
matching skill files and automatically follow the defined execution process.

### Example Prompt (LiteRT-LM Android Demo App)

```text
Please create a LiteRT-LM Android demo app

root: ~/litert_lm_litert_lm_maven_integration
Maven Integration scenario
Target: pixel 10
model: gemma 4
```
