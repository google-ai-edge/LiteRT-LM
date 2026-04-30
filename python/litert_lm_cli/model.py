# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for litert-lm models."""

import dataclasses
import glob
import importlib.util
import inspect
import json
import mimetypes
import os
import pathlib
import traceback

import click
import prompt_toolkit
from prompt_toolkit import key_binding

import litert_lm

try:
  # pylint: disable=g-import-not-at-top
  from litert_lm.adb import adb_benchmark  # pytype: disable=import-error
  from litert_lm.adb import adb_engine  # pytype: disable=import-error

  _HAS_ADB = True
except ImportError:
  _HAS_ADB = False


def get_attachment_type(path: str) -> str:
  """Returns the attachment type (audio or image) from the file path.

  Args:
    path: Path to the attachment.

  Returns:
    'audio' or 'image'.

  Raises:
    ValueError: If the file type cannot be determined or is unsupported.
  """
  mime_type, _ = mimetypes.guess_type(path)
  if mime_type:
    if mime_type.startswith("audio/"):
      return "audio"
    elif mime_type.startswith("image/"):
      return "image"
    else:
      raise ValueError(f"Unsupported attachment type for '{path}': {mime_type}")
  else:
    raise ValueError(f"Could not determine file type for attachment '{path}'.")


def load_preset(preset: str):
  """Loads a preset file and returns the tools, messages and extra_context."""
  click.echo(click.style(f"Loading preset from {preset}:", dim=True))
  if not os.path.exists(preset):
    click.echo(click.style(f"Preset file not found: {preset}", fg="red"))
    return None, None, None

  spec = importlib.util.spec_from_file_location("user_tools", preset)
  if not spec or not spec.loader:
    click.echo(click.style(f"Failed to load tools from {preset}", fg="red"))
    return None, None, None

  user_tools = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(user_tools)

  tools = getattr(user_tools, "tools", None)
  if tools is None:
    tools = [
        obj
        for name, obj in inspect.getmembers(user_tools, inspect.isfunction)
        if obj.__module__ == "user_tools"
    ]

  messages = None
  system_instruction = getattr(user_tools, "system_instruction", None)
  if system_instruction:
    click.echo(
        click.style(f"- System instruction: {system_instruction}", dim=True)
    )
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}],
    }]

  click.echo(click.style("- Tools:", dim=True))
  for tool in tools:
    click.echo(
        click.style(f"  - {getattr(tool, '__name__', str(tool))}", dim=True)
    )

  extra_context = getattr(user_tools, "extra_context", None)
  if extra_context:
    click.echo(click.style(f"- Extra context: {extra_context}", dim=True))

  return tools, messages, extra_context


class LoggingToolEventHandler(litert_lm.ToolEventHandler):
  """Log tool call and tool response events."""

  def __init__(self, model):
    self.model = model

  def approve_tool_call(self, tool_call):
    """Logs a tool call."""
    if self.model.active_channel is not None:
      click.echo("\n", nl=False)
      self.model.active_channel = None
    click.echo(
        click.style(
            f"[tool_call] {json.dumps(tool_call['function'])}", fg="green"
        )
    )
    return True

  def process_tool_response(self, tool_response):
    """Logs a tool response."""
    click.echo(
        click.style(f"[tool_response] {json.dumps(tool_response)}", fg="green")
    )
    return tool_response


def _parse_backend(backend: str) -> litert_lm.Backend:
  """Parses the backend string and returns the corresponding Backend enum."""
  backend_lower = backend.lower()
  if backend_lower == "gpu":
    return litert_lm.Backend.GPU
  return litert_lm.Backend.CPU


@dataclasses.dataclass
class Model:
  """Represents a LiteRT-LM model.

  Attributes:
    model_id: The ID of the model.
    model_path: The local path to the model file.
    active_channel: The name of the currently active channel, or None if default
      text is being printed.
  """

  model_id: str
  model_path: str
  active_channel: str | None = None

  def exists(self) -> bool:
    """Returns True if the model file exists locally."""
    return os.path.isfile(self.model_path)

  def to_str(self) -> str:
    """Returns a string representation of the model."""
    return self.model_id

  def run_interactive(
      self,
      is_android: bool = False,
      backend: str = "cpu",
      preset: str | None = None,
      prompt: str | None = None,
      enable_speculative_decoding: bool | None = None,
      no_template: bool = False,
      max_num_tokens: int | None = None,
      filter_channel_content_from_kv_cache: bool = False,
      vision_backend: str | None = None,
      audio_backend: str | None = None,
      attachments: tuple[str, ...] = (),
      top_k: int | None = None,
      top_p: float | None = None,
      temperature: float | None = None,
      seed: int | None = None,
  ):
    """Runs the model interactively or with a single prompt.

    Args:
      is_android: Whether to run the model on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
      preset: Path to a Python file containing tool functions and system
        instructions.
      prompt: A single prompt to run once and exit.
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default.
      no_template: Interact with the model directly without applying prompt
        templates or stripping stop tokens.
      max_num_tokens: Maximum number of tokens for the KV cache.
      filter_channel_content_from_kv_cache: Whether to filter channel content
        from the KV cache.
      vision_backend: The hardware backend used for vision encoding.
      audio_backend: The hardware backend used for audio encoding.
      attachments: A tuple of paths to attachments.
      top_k: The number of top logits used during sampling.
      top_p: The cumulative probability threshold for nucleus sampling.
      temperature: The temperature to use for sampling.
      seed: The seed to use for randomization.
    """
    if not self.exists():
      click.echo(
          click.style(
              f"Could not find {self.to_str()} locally in {self.model_path}.",
              fg="red",
          )
      )
      return

    try:
      backend_val = _parse_backend(backend)
      vision_backend_val = (
          _parse_backend(vision_backend) if vision_backend else None
      )
      audio_backend_val = (
          _parse_backend(audio_backend) if audio_backend else None
      )

      sampler_config = None
      if (
          top_k is not None
          or top_p is not None
          or temperature is not None
          or seed is not None
      ):
        sampler_config = litert_lm.SamplerConfig(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
        )

      if is_android:
        if not _HAS_ADB:
          raise ImportError("litert_lm.adb dependencies are not available.")
        engine_cm = adb_engine.AdbEngine(
            self.model_path,
            backend=backend_val,
            max_num_tokens=max_num_tokens,
            vision_backend=vision_backend_val,
            audio_backend=audio_backend_val,
        )
      else:
        engine_cm = litert_lm.Engine(
            self.model_path,
            backend=backend_val,
            enable_speculative_decoding=enable_speculative_decoding,
            max_num_tokens=max_num_tokens,
            vision_backend=vision_backend_val,
            audio_backend=audio_backend_val,
        )

      with engine_cm as engine:
        if no_template:
          runner_cm = engine.create_session(
              apply_prompt_template=False, sampler_config=sampler_config
          )
        else:
          tools = None
          messages = None
          extra_context = None
          if preset:
            tools, messages, extra_context = load_preset(preset)
            if tools is None and messages is None and extra_context is None:
              return

          handler = LoggingToolEventHandler(self) if tools else None

          runner_cm = engine.create_conversation(
              tools=tools,
              messages=messages,
              tool_event_handler=handler,
              extra_context=extra_context,
              filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
              sampler_config=sampler_config,
          )

        with runner_cm as runner:
          if prompt:
            if isinstance(runner, litert_lm.AbstractSession):
              self._execute_raw_prompt(runner, prompt)
            elif isinstance(runner, litert_lm.AbstractConversation):
              self._execute_prompt(runner, prompt, attachments=attachments)
            return

          click.echo(
              click.style(
                  "[enter] submit | [ctrl+j] newline | [ctrl+c] clear/exit",
                  fg="cyan",
              )
          )
          click.echo()

          history_path = os.path.join(
              os.path.expanduser("~"), ".litert-lm", "history"
          )
          os.makedirs(os.path.dirname(history_path), exist_ok=True)

          prompt_session = prompt_toolkit.PromptSession(
              history=prompt_toolkit.history.FileHistory(history_path),
              key_bindings=self._create_keybindings(),
          )

          is_first_prompt = True
          while True:
            try:
              user_prompt = prompt_session.prompt(
                  prompt_toolkit.ANSI(click.style("> ", fg="green", bold=True)),
                  multiline=True,
                  # Start the new line in the beginning of line. This makes
                  # copying respecting the text.
                  prompt_continuation=lambda width, line_number, is_soft_wrap: (
                      ""
                  ),
              )
              if not user_prompt:
                continue

              if isinstance(runner, litert_lm.AbstractSession):
                self._execute_raw_prompt(
                    runner,
                    user_prompt,
                )
              elif isinstance(runner, litert_lm.AbstractConversation):
                if is_first_prompt:
                  self._execute_prompt(
                      runner, user_prompt, attachments=attachments
                  )
                  is_first_prompt = False
                else:
                  self._execute_prompt(runner, user_prompt)

            except EOFError:
              break
            except KeyboardInterrupt:
              # Catch Ctrl+C at the input prompt
              click.echo()
              continue
            except Exception:  # pylint: disable=broad-exception-caught
              click.echo(click.style("Error during inference", fg="red"))
              traceback.print_exc()

    except Exception:  # pylint: disable=broad-exception-caught
      click.echo(click.style("An error occurred", fg="red"))
      traceback.print_exc()

  def _execute_prompt(
      self,
      conversation: litert_lm.AbstractConversation,
      prompt: str,
      attachments: tuple[str, ...] = (),
  ):
    """Executes a single prompt and prints the result."""
    self.active_channel = None

    if attachments:
      content = []
      for path in attachments:
        abs_path = os.path.abspath(path)
        content.append(
            {"type": get_attachment_type(abs_path), "path": abs_path}
        )

      if prompt:
        content.append({"type": "text", "text": prompt})

      stream = conversation.send_message_async({
          "role": "user",
          "content": content,
      })
    else:
      stream = conversation.send_message_async(prompt)

    try:
      for chunk in stream:
        # Handle regular content
        content_list = chunk.get("content", [])
        for item in content_list:
          if item.get("type") == "text":
            if self.active_channel is not None:
              click.echo()
              self.active_channel = None
            click.echo(click.style(item.get("text", ""), fg="yellow"), nl=False)

        # Handle channels
        channels = chunk.get("channels", {})
        for channel_name, channel_content in channels.items():
          if self.active_channel != channel_name:
            if self.active_channel is not None:
              click.echo()
            click.echo(click.style(f"[{channel_name}] ", fg="blue"), nl=False)
            self.active_channel = channel_name
          click.echo(click.style(channel_content, fg="yellow"), nl=False)
      if self.active_channel is not None:
        click.echo()
      else:
        click.echo()
    except KeyboardInterrupt:
      conversation.cancel_process()
      # Empty the iterator queue.
      # This ensures we don't throw away StopIteration.
      for _ in stream:
        pass
      click.echo(click.style("\n[Generation cancelled]", dim=True))

  def _execute_raw_prompt(
      self, session: litert_lm.AbstractSession, prompt: str
  ):
    """Executes a single raw prompt and prints the result."""
    session.run_prefill([prompt])
    stream = session.run_decode_async()
    try:
      for chunk in stream:
        if chunk.texts:
          click.echo(click.style(chunk.texts[0], fg="yellow"), nl=False)
      click.echo()
    except KeyboardInterrupt:
      # Empty the iterator queue.
      for _ in stream:
        pass
      click.echo(click.style("\n[Generation cancelled]", dim=True))

  def _create_keybindings(self) -> key_binding.KeyBindings:
    """Creates keybindings for the interactive prompt."""
    kb = key_binding.KeyBindings()

    # Key binding for sending the prompt.
    @kb.add("enter")
    def _handle_enter(event):
      buffer = event.current_buffer
      if buffer.text.strip():
        buffer.validate_and_handle()

    # Key binding for new line. Note that terminal cannot take
    # "shift+enter", and "ctrl+enter"
    @kb.add("c-j")  # standard terminal convention.
    @kb.add("escape", "enter")  # alt+enter and esc+enter
    def _handle_newline(event):
      event.current_buffer.insert_text("\n")

    # Key binding for clearing input or exiting.
    @kb.add("c-c")
    def _handle_clear_or_exit(event):
      buffer = event.current_buffer
      if buffer.text:
        buffer.text = ""
      else:
        event.app.exit(exception=EOFError)

    return kb

  def benchmark(
      self,
      prefill_tokens: int = 256,
      decode_tokens: int = 256,
      is_android: bool = False,
      backend: str = "cpu",
      enable_speculative_decoding: bool | None = None,
  ):
    """Benchmarks the model.

    Args:
      prefill_tokens: The number of tokens to prefill.
      decode_tokens: The number of tokens to decode.
      is_android: Whether to run the benchmark on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default.
    """
    if not self.exists():
      click.echo(
          click.style(
              f"Could not find {self.to_str()} locally in {self.model_path}.",
              fg="red",
          )
      )
      return

    try:
      backend_val = _parse_backend(backend)

      if is_android:
        if not _HAS_ADB:
          raise ImportError("litert_lm.adb dependencies are not available.")
        benchmark_obj = adb_benchmark.AdbBenchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            cache_dir=":nocache",
        )
      else:
        benchmark_obj = litert_lm.Benchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            cache_dir=":nocache",
            enable_speculative_decoding=enable_speculative_decoding,
        )

      click.echo(f"Benchmarking model: {self.to_str()} ({self.model_path})")
      click.echo(f"Number of tokens in prefill: {prefill_tokens}")
      click.echo(f"Number of tokens in decode : {decode_tokens}")
      click.echo(f"Backend                    : {backend}")

      spec_dec_str = "auto"
      if enable_speculative_decoding is True:
        spec_dec_str = "true"
      elif enable_speculative_decoding is False:
        spec_dec_str = "false"
      print(f"Speculative decoding       : {spec_dec_str}")
      if is_android:
        click.echo("Target                     : Android")

      result = benchmark_obj.run()

      click.echo("----- Results -----")
      click.echo(
          f"Prefill speed:        {result.last_prefill_tokens_per_second:.2f}"
          " tokens/s"
      )
      click.echo(
          f"Decode speed:         {result.last_decode_tokens_per_second:.2f}"
          " tokens/s"
      )
      click.echo(f"Init time:            {result.init_time_in_second:.4f} s")
      click.echo(
          f"Time to first token:  {result.time_to_first_token_in_second:.4f} s"
      )

    except Exception:  # pylint: disable=broad-exception-caught
      click.echo(click.style("An error occurred during benchmarking", fg="red"))
      traceback.print_exc()

  @classmethod
  def get_all_models(cls):
    """Returns a list of all locally available models."""
    model_paths = glob.glob(
        "*/model.litertlm",
        root_dir=get_converted_models_base_dir(),
        recursive=True,
    )

    return [
        Model.from_model_id(
            path.removesuffix("/model.litertlm").replace("--", "/")
        )
        for path in model_paths
    ]

  @classmethod
  def from_model_reference(cls, model_reference):
    """Creates a Model instance from a model reference."""
    if os.path.exists(model_reference):
      return cls.from_model_path(model_reference)
    else:
      # assume the reference is model_id
      return cls.from_model_id(model_reference)

  @classmethod
  def from_model_path(cls, model_path):
    """Creates a Model instance from a model path."""
    return cls(
        model_id=os.path.basename(model_path),
        model_path=os.path.abspath(model_path),
    )

  @classmethod
  def from_model_id(cls, model_id):
    """Creates a Model instance from a model ID."""
    return cls(
        model_id=model_id,
        model_path=os.path.join(
            get_converted_models_base_dir(),
            model_id.replace("/", "--"),
            "model.litertlm",
        ),
    )


# Just to use the huggingface convention. Likely to change.
def model_id_dir_name(model_id):
  """Converts a model ID to a directory name."""
  return model_id.replace("/", "--")


# ~/.litert-lm/models
def get_converted_models_base_dir():
  """Gets the base directory for all converted models."""
  return os.path.join(os.path.expanduser("~"), ".litert-lm", "models")


# ~/.litert-lm/models/<model_id>
def get_model_dir(model_id):
  """Gets the model directory for a given model ID."""
  return os.path.join(
      get_converted_models_base_dir(),
      model_id_dir_name(model_id),
  )
