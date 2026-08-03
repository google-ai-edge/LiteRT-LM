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

"""Run subcommand for LiteRT-LM CLI."""

import json
import os
import sys
import traceback

import click
import prompt_toolkit
from prompt_toolkit import key_binding

import litert_lm
from litert_lm_builder import litertlm_builder
from litert_lm_cli import cli_helpers
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import huggingface_download
from litert_lm_cli import model
from litert_lm_cli.commands import convert as _convert_module

try:
  # pylint: disable=g-import-not-at-top
  from litert_lm.adb import adb_engine  # pytype: disable=import-error

  _HAS_ADB = True
except ImportError:
  _HAS_ADB = False


class SessionState:
  """State for the interactive session."""

  def __init__(self):
    self.active_channel = None


class LoggingToolEventHandler(litert_lm.ToolEventHandler):
  """Log tool call and tool response events."""

  def __init__(self, state: SessionState):
    self.state = state

  def approve_tool_call(self, tool_call):
    """Logs a tool call."""
    if self.state.active_channel is not None:
      click.echo("\n", nl=False)
      self.state.active_channel = None
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


def _execute_prompt(
    state: SessionState,
    conversation: litert_lm.AbstractConversation,
    prompt: str,
    attachments: tuple[str, ...] = (),
):
  """Executes a single prompt and prints the result."""
  state.active_channel = None

  if attachments:
    content = []
    for path in attachments:
      abs_path = os.path.abspath(path)
      content.append(
          {"type": model.get_attachment_type(abs_path), "path": abs_path}
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
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          if state.active_channel is not None:
            click.echo()
            state.active_channel = None
          click.echo(click.style(item.get("text", ""), fg="yellow"), nl=False)

      channels = chunk.get("channels", {})
      for channel_name, channel_content in channels.items():
        if state.active_channel != channel_name:
          if state.active_channel is not None:
            click.echo()
          click.echo(click.style(f"[{channel_name}] ", fg="blue"), nl=False)
          state.active_channel = channel_name
        click.echo(click.style(channel_content, fg="yellow"), nl=False)
    if state.active_channel is not None:
      click.echo()
    else:
      click.echo()
  except KeyboardInterrupt:
    conversation.cancel_process()
    for _ in stream:
      pass
    click.echo(click.style("\n[Generation cancelled]", dim=True))


def _execute_raw_prompt(session: litert_lm.AbstractSession, prompt: str):
  """Executes a single raw prompt and prints the result."""
  session.run_prefill([prompt])
  stream = session.run_decode_async()
  try:
    for chunk in stream:
      if chunk.texts:
        click.echo(click.style(chunk.texts[0], fg="yellow"), nl=False)
    click.echo()
  except KeyboardInterrupt:
    for _ in stream:
      pass
    click.echo(click.style("\n[Generation cancelled]", dim=True))


def _create_keybindings() -> key_binding.KeyBindings:
  """Creates keybindings for the interactive prompt."""
  kb = key_binding.KeyBindings()

  @kb.add("enter")
  def _handle_enter(event):
    buffer = event.current_buffer
    if buffer.text.strip():
      buffer.validate_and_handle()

  @kb.add("c-j")
  @kb.add("escape", "enter")
  def _handle_newline(event):
    event.current_buffer.insert_text("\n")

  @kb.add("c-c")
  def _handle_clear_or_exit(event):
    buffer = event.current_buffer
    if buffer.text:
      buffer.text = ""
    else:
      event.app.exit(exception=EOFError)

  return kb


def run_interactive(
    model_obj: model.Model,
    *,
    is_android: bool = False,
    backend: str | None = None,
    preset: str | None = None,
    prompt: str | None = None,
    enable_speculative_decoding: bool | None = None,
    no_template: bool = False,
    max_num_tokens: int | None = None,
    max_num_images: int | None = None,
    filter_channel_content_from_kv_cache: bool = False,
    vision_backend: str | None = None,
    audio_backend: str | None = None,
    attachments: tuple[str, ...] = (),
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    cache: str | None = None,
    cpu_thread_count: int | None = None,
    activation_data_type: litert_lm.ActivationDataType | None = None,
) -> None:
  """Runs the model interactively or with a single prompt."""
  if not model_obj.exists():
    click.echo(
        click.style(
            f"Could not find {model_obj.to_str()} locally in"
            f" {model_obj.model_path}.",
            fg="red",
        )
    )
    return

  state = SessionState()

  try:
    backend_val = model.parse_backend(
        backend, model_obj=model_obj, cpu_thread_count=cpu_thread_count
    )
    vision_backend_val = model.parse_backend(
        vision_backend,
        model_obj=model_obj,
        target_model_types={
            litertlm_builder.TfLiteModelType.VISION_ENCODER.value,
        },
        label="vision",
    )
    audio_backend_val = model.parse_backend(
        audio_backend,
        model_obj=model_obj,
        target_model_types={
            litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
        },
        label="audio",
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

    cache_dir_val = common.cache_dir_value_from_cache_mode(cache)

    if is_android:
      if not _HAS_ADB:
        raise ImportError("litert_lm.adb dependencies are not available.")
      engine_cm = adb_engine.AdbEngine(
          model_obj.model_path,
          backend=backend_val,
          max_num_tokens=max_num_tokens,
          vision_backend=vision_backend_val,
          audio_backend=audio_backend_val,
          cache_dir=cache_dir_val,
      )
    else:
      engine_cm = litert_lm.Engine(
          model_obj.model_path,
          backend=backend_val,
          enable_speculative_decoding=enable_speculative_decoding,
          max_num_tokens=max_num_tokens,
          max_num_images=max_num_images,
          vision_backend=vision_backend_val,
          audio_backend=audio_backend_val,
          cache_dir=cache_dir_val,
          activation_data_type=activation_data_type,
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
          tools, messages, extra_context = model.load_preset(preset)
          if tools is None and messages is None and extra_context is None:
            return

        handler = LoggingToolEventHandler(state) if tools else None

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
            _execute_raw_prompt(runner, prompt)
          elif isinstance(runner, litert_lm.AbstractConversation):
            _execute_prompt(state, runner, prompt, attachments=attachments)
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
            key_bindings=_create_keybindings(),
        )

        is_first_prompt = True
        while True:
          try:
            user_prompt = prompt_session.prompt(
                prompt_toolkit.ANSI(click.style("> ", fg="green", bold=True)),
                multiline=True,
                prompt_continuation=lambda width, line_number, is_soft_wrap: (
                    ""
                ),
            )
            if not user_prompt:
              continue

            if isinstance(runner, litert_lm.AbstractSession):
              _execute_raw_prompt(
                  runner,
                  user_prompt,
              )
            elif isinstance(runner, litert_lm.AbstractConversation):
              if is_first_prompt:
                _execute_prompt(
                    state, runner, user_prompt, attachments=attachments
                )
                is_first_prompt = False
              else:
                _execute_prompt(state, runner, user_prompt)

          except EOFError:
            break
          except KeyboardInterrupt:
            click.echo()
            continue
          except Exception:  # pylint: disable=broad-exception-caught
            click.echo(click.style("Error during inference", fg="red"))
            traceback.print_exc()

  except Exception:  # pylint: disable=broad-exception-caught
    click.echo(click.style("An error occurred", fg="red"))
    traceback.print_exc()


@click.command(
    cls=help_formatter.ColorCommand,
    help="""Runs a LiteRT-LM model interactively or with a single prompt.
  \b
  Examples:
    # Run interactively using a model ID from 'litert-lm list'
    litert-lm run my-model

    # Run with a single prompt using a local path
    litert-lm run ./model.litertlm --prompt "Hi there!"

    # Run directly from a HuggingFace repository
    litert-lm run --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference", required=False)
@click.option(
    "--prompt", default=None, help="A single prompt to run once and exit."
)
@click.option(
    "--preset",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a Python file containing tool functions and system"
        " instructions."
    ),
)
@click.option(
    "--no-template",
    is_flag=True,
    default=False,
    help=(
        "Interact with the model directly without applying prompt templates."
        " That means the input should include all control tokens for the model"
        " expected."
    ),
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=None,
    help=(
        "Maximum number of tokens for the KV cache. If not set, use the"
        " default from the native engine."
    ),
)
@click.option(
    "--filter-channel-content-from-kv-cache",
    is_flag=True,
    default=False,
    help="Whether to filter channel content from the KV cache.",
)
@click.option(
    "--vision-backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default=None,
    help=(
        "The backend to use for vision encoding. If not set, use the model's"
        " configured value."
    ),
)
@click.option(
    "--audio-backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default=None,
    help=(
        "The backend to use for audio encoding. If not set, use the model's"
        " configured value."
    ),
)
@click.option(
    "--attachment",
    multiple=True,
    type=click.Path(dir_okay=False),
    help=(
        "Path to an attachment (image or audio only). Can be specified multiple"
        " times. Attachements are placed before the first user text prompt."
    ),
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "The number of top logits used during sampling. If not set, use the"
        " default from the model or engine."
    ),
)
@click.option(
    "--top-p",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "The cumulative probability threshold for nucleus sampling. If not set,"
        " use the default from the model or engine."
    ),
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=None,
    help=(
        "The temperature to use for sampling. If not set, use the default from"
        " the model or engine."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=(
        "The seed to use for randomization. If not set, use the default from"
        " the model or engine."
    ),
)
@common.common_inference_options
def run(
    model_reference: str | None = None,
    prompt: str | None = None,
    preset: str | None = None,
    backend: str | None = None,
    android: bool = False,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
    no_template: bool = False,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
    max_num_tokens: int | None = None,
    filter_channel_content_from_kv_cache: bool = False,
    vision_backend: str | None = None,
    audio_backend: str | None = None,
    attachment: tuple[str, ...] = (),
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    cache: str | None = None,
    cpu_thread_count: int | None = None,
    activation_data_type: str | None = None,
) -> None:
  r"""Runs a LiteRT-LM model interactively or with a single prompt.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prompt: A single prompt to run once and exit.
    preset: Path to a Python file containing tool functions and system
      instructions.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    no_template: Interact with the model directly without applying prompt
      templates or stripping stop tokens.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    filter_channel_content_from_kv_cache: Whether to filter channel content from
      the KV cache.
    vision_backend: The backend to use for vision tasks.
    audio_backend: The backend to use for audio tasks.
    attachment: Path to an attachment (e.g., image or audio).
    top_k: The number of top logits used during sampling.
    top_p: The cumulative probability threshold for nucleus sampling.
    temperature: The temperature to use for sampling.
    seed: The seed to use for randomization.
    cache: The cache mode to use (no, memory, or disk).
    cpu_thread_count: The number of threads to use for CPU backend.
    activation_data_type: The activation data type to use for inference.
  """
  if attachment and no_template:
    click.echo(
        click.style(
            "Error: Attachments are not supported with --no-template.",
            fg="red",
        )
    )
    return

  expanded_attachments = []
  num_images = 0
  for a in attachment:
    expanded = os.path.expanduser(a)
    if not os.path.exists(expanded):
      raise click.BadParameter(f"File '{a}' does not exist.")
    expanded_attachments.append(expanded)

    try:
      a_type = model.get_attachment_type(expanded)
      if a_type == "image":
        num_images += 1
    except ValueError as e:
      raise click.BadParameter(str(e)) from e
  # If the stdin is not connected to the terminal, e.g., piped or redirected
  # input, then handle the input as the one-shot prompt.
  #
  # # Redirected input:
  # $ litert-lm run < prompt.txt
  # $ litert-lm run --prompt="Explain this error log" < error.log
  #
  # # Piped input:
  # $ cat text.txt | litert-lm run --prompt="Summarize the content."
  if not sys.stdin.isatty():
    piped_input = sys.stdin.read().strip()
    if piped_input:
      prompt = f"{prompt}\n\n{piped_input}" if prompt else piped_input
    elif not prompt:
      # If no prompt is provided and it's not a TTY, we can't be interactive.
      return

  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  model_reference = model_reference or cli_helpers.resolve_model_file(
      from_huggingface_repo,
      huggingface_token,
  )

  if from_huggingface_repo:
    model_path = huggingface_download.download_from_huggingface(
        repo_id=from_huggingface_repo,
        filename=model_reference,
        token=huggingface_token,
    )
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)
    if not model_obj.exists():
      # Only auto-convert if it looks like a HuggingFace repo ID (account/repo)
      # and is not a local path.
      parts = model_reference.split("/")
      if len(parts) == 2 and all(parts) and not os.path.exists(model_reference):
        click.echo(
            click.style(
                f"Model '{model_reference}' not found. Attempting to convert"
                f" from https://huggingface.co/{model_reference} ...",
                fg="yellow",
            )
        )
        model_obj = model.Model.from_model_reference(model_reference)

      if not model_obj.exists():
        click.echo(
            click.style(
                f"Failed to find or convert model '{model_reference}'.",
                fg="red",
            )
        )
        return

  max_num_images = None if num_images == 0 else num_images

  run_interactive(
      model_obj,
      prompt=prompt,
      is_android=android,
      backend=backend,
      preset=preset,
      enable_speculative_decoding=enable_speculative_decoding,
      no_template=no_template,
      max_num_tokens=max_num_tokens,
      max_num_images=max_num_images,
      filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
      vision_backend=vision_backend,
      audio_backend=audio_backend,
      attachments=tuple(expanded_attachments),
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      seed=seed,
      cache=cache,
      cpu_thread_count=cpu_thread_count,
      activation_data_type=(
          litert_lm.ActivationDataType.from_str(activation_data_type)
          if activation_data_type
          else None
      ),
  )


def register(cli: click.Group) -> None:
  """Registers the run command."""
  cli.add_command(run)
