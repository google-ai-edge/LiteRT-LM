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

"""OpenAI API compatible HTTP request handler for LiteRT-LM.

References:
* Responses API:
https://developers.openai.com/api/reference/resources/responses/methods/create
* Chat Completions API:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
"""

from __future__ import annotations

import http.server
import json
import traceback
from typing import Any

import click

import litert_lm
from litert_lm_cli.commands.serve import openai_chat_completions
from litert_lm_cli.commands.serve import openai_common
from litert_lm_cli.commands.serve import openai_embeddings
from litert_lm_cli.commands.serve import openai_models
from litert_lm_cli.commands.serve import openai_responses
from litert_lm_cli.commands.serve import util

# Re-export common helper functions for backwards compatibility with callers and
# tests.
_build_name_by_tool_call_id_map = openai_common.build_name_by_tool_call_id_map
_compute_token_usage = openai_common.compute_token_usage
_embedding_to_base64 = openai_embeddings.embedding_to_base64
_l2_normalize = openai_embeddings.l2_normalize
_normalize_embedding_input = openai_embeddings.normalize_embedding_input
_parse_model_parameter = openai_common.parse_model_parameter
_parse_response_format = openai_common.parse_response_format
_parse_sampler_config = openai_chat_completions.parse_sampler_config
_parse_thinking_config = openai_common.parse_thinking_config
_translate_openai_message = openai_common.translate_openai_message


class OpenAIHandler(util.CORSRequestHandler):
  """Handler for OpenAI API requests.

  Responses API:
  https://developers.openai.com/api/reference/resources/responses/methods/create

  Chat Completions API:
  https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

  Attributes:
    _headers_sent: Boolean flag tracking if HTTP response status headers have
      already been transmitted.
  """

  def __init__(
      self,
      request: Any,
      client_address: Any,
      server: http.server.HTTPServer,
  ):
    """Pre-assigns internal routing state flags before standard lifecycle execution."""
    self._headers_sent = False
    super().__init__(request, client_address, server)

  @property
  def headers_sent(self) -> bool:
    """Returns whether HTTP response headers have already been sent."""
    return self._headers_sent

  @headers_sent.setter
  def headers_sent(self, value: bool) -> None:
    """Sets whether HTTP response headers have already been sent."""
    self._headers_sent = value

  def _stream_response(
      self,
      conv: litert_lm.Conversation,
      prompt: str | dict[str, Any],
      formatter: openai_common.OpenAIStreamFormatter,
      *,
      max_completion_tokens: int | None = None,
      include_usage: bool = False,
      response_format: litert_lm.ResponseFormat | None = None,
  ) -> None:
    """Streams server-sent events using the provided formatter.

    Args:
      conv: The active LiteRT-LM conversation session.
      prompt: The input prompt payload (string or dictionary).
      formatter: The protocol-specific stream formatter.
      max_completion_tokens: The maximum number of tokens to generate.
      include_usage: Whether to emit a token usage chunk right before [DONE].
      response_format: Optional response format for constrained decoding.
    """
    self._headers_sent = True
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()

    try:
      self.wfile.write(formatter.format_initial())
      self.wfile.flush()

      has_tool_calls = False
      reasoning_tokens = 0
      for chunk in conv.send_message_async(
          prompt,
          max_output_tokens=max_completion_tokens,
          response_format=response_format,
      ):
        if chunk.get("channels"):
          reasoning_tokens += 1
        text_output = "".join(
            item.get("text", "")
            for item in chunk.get("content", [])
            if item.get("type") == "text"
        )
        if text_output:
          self.wfile.write(formatter.format_delta(text_output))
          self.wfile.flush()

        tool_calls = chunk.get("tool_calls", [])
        if tool_calls:
          has_tool_calls = True
          if hasattr(formatter, "format_tool_call_delta"):
            self.wfile.write(formatter.format_tool_call_delta(tool_calls))
            self.wfile.flush()

      finish_reason = "tool_calls" if has_tool_calls else "stop"
      self.wfile.write(formatter.format_complete(finish_reason=finish_reason))
      self.wfile.flush()
      if include_usage and hasattr(formatter, "format_usage"):
        usage_dict = openai_common.compute_token_usage(
            conv, reasoning_tokens=reasoning_tokens
        )
        self.wfile.write(formatter.format_usage(usage_dict))
        self.wfile.flush()
      self.wfile.write(formatter.format_final())
      self.wfile.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      click.echo(
          click.style(
              f"Error during streaming with prompt {prompt!r}: {e!r}\n"
              f"{traceback.format_exc()}",
              fg="red",
          )
      )
      conv.cancel_process()
      try:
        self.wfile.write(formatter.format_error(e))
        self.wfile.flush()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def stream_response(
      self,
      conv: litert_lm.Conversation,
      prompt: str | dict[str, Any],
      formatter: openai_common.OpenAIStreamFormatter,
      *,
      max_completion_tokens: int | None = None,
      include_usage: bool = False,
      response_format: litert_lm.ResponseFormat | None = None,
  ) -> None:
    """Streams server-sent events using the provided formatter."""
    self._stream_response(
        conv,
        prompt,
        formatter,
        max_completion_tokens=max_completion_tokens,
        include_usage=include_usage,
        response_format=response_format,
    )

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Handles GET requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)
    if path_without_query == "/v1/models":
      openai_models.handle_get_models(self)
    else:
      self.send_error(404, "Not Found")

  def _get_post_data(self) -> dict[str, Any] | None:
    """Extracts and parses the JSON payload safely."""
    try:
      content_length = int(self.headers.get("Content-Length", 0))
      raw_data = self.rfile.read(content_length)
      return json.loads(raw_data.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
      return None

  def _parse_request_body(self) -> dict[str, Any] | None:
    """Parses the request body as JSON, sending a 400 error if invalid."""
    body = self._get_post_data()
    if body is None:
      self.send_error(400, "Invalid JSON")
      return None
    return body

  def parse_request_body(self) -> dict[str, Any] | None:
    """Parses the request body as JSON, sending a 400 error if invalid."""
    return self._parse_request_body()

  def _get_engine(
      self,
      model_id: str,
      translated_messages: list[dict[str, Any]] | None = None,
      prompt: Any = None,
      backend: str | None = None,
      max_num_tokens: int | None = None,
  ) -> litert_lm.Engine | None:
    """Retrieves or initializes the engine for the given model ID.

    Args:
      model_id: The model identifier string.
      translated_messages: Optional list of already translated messages.
      prompt: Optional prompt payload.
      backend: Optional requested backend override.
      max_num_tokens: Optional requested max_num_tokens override.

    Returns:
      The LiteRT-LM Engine instance, or None if initialization failed.
    """
    del translated_messages, prompt
    try:
      assert isinstance(self.server, util.LiteRTLMServer)
      return util.get_or_initialize_server_engine(
          self.server,
          model_id=model_id,
          backend=backend,
          max_num_tokens=max_num_tokens,
      )
    except FileNotFoundError as e:
      self.send_error(404, "".join(traceback.format_exception_only(e)))
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load engine: {e!r}")
      return None

  def get_engine(
      self,
      model_id: str,
      translated_messages: list[dict[str, Any]] | None = None,
      prompt: Any = None,
      backend: str | None = None,
      max_num_tokens: int | None = None,
  ) -> litert_lm.Engine | None:
    """Retrieves or initializes the engine for the given model ID."""
    return self._get_engine(
        model_id,
        translated_messages=translated_messages,
        prompt=prompt,
        backend=backend,
        max_num_tokens=max_num_tokens,
    )

  def _get_embedding_engine(
      self,
      model_id: str,
      backend: str | None = None,
  ) -> litert_lm.EmbeddingEngine | None:
    """Retrieves or initializes the embedding engine for the given model ID.

    Args:
      model_id: The model identifier string.
      backend: Optional requested backend override.

    Returns:
      The LiteRT-LM EmbeddingEngine instance, or None if initialization failed.
    """
    try:
      assert isinstance(self.server, util.LiteRTLMServer)
      return util.get_or_initialize_server_embedding_engine(
          self.server,
          model_id=model_id,
          backend=backend,
      )
    except FileNotFoundError as e:
      self.send_error(404, "".join(traceback.format_exception_only(e)))
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load embedding engine: {e!r}")
      return None

  def get_embedding_engine(
      self,
      model_id: str,
      backend: str | None = None,
  ) -> litert_lm.EmbeddingEngine | None:
    """Retrieves or initializes the embedding engine for the given model ID."""
    return self._get_embedding_engine(model_id, backend=backend)

  def _handle_inference_error(
      self, e: Exception, model_id: str, prompt: Any
  ) -> None:
    """Handles errors occurring during inference by logging and sending 500.

    Args:
      e: The caught exception.
      model_id: The model identifier.
      prompt: The prompt payload.
    """
    click.echo(
        click.style(
            f"Error during inference for model {model_id!r} with prompt "
            f"{prompt!r}: {e!r}\n{traceback.format_exc()}",
            fg="red",
        )
    )
    if not self.wfile.closed and not self._headers_sent:
      try:
        self.send_error(500, "".join(traceback.format_exception_only(e)))
      except BrokenPipeError:
        pass

  def handle_inference_error(
      self, e: Exception, model_id: str, prompt: Any
  ) -> None:
    """Handles errors occurring during inference by logging and sending 500."""
    self._handle_inference_error(e, model_id, prompt)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles POST requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)

    router = {
        "/v1/chat/completions": (
            lambda: openai_chat_completions.handle_post_chat_completions(self)
        ),
        "/v1/responses": lambda: openai_responses.handle_post_responses(self),
        "/v1/embeddings": lambda: openai_embeddings.handle_post_embeddings(
            self
        ),
    }

    if path_without_query in router:
      router[path_without_query]()
    else:
      self.send_error(404, "Not Found")
