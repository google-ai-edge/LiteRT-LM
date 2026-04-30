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

"""LiteRT LM model runner for LM Eval Harness."""

import base64
import io
import json
from typing import Any, Dict, List, Tuple

from lm_eval.api.model import LM  # pylint: disable=g-importing-member
from lm_eval.api.registry import register_model  # pylint: disable=g-importing-member

import litert_lm

# Map string backend to litert_lm.Backend enum.
_BACKEND_MAP = {
    "CPU": litert_lm.Backend.CPU,
    "GPU": litert_lm.Backend.GPU,
}

# Define the placeholder used by the evaluation harness.
DEFAULT_IMAGE_PLACEHOLDER = "<image>"


# TODO(b/496211682): Add audio support in LitertLmModelRunner.
@register_model("litert_lm")
class LitertLmModelRunner(LM):
  """A wrapper for the LiteRT LM model to be used with the LM Eval Harness."""

  MULTIMODAL = True

  def __init__(
      self,
      model_path: str,
      backend: str = "CPU",
      max_num_tokens: int = 4096,
      **kwargs
  ):
    super().__init__()
    self.model_path = model_path
    self.max_num_tokens = max_num_tokens

    # Default all modalities to the main backend if not specified.
    self.backend = _BACKEND_MAP.get(backend.upper(), litert_lm.Backend.CPU)

    # Extract optional specialized backends if present in kwargs.
    self.vision_backend = None
    if "vision_backend" in kwargs:
      self.vision_backend = _BACKEND_MAP.get(
          kwargs["vision_backend"].upper(), litert_lm.Backend.CPU
      )

    self.engine = litert_lm.Engine(
        model_path=self.model_path,
        backend=self.backend,
        max_num_tokens=self.max_num_tokens,
        cache_dir="",
        vision_backend=self.vision_backend,
    )
    self.engine.__enter__()

  def __del__(self):
    if hasattr(self, "engine"):
      try:
        self.engine.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-except
        pass

  def apply_chat_template(
      self,
      chat_history: List[Dict[str, str]],
      add_generation_prompt: bool = True,
  ) -> str:
    """Returns the chat history prepared for evaluation runner processing."""
    # Serializes the chat history to a JSON string to satisfy lm_eval hashing.
    return json.dumps(chat_history)

  def _extract_text(self, response_content: List[Dict[str, Any]]) -> str:
    """Extracts text content from a LiteRT LM response content list."""
    # response_content will be in the format of
    # [{"type": "text", "text": "..."},...].
    # Therefore we only extract the "text" fields from the response content and
    # concatenate them into a single string, ignoring other types of content.
    return "".join(
        item.get("text", "")
        for item in response_content
        if item.get("type") == "text"
    )

  @classmethod
  def _is_json_list(cls, json_string: Any) -> bool:
    """Checks if a given input is a valid JSON-serialized list string."""
    try:
      parsed_data = json.loads(json_string)
      return isinstance(parsed_data, list)
    except (json.JSONDecodeError, TypeError):
      return False

  def _split_context(self, context: str | List[Dict[str, str]]):
    """Splits context into conversation history and the final message."""
    # Note: For chat tasks with apply_chat_template=True, `context` originates
    # as a JSON string serialized from the `chat_history` list within
    # `apply_chat_template`. Therefore, we safely check for a leading '[' to
    # unpack structured multimodal histories.
    if isinstance(context, str) and self.__class__._is_json_list(context):
      try:
        context = json.loads(context)
      except Exception as e:
        raise ValueError(
            f"Failed to parse chat history JSON context string: {e}"
        ) from e
    if isinstance(context, list) and len(context) > 0:
      return context[:-1], context[-1]
    return None, context

  def _render_context(
      self, conversation, context: str | List[Dict[str, str]]
  ) -> str:
    """Renders the context into a single string using the conversation template."""
    if isinstance(context, str) and self.__class__._is_json_list(context):
      try:
        context = json.loads(context)
      except Exception as e:
        raise ValueError(
            f"Failed to parse chat history JSON context string: {e}"
        ) from e
    if isinstance(context, list):
      return "".join(conversation.render_message_to_string(m) for m in context)
    return conversation.render_message_to_string(context)

  def _process_multimodal_message(
      self, message: str | Dict[str, Any], mm_args: Dict[str, Any]
  ):
    """Injects visual/audio data into a message for multi-modal support."""
    if not mm_args:
      return message
    processed_msg = (
        {"role": "user", "content": message}
        if isinstance(message, str)
        else message.copy()
    )
    # Prepare the actual media data from mm_args.
    visual_items = []
    for img in mm_args.get("visual", []):
      if hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
      else:
        img_bytes = img
      img_b64 = base64.b64encode(img_bytes).decode("utf-8")
      visual_items.append({"type": "image", "blob": img_b64})

    content = processed_msg.get("content", "")

    # Standardize content to a list of parts.
    if isinstance(content, str):
      content = [{"type": "text", "text": content}] if content else []
    elif not isinstance(content, list):
      content = []

    new_content = []
    image_idx = 0

    for part in content:
      if part.get("type") == "text":
        text_parts = part["text"].split(DEFAULT_IMAGE_PLACEHOLDER)
        for i, segment in enumerate(text_parts):
          if segment:
            new_content.append({"type": "text", "text": segment})

          # Every segment except the final one corresponds to an actual inner
          # placeholder in the text. Interleave the corresponding image mapping.
          if i < len(text_parts) - 1:
            if image_idx < len(visual_items):
              # Replace the placeholder with the actual image data.
              new_content.append(visual_items[image_idx])
              image_idx += 1
            else:
              # Raise error to prevent silent dropping of the placeholder
              raise ValueError(
                  f"The prompt contains more '{DEFAULT_IMAGE_PLACEHOLDER}' "
                  f"placeholders than the {len(visual_items)} images provided."
              )
      else:
        new_content.append(part)

    # Append remaining media if they weren't matched to placeholders.
    if image_idx < len(visual_items):
      new_content.extend(visual_items[image_idx:])

    processed_msg["content"] = new_content
    return processed_msg

  def generate_until(self, requests) -> List[str]:
    r"""Generate greedily until a stopping sequence.

    Args:
        requests: List of ``Instance`` objects. Each ``Instance.args`` is a
          ``(context, gen_kwargs)`` tuple. *context* — the conditioning text
          (implementations must handle empty string). *gen_kwargs* — dictionary
          of keyword arguments for generation, which can include an "until" key
          with string(s) to generate until, e.g. ``{"until": ["\n", ".",
          "\n\n"]}``

    Returns:
        A list of strings containing the generated text.
    """
    res = []
    for request in requests:
      args = request.args
      context, gen_args = args[0], args[1]
      mm_args = args[2] if len(args) == 3 else {}

      until = gen_args.get("until", [])
      if until and not isinstance(until, list):
        until = [until]

      history, last_msg = self._split_context(context)
      last_msg = self._process_multimodal_message(last_msg, mm_args)

      with self.engine.create_conversation(messages=history) as conversation:
        response = conversation.send_message(last_msg)
        text_response = self._extract_text(response.get("content", []))

        if until:
          stop_indices = [
              text_response.find(stop_seq)
              for stop_seq in until
              if text_response.find(stop_seq) != -1
          ]
          if stop_indices:
            text_response = text_response[: min(stop_indices)]
        res.append(text_response)
    return res

  def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
    """Compute log-likelihood of generating a continuation from a context.

    Args:
        requests: List of ``Instance`` objects. Each ``Instance.args`` is a
          ``(context, continuation)`` tuple. *context* — the conditioning text
          (implementations must handle empty string). *continuation* — the text
          to score. Word-boundary spaces belong in the continuation (e.g.
          ``context="hello"  continuation=" world"``).

    Returns:
        A list of ``(logprob, is_greedy)`` tuples — the log-probability of
        the continuation and whether it would be produced by greedy decoding.
    """
    res = []
    rendered_contexts = []
    with self.engine.create_conversation() as conversation:
      for request in requests:
        context = request.args[0]
        # We turn off apply_prompt_template in create_session, and only apply
        # the chat template to the context here.
        rendered_contexts.append(self._render_context(conversation, context))

    cached_text_responses = {}
    for request, rendered_context in zip(requests, rendered_contexts):
      context, continuation = request.args[:2]
      mm_args = request.args[2] if len(request.args) == 3 else {}
      if mm_args:
        raise ValueError(
            "Multimodal arguments are not supported for scoring tasks."
        )

      cache_key = str(context)
      if cache_key not in cached_text_responses:
        with self.engine.create_session(apply_prompt_template=False) as session:
          session.run_prefill([rendered_context])
          response = session.run_decode()
          cached_text_responses[cache_key] = (
              response.texts[0] if response.texts else ""
          )

      with self.engine.create_session(apply_prompt_template=False) as session:
        session.run_prefill([rendered_context])
        scoring_res = session.run_text_scoring([continuation])
        # scoring.res.scores is a list of scores, one for each continuation.
        # Since we only have one continuation, we take the first element, or
        # 0.0 if there are no scores (this case should not happen).
        score = scoring_res.scores[0] if scoring_res.scores else 0.0

      is_greedy = cached_text_responses[cache_key].startswith(continuation)
      res.append((score, is_greedy))
    return res

  def loglikelihood_rolling(self, requests: Any) -> list[float]:
    # Pending to expose per-token logprobs.
    raise NotImplementedError()

  @property
  def tokenizer_name(self) -> str:
    return "litert_lm"
