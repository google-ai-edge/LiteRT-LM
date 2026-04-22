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

from typing import Any, Dict, List, Tuple

from lm_eval.api.model import LM  # pylint: disable=g-importing-member
from lm_eval.api.registry import register_model  # pylint: disable=g-importing-member

import litert_lm

# Map string backend to litert_lm.Backend enum.
_BACKEND_MAP = {
    "CPU": litert_lm.Backend.CPU,
    "GPU": litert_lm.Backend.GPU,
}


@register_model("litert_lm")
class LitertLmModelRunner(LM):
  """A wrapper for the LiteRT LM model to be used with the LM Eval Harness."""

  def __init__(
      self,
      model_path: str,
      backend: str = "CPU",
      max_num_tokens: int = 4096,
      **kwargs
  ):
    super().__init__()
    self.model_path = model_path

    self.backend = _BACKEND_MAP.get(backend.upper(), litert_lm.Backend.CPU)
    self.max_num_tokens = max_num_tokens

    self.engine = litert_lm.Engine(
        model_path=self.model_path,
        backend=self.backend,
        max_num_tokens=self.max_num_tokens,
        cache_dir="",
    )
    self.engine.__enter__()

  def __del__(self):
    if hasattr(self, "engine"):
      try:
        self.engine.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-except
        pass

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
      context, gen_args = request.args

      until = gen_args.get("until")
      if until and not isinstance(until, list):
        until = [until]

      with self.engine.create_conversation() as conversation:
        response = conversation.send_message(context)
        # content_list represents a list of message content blocks
        # (e.g., text, image), where each block is a dictionary defining its
        # 'type' and associated data.
        content_list: List[Dict[str, Any]] = response.get("content", [])
        text_response = ""
        for item in content_list:
          if item.get("type") == "text":
            text_response += item.get("text", "")

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
    cached_text_responses = {}

    for request in requests:
      context, continuation = request.args

      # Using conversation.send_message to check is_greedy is a slow workaround
      # since it does full generation. We should ideally get this efficiently
      # from a token-level API directly during run_text_scoring.
      if context not in cached_text_responses:
        with self.engine.create_conversation() as conversation:
          response = conversation.send_message(context)
          content_list: List[Dict[str, Any]] = response.get("content", [])
          text_response = ""
          for item in content_list:
            if item.get("type") == "text":
              text_response += item.get("text", "")
          cached_text_responses[context] = text_response

      with self.engine.create_session() as session:
        session.run_prefill([context])
        scoring_responses = session.run_text_scoring([continuation])
        # We provide exactly one continuation string per request, so the engine
        # returns a single score.
        score = scoring_responses.scores[0] if scoring_responses.scores else 0.0

      is_greedy = cached_text_responses[context].startswith(continuation)
      res.append((score, is_greedy))
    return res

  def loglikelihood_rolling(self, requests: Any) -> list[float]:
    # Pending to expose per-token logprobs.
    raise NotImplementedError()
