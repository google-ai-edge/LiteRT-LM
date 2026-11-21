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
"""Python wrapper of the experimental LiteRT-LM Decision Engine C API.

A decision engine answers typed questions (`NoulQuestion`, `ChoiceQuestion`
and `ScoreQuestion`) about a shared input with calibrated probabilities instead
of generated text. The types follow the Gemini Decisions API, and
`DecisionEngine.decide_json` accepts and returns its REST JSON (see
`_decision_json.py` for the format).

Experimental: the API may change without notice, and this module is not part of
the `litert-lm-api` PyPI package yet. Import it as a module, e.g.
`from litert_lm import decision_engine`.

Example:

  with decision_engine.DecisionEngine("model.litertlm") as engine:
    response = engine.decide(
        "I was charged twice for my subscription.",
        {
            "refund": decision_engine.NoulQuestion(
                "Does the customer ask for a refund?"
            ),
            "topic": decision_engine.ChoiceQuestion(
                "What is the message about?",
                criteria={"billing": "Invoices and charges", "shipping": ""},
            ),
        },
    )
    if response.answers["refund"].value:
      ...
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import dataclasses
from typing import Any, Union

from . import _decision_json
from . import interfaces
from ._ffi import _get_lib
from ._ffi import c_string_p
from ._ffi import call_checked
from ._ffi import create_checked
from ._ffi import get_checked
from ._ffi import InputDataType

# `LiteRtLmDecisionAnswerType` values.
_ANSWER_TYPE_NOUL = 0
_ANSWER_TYPE_CHOICE = 1
_ANSWER_TYPE_SCORE = 2

_P = ctypes.POINTER
_VOID_P = ctypes.c_void_p
_SIZE_T = ctypes.c_size_t
_FLOAT = ctypes.c_float
_STRINGS = _P(ctypes.c_char_p)


def _get_decision_lib() -> ctypes.CDLL:
  """Returns the native library with the Decision Engine signatures set."""
  lib = _get_lib()
  prefix = "litert_lm_experimental_decision_"
  signatures = {
      "engine_create": [_VOID_P, _P(_VOID_P)],
      "engine_create_from_engine": [_VOID_P, _P(_VOID_P)],
      "engine_decide": [_VOID_P, _P(_VOID_P), _SIZE_T, _VOID_P, _P(_VOID_P)],
      "question_create_noul": [c_string_p, c_string_p, _P(_VOID_P)],
      "question_create_choice": [c_string_p, c_string_p, _P(_VOID_P)],
      "question_create_score": [c_string_p, c_string_p, _P(_VOID_P)],
      "question_set_true_description": [_VOID_P, c_string_p],
      "question_set_false_description": [_VOID_P, c_string_p],
      "option_create": [c_string_p, _P(_VOID_P)],
      "option_set_description": [_VOID_P, c_string_p],
      "question_add_option": [_VOID_P, _VOID_P],
      "question_add_level": [_VOID_P, c_string_p],
      "request_create": [_P(_VOID_P)],
      "request_set_temperature": [_VOID_P, _FLOAT],
      "request_add_question": [_VOID_P, _VOID_P],
      "response_get_num_answers": [_VOID_P, _P(_SIZE_T)],
      "response_get_answer_id_at": [_VOID_P, _SIZE_T, _STRINGS],
      "response_get_answer_type_at": [_VOID_P, _SIZE_T, _P(ctypes.c_int)],
      "response_get_confidence_at": [_VOID_P, _SIZE_T, _P(_FLOAT)],
      "response_get_noul_at": [_VOID_P, _SIZE_T, _P(_FLOAT)],
      "response_get_choice_at": [_VOID_P, _SIZE_T, _STRINGS],
      "response_get_score_at": [_VOID_P, _SIZE_T, _P(_FLOAT)],
      "response_get_num_probabilities_at": [_VOID_P, _SIZE_T, _P(_SIZE_T)],
      "response_get_probabilities_at": [_VOID_P, _SIZE_T, _P(_P(_FLOAT))],
      "response_get_option_ids_at": [_VOID_P, _SIZE_T, _P(_STRINGS)],
  }
  for name, argtypes in signatures.items():
    func = getattr(lib, prefix + name)
    func.restype = ctypes.c_int
    func.argtypes = argtypes
  for name in (
      "engine_delete",
      "option_delete",
      "question_delete",
      "request_delete",
      "response_delete",
  ):
    func = getattr(lib, prefix + name)
    func.restype = None
    func.argtypes = [_VOID_P]
  return lib


@dataclasses.dataclass(frozen=True)
class NoulQuestion:
  """A true / false question.

  Attributes:
    instructions: The question, guidelines or rubric to evaluate.
    true_description: Optional description of when the answer is true.
    false_description: Optional description of when the answer is false.
  """

  instructions: str
  true_description: str = ""
  false_description: str = ""

  def to_dict(self) -> dict[str, Any]:
    """Returns the question in the Gemini Decisions API JSON format."""
    out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
    criteria = {}
    if self.true_description:
      criteria["true_description"] = self.true_description
    if self.false_description:
      criteria["false_description"] = self.false_description
    if criteria:
      out["criteria"] = criteria
    return out


@dataclasses.dataclass(frozen=True)
class ChoiceQuestion:
  """A single-select question over mutually exclusive options.

  Attributes:
    instructions: The question, guidelines or rubric to evaluate.
    criteria: 2 to 26 options, mapping each option ID to an optional
      description. The options are shown to the model in this order, labeled A
      to Z.
  """

  instructions: str
  criteria: Mapping[str, str]

  def to_dict(self) -> dict[str, Any]:
    """Returns the question in the Gemini Decisions API JSON format."""
    return {
        "type": "choice",
        "instructions": self.instructions,
        "criteria": dict(self.criteria),
    }


@dataclasses.dataclass(frozen=True)
class ScoreQuestion:
  """A rubric question over ordered levels.

  Attributes:
    instructions: The question, guidelines or rubric to evaluate.
    criteria: 2 to 26 level descriptions, from the lowest level (0) to the
      highest.
  """

  instructions: str
  criteria: Sequence[str]

  def to_dict(self) -> dict[str, Any]:
    """Returns the question in the Gemini Decisions API JSON format."""
    return {
        "type": "score",
        "instructions": self.instructions,
        "criteria": list(self.criteria),
    }


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


@dataclasses.dataclass(frozen=True)
class NoulAnswer:
  """Answer to a `NoulQuestion`.

  Attributes:
    confidence: Probability of the most likely answer, `max(noul, 1 - noul)`.
    noul: Calibrated probability that the answer is true.
  """

  confidence: float
  noul: float

  @property
  def value(self) -> bool:
    """The most likely answer: `noul >= 0.5`."""
    return self.noul >= 0.5

  def to_dict(self) -> dict[str, Any]:
    """Returns the answer in the Gemini Decisions API JSON format."""
    return {"type": "noul", "confidence": self.confidence, "noul": self.noul}


@dataclasses.dataclass(frozen=True)
class ChoiceAnswer:
  """Answer to a `ChoiceQuestion`.

  Attributes:
    confidence: Probability of `choice`.
    choice: ID of the most likely option.
    probabilities: Calibrated probability of each option ID, in the order of the
      question's criteria.
  """

  confidence: float
  choice: str
  probabilities: dict[str, float]

  def to_dict(self) -> dict[str, Any]:
    """Returns the answer in the Gemini Decisions API JSON format."""
    return {
        "type": "choice",
        "confidence": self.confidence,
        "choice": self.choice,
        "probabilities": dict(self.probabilities),
    }


@dataclasses.dataclass(frozen=True)
class ScoreAnswer:
  """Answer to a `ScoreQuestion`.

  Attributes:
    confidence: Probability of `most_likely_level`.
    score: Expected level `sum(k * probabilities[k])`, in `[0, K-1]`.
    probabilities: Calibrated probability of each level, indexed by level.
    legend: Description of each level, indexed by level.
  """

  confidence: float
  score: float
  probabilities: list[float]
  legend: list[str]

  @property
  def most_likely_level(self) -> int:
    """The most likely level (the lowest one on ties)."""
    return self.probabilities.index(max(self.probabilities))

  def to_dict(self) -> dict[str, Any]:
    """Returns the answer in the Gemini Decisions API JSON format."""
    return {
        "type": "score",
        "confidence": self.confidence,
        "score": self.score,
        "probabilities": {str(k): p for k, p in enumerate(self.probabilities)},
        "legend": {str(k): text for k, text in enumerate(self.legend)},
    }


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclasses.dataclass(frozen=True)
class DecisionResponse:
  """Response returned by `DecisionEngine.decide`.

  Attributes:
    answers: One answer per question ID, in the order of the questions.
  """

  answers: dict[str, Answer]

  def to_dict(self) -> dict[str, Any]:
    """Returns the response in the Gemini Decisions API JSON format."""
    return {"answers": {k: a.to_dict() for k, a in self.answers.items()}}


def _question_from_json(question: Mapping[str, Any]) -> Question:
  """Returns the question of a normalized `_decision_json` question."""
  question_type = question["type"]
  instructions = question["instructions"]
  if question_type == "noul":
    return NoulQuestion(instructions, **question.get("criteria", {}))
  if question_type == "choice":
    return ChoiceQuestion(instructions, question["criteria"])
  return ScoreQuestion(instructions, question["criteria"])


class DecisionEngine:
  """Manages the lifecycle of a LiteRT-LM Decision Engine."""

  def __init__(
      self,
      model_path: str,
      backend: interfaces.Backend | str = interfaces.Backend.CPU(),
      vision_backend: interfaces.Backend | str | None = None,
      audio_backend: interfaces.Backend | str | None = None,
      max_num_tokens: int | None = None,
      cache_dir: str | None = None,
  ):
    """Creates a decision engine.

    Args:
      model_path: Path to the `.litertlm` model file.
      backend: Backend for the main model (e.g. `Backend.CPU()` or "cpu").
      vision_backend: Optional backend for the vision encoder.
      audio_backend: Optional backend for the audio encoder.
      max_num_tokens: Optional maximum number of tokens (context size). Uses the
        engine default if None.
      cache_dir: Optional directory for compiled model caches. Use ":nocache" to
        disable caching.

    Raises:
      RuntimeError: If the engine cannot be created.
    """
    self._engine_ptr: int | None = None
    self._shared_engine: Any = None
    self._model_path = model_path
    self._lib = _get_decision_lib()

    def backend_name(b: interfaces.Backend | str | None) -> str | None:
      return b if b is None or isinstance(b, str) else b.get_name()

    settings = create_checked(
        self._lib,
        "litert_lm_engine_settings_create",
        model_path,
        backend_name(backend),
        backend_name(vision_backend),
        backend_name(audio_backend),
    )
    try:
      if (
          isinstance(backend, interfaces.Backend.NPU)
          and backend.litert_dispatch_lib_dir
      ):
        call_checked(
            self._lib,
            "litert_lm_engine_settings_set_litert_dispatch_lib_dir",
            settings,
            backend.litert_dispatch_lib_dir,
        )
      if (
          isinstance(backend, interfaces.CPU)
          and backend.thread_count is not None
      ):
        call_checked(
            self._lib,
            "litert_lm_engine_settings_set_num_threads",
            settings,
            backend.thread_count,
        )
      if max_num_tokens is not None:
        call_checked(
            self._lib,
            "litert_lm_engine_settings_set_max_num_tokens",
            settings,
            max_num_tokens,
        )
      if cache_dir:
        call_checked(
            self._lib,
            "litert_lm_engine_settings_set_cache_dir",
            settings,
            cache_dir,
        )
      self._engine_ptr = create_checked(
          self._lib, "litert_lm_experimental_decision_engine_create", settings
      )
    finally:
      self._lib.litert_lm_engine_settings_delete(settings)

  @classmethod
  def from_engine(cls, engine: interfaces.AbstractEngine) -> DecisionEngine:
    """Creates a `DecisionEngine` sharing an existing `Engine`'s weights.

    The returned `DecisionEngine` uses the native engine of `engine` and must
    be closed before `engine`, like sessions and conversations; `decide()`
    raises `RuntimeError` once `engine` is closed. It keeps no session between
    `decide()` calls, so generative sessions can be opened on `engine` at any
    time.

    Args:
      engine: An open `litert_lm.Engine` instance.

    Returns:
      A `DecisionEngine` sharing `engine`'s underlying native engine.

    Raises:
      ValueError: If `engine` is closed or is not a native `litert_lm.Engine`.
      RuntimeError: If the native decision engine cannot be created.
    """
    engine_ptr = getattr(engine, "_engine_ptr", None)
    if engine_ptr is None:
      raise ValueError("Expected an open native litert_lm.Engine instance.")
    instance = cls.__new__(cls)
    instance._engine_ptr = None
    instance._shared_engine = engine
    instance._model_path = getattr(engine, "model_path", "")
    instance._lib = _get_decision_lib()
    instance._engine_ptr = create_checked(
        instance._lib,
        "litert_lm_experimental_decision_engine_create_from_engine",
        engine_ptr,
    )
    return instance

  @property
  def model_path(self) -> str:
    return self._model_path

  def _check_alive(self) -> None:
    if self._engine_ptr is None:
      raise RuntimeError("DecisionEngine has already been closed.")
    if (
        self._shared_engine is not None
        and getattr(self._shared_engine, "_engine_ptr", None) is None
    ):
      raise RuntimeError(
          "The Engine shared by this DecisionEngine has been closed; close the"
          " DecisionEngine before the Engine."
      )

  def decide(
      self,
      input: str | Sequence[str],  # pylint: disable=redefined-builtin
      questions: Mapping[str, Question],
      temperature: float | None = None,
  ) -> DecisionResponse:
    """Answers typed questions about a shared input.

    Args:
      input: The text to evaluate, as one string or several strings (joined with
        newlines).
      questions: 1 to 1000 questions, keyed by question ID (1 to 64 characters
        in `[A-Za-z0-9_-]`). Answers follow this order.
      temperature: Optional softmax temperature (> 0) for all questions. If
        None, the engine uses its calibrated temperature for the model.

    Returns:
      A `DecisionResponse` with one answer per question, in question order.

    Raises:
      ValueError: If `input` or `questions` is empty.
      TypeError: If an input item or a question has an unsupported type.
      RuntimeError: If the engine rejects the request or the decision fails.
    """
    self._check_alive()
    texts = [input] if isinstance(input, str) else list(input)
    if not texts:
      raise ValueError("input must not be empty.")
    for text in texts:
      if not isinstance(text, str):
        raise TypeError(f"Unsupported input item type: {type(text)}")
    if not questions:
      raise ValueError("questions must not be empty.")
    for question in questions.values():
      if not isinstance(
          question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)
      ):
        raise TypeError(f"Unsupported question type: {type(question)}")

    lib = self._lib
    request_ptr = create_checked(
        lib, "litert_lm_experimental_decision_request_create"
    )
    input_ptrs: list[int] = []
    try:
      self._add_questions(request_ptr, questions)
      if temperature is not None:
        call_checked(
            lib,
            "litert_lm_experimental_decision_request_set_temperature",
            request_ptr,
            temperature,
        )
      for text in texts:
        data = text.encode("utf-8")
        input_ptrs.append(
            create_checked(
                lib,
                "litert_lm_input_data_create",
                InputDataType.TEXT,
                data,
                len(data),
            )
        )
      response_ptr = create_checked(
          lib,
          "litert_lm_experimental_decision_engine_decide",
          self._engine_ptr,
          (_VOID_P * len(input_ptrs))(*input_ptrs),
          len(input_ptrs),
          request_ptr,
      )
      try:
        return self._read_response(response_ptr, questions)
      finally:
        lib.litert_lm_experimental_decision_response_delete(response_ptr)
    finally:
      lib.litert_lm_experimental_decision_request_delete(request_ptr)
      for ptr in input_ptrs:
        lib.litert_lm_input_data_delete(ptr)

  def decide_json(self, request: Mapping[str, Any] | str) -> dict[str, Any]:
    """Answers a request in the Gemini Decisions API REST JSON format.

    See `_decision_json.py` for the request and response formats.

    Args:
      request: The request, as a JSON string or a parsed JSON mapping.

    Returns:
      The response JSON, `{"answers": {...}}`, in question order.

    Raises:
      ValueError: If the request is malformed: an unknown or missing field, an
        empty value or (for a JSON string) invalid JSON or a duplicate key.
      TypeError: If a request field has the wrong JSON type.
      NotImplementedError: If the input has image, audio, video or document
        content.
      RuntimeError: If the engine rejects the request or the decision fails.
    """
    parsed = _decision_json.parse_request(request)
    questions = {
        question_id: _question_from_json(question)
        for question_id, question in parsed.questions.items()
    }
    return self.decide(parsed.input, questions, parsed.temperature).to_dict()

  def _add_questions(
      self, request_ptr: int, questions: Mapping[str, Question]
  ) -> None:
    """Adds `questions` to the native request `request_ptr`."""
    prefix = "litert_lm_experimental_decision_question_"
    for question_id, q in questions.items():
      if isinstance(q, NoulQuestion):
        kind = "noul"
      elif isinstance(q, ChoiceQuestion):
        kind = "choice"
      else:
        kind = "score"
      q_ptr = create_checked(
          self._lib, f"{prefix}create_{kind}", question_id, q.instructions
      )
      try:
        if isinstance(q, NoulQuestion):
          if q.true_description:
            call_checked(
                self._lib,
                f"{prefix}set_true_description",
                q_ptr,
                q.true_description,
            )
          if q.false_description:
            call_checked(
                self._lib,
                f"{prefix}set_false_description",
                q_ptr,
                q.false_description,
            )
        elif isinstance(q, ChoiceQuestion):
          for option_id, description in q.criteria.items():
            self._add_option(q_ptr, option_id, description)
        else:
          for level_description in q.criteria:
            call_checked(
                self._lib,
                f"{prefix}add_level",
                q_ptr,
                level_description,
            )
        call_checked(
            self._lib,
            "litert_lm_experimental_decision_request_add_question",
            request_ptr,
            q_ptr,
        )
      finally:
        self._lib.litert_lm_experimental_decision_question_delete(q_ptr)

  def _add_option(self, q_ptr: Any, option_id: str, description: str) -> None:
    """Appends the option `option_id` to the choice question `q_ptr`."""
    option_ptr = create_checked(
        self._lib, "litert_lm_experimental_decision_option_create", option_id
    )
    try:
      if description:
        call_checked(
            self._lib,
            "litert_lm_experimental_decision_option_set_description",
            option_ptr,
            description,
        )
      call_checked(
          self._lib,
          "litert_lm_experimental_decision_question_add_option",
          q_ptr,
          option_ptr,
      )
    finally:
      self._lib.litert_lm_experimental_decision_option_delete(option_ptr)

  def _read_response(
      self, response_ptr: int, questions: Mapping[str, Question]
  ) -> DecisionResponse:
    """Reads the native response `response_ptr` to `questions`."""

    def get(name: str, out_type: Any, index: int) -> Any:
      return get_checked(
          self._lib,
          f"litert_lm_experimental_decision_response_get_{name}_at",
          out_type,
          response_ptr,
          index,
      )

    num_answers = get_checked(
        self._lib,
        "litert_lm_experimental_decision_response_get_num_answers",
        _SIZE_T,
        response_ptr,
    )
    answers: dict[str, Answer] = {}
    for i in range(num_answers):
      answer_id = get("answer_id", ctypes.c_char_p, i).decode("utf-8")
      answer_type = get("answer_type", ctypes.c_int, i)
      confidence = get("confidence", _FLOAT, i)
      if answer_type == _ANSWER_TYPE_NOUL:
        answers[answer_id] = NoulAnswer(
            confidence=confidence, noul=get("noul", _FLOAT, i)
        )
        continue
      num = get("num_probabilities", _SIZE_T, i)
      probabilities_ptr = get("probabilities", _P(_FLOAT), i)
      probabilities = [float(probabilities_ptr[k]) for k in range(num)]
      if answer_type == _ANSWER_TYPE_CHOICE:
        option_ids = get("option_ids", _STRINGS, i)
        answers[answer_id] = ChoiceAnswer(
            confidence=confidence,
            choice=get("choice", ctypes.c_char_p, i).decode("utf-8"),
            probabilities={
                option_ids[k].decode("utf-8"): probabilities[k]
                for k in range(num)
            },
        )
      elif answer_type == _ANSWER_TYPE_SCORE:
        question = questions[answer_id]
        assert isinstance(question, ScoreQuestion)
        answers[answer_id] = ScoreAnswer(
            confidence=confidence,
            score=get("score", _FLOAT, i),
            probabilities=probabilities,
            legend=list(question.criteria),
        )
      else:
        raise RuntimeError(f"Unknown decision answer type {answer_type}.")
    return DecisionResponse(answers=answers)

  def close(self) -> None:
    """Closes the engine and releases native resources."""
    engine_ptr = getattr(self, "_engine_ptr", None)
    shared_engine = getattr(self, "_shared_engine", None)
    if engine_ptr is not None:
      # A decision engine from `from_engine` must be deleted before its
      # `Engine`. If the `Engine` was closed first, leak the native decision
      # engine instead of deleting it after its engine.
      shared_closed = (
          shared_engine is not None
          and getattr(shared_engine, "_engine_ptr", None) is None
      )
      lib = getattr(self, "_lib", None)
      # `lib` can be None in `__del__` during interpreter shutdown.
      if not shared_closed and lib is not None:
        try:
          lib.litert_lm_experimental_decision_engine_delete(engine_ptr)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      self._engine_ptr = None
    self._shared_engine = None

  def __enter__(self) -> DecisionEngine:
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()
