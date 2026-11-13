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
"""Experimental `decide` command for LiteRT-LM.

Not registered in `litert-lm` yet; run it with the `litert-lm-decide` binary.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from litert_lm import decision_engine
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import model


def _load_json(text: str, source: str) -> Any:
  try:
    return json.loads(text)
  except json.JSONDecodeError as e:
    raise click.ClickException(f"Invalid JSON in {source}: {e}") from e


def _flag_question(
    instructions: str, question_type: str | None, options: tuple[str, ...]
) -> dict[str, Any]:
  """Returns the question of the `--instructions` and `--option` flags."""
  q_type = (question_type or ("choice" if options else "noul")).lower()
  question: dict[str, Any] = {"type": q_type, "instructions": instructions}
  if q_type == "noul":
    if options:
      raise click.ClickException("--option is not used by noul questions.")
  elif q_type == "score":
    question["criteria"] = list(options)
  else:
    criteria: dict[str, str] = {}
    for raw in options:
      option_id, description = raw.split(":", 1) if ":" in raw else (raw, "")
      option_id = option_id.strip()
      if not option_id:
        raise click.BadParameter(
            f"{raw!r}: the option ID must not be empty.", param_hint="--option"
        )
      criteria[option_id] = description.strip()
    question["criteria"] = criteria
  return question


def _build_questions(
    questions_raw: tuple[str, ...],
    instructions: str | None,
    question_id: str | None,
    question_type: str | None,
    options: tuple[str, ...],
) -> dict[str, Any]:
  """Builds the questions from the `--question` and `--instructions` flags."""
  questions: dict[str, Any] = {}

  def add(qid: str, question: Any) -> None:
    if qid in questions:
      raise click.ClickException(f"Duplicate question ID {qid!r}.")
    questions[qid] = question

  for raw in questions_raw:
    parsed = _load_json(raw, "--question")
    if not isinstance(parsed, dict):
      raise click.ClickException(
          "--question must be a JSON object mapping question IDs to questions."
      )
    for qid, question in parsed.items():
      add(qid, question)
  if instructions is not None:
    add(
        question_id if question_id is not None else "q0",
        _flag_question(instructions, question_type, options),
    )
  elif options or question_type is not None or question_id is not None:
    raise click.ClickException(
        "--id, --option and --type require --instructions/-p."
    )
  return questions


@click.command(
    cls=help_formatter.ColorCommand,
    name="decide",
    help="""(Experimental) Answers calibrated decision questions about an input.

  Each question is answered from the model's next-token probabilities over the
  options, without generating text. Requests and responses use the Gemini
  Decisions API JSON format; the response is printed as JSON.

  \b
  Examples:
    # One noul (true / false) question.
    litert-lm-decide my-model -i "I was charged twice." \\
        -p "Does the customer ask for a refund?"

    # One choice question with described options.
    litert-lm-decide my-model -i "I was charged twice." --id topic \\
        -t choice -p "What is the message about?" \\
        -o "billing:Invoices and charges" -o "shipping:Delivery"

    # Several questions as JSON.
    litert-lm-decide my-model -i "I was charged twice." \\
        -q '{"refund": {"type": "noul", "instructions": "Refund?"},
             "urgency": {"type": "score", "instructions": "How urgent?",
                         "criteria": ["Low", "Medium", "High"]}}'

    # A full request from a file or stdin (see
    # python/litert_lm/_decision_json.py for the format).
    litert-lm-decide my-model -f request.json
    litert-lm-decide my-model < request.json
""",
)
@click.argument("model_reference")
@common.config_option
@click.option(
    "--input",
    "-i",
    "input_text",
    type=str,
    default=None,
    help=(
        "Input text to evaluate. Read from stdin if omitted and piped; piped"
        " input that starts with '{' is read as a full request instead."
    ),
)
@click.option(
    "--id",
    "question_id",
    type=str,
    default=None,
    help="Question ID for the -p question. Defaults to 'q0'.",
)
@click.option(
    "--type",
    "-t",
    "question_type",
    type=click.Choice(["noul", "choice", "score"], case_sensitive=False),
    default=None,
    help=(
        "Type of the -p question. Defaults to 'choice' if --option is set,"
        " otherwise 'noul'."
    ),
)
@click.option(
    "--instructions",
    "-p",
    "instructions",
    type=str,
    default=None,
    help="Question to answer about the input.",
)
@click.option(
    "--option",
    "-o",
    "options",
    multiple=True,
    type=str,
    help=(
        "Option of the -p question (repeatable): 'id' or 'id:description' for"
        " a choice question (the ID cannot contain ':'), or a level"
        " description, from the lowest level, for a score question."
    ),
)
@click.option(
    "--question",
    "-q",
    "questions_raw",
    multiple=True,
    type=str,
    help="Questions as a JSON object keyed by question ID (repeatable).",
)
@click.option(
    "--request",
    "-r",
    "request_json",
    type=str,
    default=None,
    help="Full request JSON with 'input' and 'questions'.",
)
@click.option(
    "--request-file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a file with a full request JSON.",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu", "npu"], case_sensitive=False),
    default=None,
    help="Hardware backend.",
)
@click.option(
    "--max-num-tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of tokens (context size).",
)
def decide_command(
    model_reference: str,
    *,
    input_text: str | None,
    question_id: str | None,
    question_type: str | None,
    instructions: str | None,
    options: tuple[str, ...],
    questions_raw: tuple[str, ...],
    request_json: str | None,
    request_file: str | None,
    backend: str | None,
    max_num_tokens: int | None,
) -> None:
  """Answers calibrated decision questions about an input."""
  request: Any = None
  if request_file is not None and request_json is not None:
    raise click.ClickException("Use only one of --request and --request-file.")
  if request_file is not None:
    with open(request_file, "r", encoding="utf-8") as f:
      request = _load_json(f.read(), request_file)
  elif request_json is not None:
    request = _load_json(request_json, "--request")
  elif input_text is None and not sys.stdin.isatty():
    piped = sys.stdin.read().strip()
    if piped.startswith("{") and not questions_raw and instructions is None:
      request = _load_json(piped, "stdin")
    elif piped:
      input_text = piped

  if request is not None:
    if not isinstance(request, dict):
      raise click.ClickException("The request must be a JSON object.")
    if (
        input_text is not None
        or questions_raw
        or instructions is not None
        or question_id is not None
        or question_type is not None
        or options
    ):
      raise click.ClickException(
          "--request/--request-file cannot be combined with --input,"
          " --question, --instructions, --id, --type or --option."
      )
  else:
    if input_text is None:
      raise click.ClickException(
          "Provide --input/-i, --request/-r or --request-file/-f."
      )
    questions = _build_questions(
        questions_raw, instructions, question_id, question_type, options
    )
    if not questions:
      raise click.ClickException(
          "Provide --instructions/-p or at least one --question/-q."
      )
    request = {"input": input_text, "questions": questions}

  model_obj = model.Model.from_model_reference(model_reference)
  if not model_obj.exists():
    raise click.ClickException(f"Failed to find model '{model_reference}'.")
  resolved_backend = model.parse_backend(backend, model_obj=model_obj)
  resolved_max_tokens = model.resolve_config_option(
      max_num_tokens, model_obj, "max_num_tokens"
  )
  cache = model.resolve_config_option(None, model_obj, "cache")
  engine_kwargs: dict[str, Any] = {
      "max_num_tokens": resolved_max_tokens,
      "cache_dir": common.cache_dir_value_from_cache_mode(cache) or None,
  }
  if resolved_backend is not None:
    engine_kwargs["backend"] = resolved_backend

  try:
    with decision_engine.DecisionEngine(
        model_obj.model_path, **engine_kwargs
    ) as engine:
      output = engine.decide_json(request)
  except (RuntimeError, ValueError, TypeError, NotImplementedError) as e:
    raise click.ClickException(str(e)) from e

  click.echo(json.dumps(output, indent=2, ensure_ascii=False))


def register(cli: click.Group) -> None:
  """Registers the decide command."""
  cli.add_command(decide_command)
