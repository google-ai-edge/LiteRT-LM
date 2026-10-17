#!/usr/bin/env python3
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

"""Manages Hugging Face README.md model cards for LiteRT-LM Omni models.

Each supported Omni model card under `https://huggingface.co/litert-community`
is tracked in this directory as `<model-id>.README.md`.

Examples:
  # Upload any updated <model-id>.README.md files to Hugging Face:
  python3 omni/docs/upload_readmes.py

  # Check which local README.md files differ from Hugging Face without
  uploading:
  python3 omni/docs/upload_readmes.py --dry-run --diff

  # Upload only specific model(s) (by short alias or Hugging Face model ID):
  python3 omni/docs/upload_readmes.py parakeet-tdt kokoro

  # Re-download README.md files from Hugging Face into omni/docs/:
  python3 omni/docs/upload_readmes.py --download
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import difflib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

DEFAULT_HF_ORG = "litert-community"

# Canonical mapping from Omni model aliases to <model-id> under
# https://huggingface.co/litert-community.
OMNI_MODELS: tuple[tuple[str, str], ...] = (
    ("parakeet-tdt", "parakeet-tdt-0.6b-v3"),
    ("parakeet-ctc", "parakeet-ctc-0.6b"),
    ("whisper-tiny", "whisper-tiny"),
    ("moonshine-tiny", "moonshine-tiny"),
    ("qwen3-asr", "Qwen3-ASR-0.6B"),
    ("qwen3-tts", "Qwen3-TTS-12Hz-0.6B-Base"),
    ("kokoro", "Kokoro-82M"),
    ("bonsai-flux2", "Bonsai-Image-ternary-4B"),
    ("flux2-klein", "FLUX.2-klein-4B-LiteRT"),
)

ALIAS_TO_MODEL_ID: dict[str, str] = {}
for _alias, _model_id in OMNI_MODELS:
  ALIAS_TO_MODEL_ID[_alias.lower()] = _model_id
  ALIAS_TO_MODEL_ID[_model_id.lower()] = _model_id

_LINTER_HEADER = (
    "\n<!-- mdformat off(Hugging Face model card GFM) -->\n"
    "<!-- disableFinding(LINE_OVER_80) -->\n"
    "<!-- disableFinding(LINK_ID) -->\n"
    "<!-- disableFinding(SNIPPET_EM_DASH) -->\n"
    "<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->\n"
    "<!-- disableFinding(WHITESPACE_LINES) -->\n"
)

_LINTER_HEADER_RE = re.compile(
    r"\n(?:[ \t]*<!--\s*(?:mdformat\b|disableFinding\().*?-->[ \t]*\n)+",
    re.MULTILINE,
)


def strip_linter_directives(text: str) -> str:
  """Removes markdown formatter/linter comments before comparing or uploading."""
  cleaned = _LINTER_HEADER_RE.sub("", text)
  if not cleaned.endswith("\n"):
    cleaned += "\n"
  return cleaned


def add_linter_directives(text: str) -> str:
  """Inserts markdown formatter/linter directives after YAML frontmatter."""
  cleaned = strip_linter_directives(text)
  if cleaned.startswith("---\n"):
    end_idx = cleaned.find("\n---\n", 4)
    if end_idx != -1:
      split_pos = end_idx + len("\n---\n")
      body = cleaned[split_pos:]
      return cleaned[:split_pos] + _LINTER_HEADER + body
    if cleaned.rstrip("\n").endswith("\n---"):
      return cleaned.rstrip("\n") + "\n" + _LINTER_HEADER
  return _LINTER_HEADER.lstrip("\n") + cleaned


def resolve_model_ids(
    targets: Sequence[str], docs_dir: pathlib.Path, org: str
) -> list[str]:
  """Resolves user-specified model aliases or IDs into canonical model IDs."""
  if not targets:
    known_ids = [model_id for _, model_id in OMNI_MODELS]
    known_set = set(known_ids)
    for path in sorted(docs_dir.glob("*.README.md")):
      model_id = path.name[: -len(".README.md")]
      if model_id not in known_set:
        known_ids.append(model_id)
        known_set.add(model_id)
    return known_ids

  resolved: list[str] = []
  for target in targets:
    cleaned = target.strip()
    if cleaned.startswith(f"{org}/"):
      cleaned = cleaned[len(org) + 1 :]
    if cleaned.endswith(".README.md"):
      cleaned = cleaned[: -len(".README.md")]
    model_id = ALIAS_TO_MODEL_ID.get(cleaned.lower(), cleaned)
    if model_id not in resolved:
      resolved.append(model_id)
  return resolved


def fetch_remote_readme(org: str, model_id: str) -> str | None:
  """Fetches the current README.md from Hugging Face main branch."""
  url = f"https://huggingface.co/{org}/{model_id}/raw/main/README.md"
  req = urllib.request.Request(
      url, headers={"User-Agent": "litert-lm-omni-readme-sync/1.0"}
  )
  try:
    with urllib.request.urlopen(req, timeout=30) as resp:
      text = resp.read().decode("utf-8")
      if not text.endswith("\n"):
        text += "\n"
      return text
  except urllib.error.HTTPError as e:
    if e.code == 404:
      return None
    raise RuntimeError(f"Failed to fetch {url}: HTTP {e.code}") from e


def upload_readme_to_hf(
    org: str, model_id: str, content: str, commit_message: str
) -> None:
  """Uploads README.md content to https://huggingface.co/<org>/<model_id>."""
  repo_id = f"{org}/{model_id}"
  try:
    from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top

    api = HfApi()
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return
  except ImportError:
    pass

  hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")
  if not hf_cli:
    raise RuntimeError(
        "Neither `huggingface_hub` Python package nor `hf` / "
        "`huggingface-cli` binary was found. Install with: "
        "`pip install -U huggingface_hub`."
    )

  with tempfile.NamedTemporaryFile(
      mode="w", encoding="utf-8", suffix=".README.md", delete=False
  ) as tmp:
    tmp.write(content)
    tmp_path = pathlib.Path(tmp.name)

  try:
    cmd = [
        hf_cli,
        "upload",
        repo_id,
        str(tmp_path),
        "README.md",
        "--repo-type=model",
        f"--commit-message={commit_message}",
    ]
    subprocess.run(cmd, check=True)
  finally:
    tmp_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Download or upload LiteRT-LM Omni model README.md files under "
          "https://huggingface.co/litert-community."
      )
  )
  parser.add_argument(
      "models",
      nargs="*",
      help=(
          "Optional model aliases (e.g. parakeet-tdt, kokoro) or Hugging Face "
          "model IDs (e.g. parakeet-tdt-0.6b-v3). Defaults to all 9 Omni "
          "models."
      ),
  )
  parser.add_argument(
      "--download",
      action="store_true",
      help=(
          "Download the latest README.md from Hugging Face into "
          "<docs-dir>/<model-id>.README.md instead of uploading."
      ),
  )
  parser.add_argument(
      "-n",
      "--dry-run",
      action="store_true",
      help="Compare local README.md files with Hugging Face without uploading.",
  )
  parser.add_argument(
      "-d",
      "--diff",
      action="store_true",
      help="Print unified diff when local and remote README.md differ.",
  )
  parser.add_argument(
      "-f",
      "--force",
      action="store_true",
      help="Upload README.md even when remote content is identical.",
  )
  parser.add_argument(
      "-m",
      "--commit-message",
      default="Update README.md",
      help=(
          "Commit message for Hugging Face upload (default: 'Update"
          " README.md')."
      ),
  )
  parser.add_argument(
      "--org",
      default=DEFAULT_HF_ORG,
      help=f"Hugging Face organization (default: '{DEFAULT_HF_ORG}').",
  )
  parser.add_argument(
      "--docs-dir",
      type=pathlib.Path,
      default=pathlib.Path(__file__).resolve().parent,
      help="Directory containing <model-id>.README.md files.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  docs_dir: pathlib.Path = args.docs_dir.resolve()
  docs_dir.mkdir(parents=True, exist_ok=True)
  model_ids = resolve_model_ids(args.models, docs_dir, args.org)

  if args.download:
    for model_id in model_ids:
      remote_text = fetch_remote_readme(args.org, model_id)
      if remote_text is None:
        print(
            f"[MISSING] https://huggingface.co/{args.org}/{model_id} has no"
            " README.md",
            file=sys.stderr,
        )
        return 1
      local_path = docs_dir / f"{model_id}.README.md"
      formatted = add_linter_directives(remote_text)
      local_path.write_text(formatted, encoding="utf-8")
      print(f"[DOWNLOADED] {args.org}/{model_id} -> {local_path.name}")
    return 0

  updated_count = 0
  unchanged_count = 0
  for model_id in model_ids:
    local_path = docs_dir / f"{model_id}.README.md"
    if not local_path.is_file():
      print(
          f"[ERROR] Local file not found: {local_path}",
          file=sys.stderr,
      )
      return 1

    local_clean = strip_linter_directives(
        local_path.read_text(encoding="utf-8")
    )
    remote_raw = fetch_remote_readme(args.org, model_id)
    remote_clean = (
        strip_linter_directives(remote_raw) if remote_raw is not None else ""
    )

    is_changed = local_clean != remote_clean
    if not is_changed and not args.force:
      print(f"[UP-TO-DATE] {args.org}/{model_id} ({local_path.name})")
      unchanged_count += 1
      continue

    if args.diff and is_changed:
      diff_lines = difflib.unified_diff(
          remote_clean.splitlines(keepends=True),
          local_clean.splitlines(keepends=True),
          fromfile=f"hf://{args.org}/{model_id}/README.md",
          tofile=str(local_path.name),
      )
      sys.stdout.writelines(diff_lines)

    if args.dry_run:
      status = "MODIFIED" if is_changed else "FORCED"
      print(f"[{status}] {args.org}/{model_id} ({local_path.name})")
      updated_count += 1
      continue

    upload_readme_to_hf(args.org, model_id, local_clean, args.commit_message)
    print(f"[UPLOADED] {local_path.name} -> {args.org}/{model_id}/README.md")
    updated_count += 1

  action = "would be uploaded" if args.dry_run else "uploaded"
  print(
      f"Summary: {updated_count} {action}, {unchanged_count} up-to-date"
      f" (total {len(model_ids)})."
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
