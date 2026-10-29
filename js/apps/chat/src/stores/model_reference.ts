/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {parseHuggingFaceUri} from '@litert-lm/core';

/**
 * A Hugging Face repo ID, `<org>/<repo>`, as copied by the "Copy model name"
 * button on its page.
 */
const REPO_ID = /^[\w.-]+\/[\w.-]+$/;

/**
 * A Hugging Face repo's page, `https://huggingface.co/<org>/<repo>`, also with
 * `http://` or `www.`.
 */
const REPO_PAGE =
    /^https?:\/\/(?:www\.)?huggingface\.co\/([\w.-]+\/[\w.-]+)\/?$/;

/**
 * Converts text that the user entered to a model URL or `hf://` URI, or
 * returns undefined if it's neither. Accepts:
 *
 * - `hf://<org>/<repo>[@<revision>][/<path>]` URIs.
 * - Hugging Face repo IDs (`<org>/<repo>`) and repo pages
 *   (`https://huggingface.co/<org>/<repo>`), as `hf://` URIs.
 * - Other `http(s)://` URLs, as is.
 */
export function parseModelReference(text: string): string|undefined {
  const trimmed = text.trim();
  if (trimmed.startsWith('hf://')) {
    return parseHuggingFaceUri(trimmed) ? trimmed : undefined;
  }
  if (REPO_ID.test(trimmed)) {
    return `hf://${trimmed}`;
  }
  const repoPage = REPO_PAGE.exec(trimmed);
  if (repoPage) {
    return `hf://${repoPage[1]}`;
  }
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return undefined;
  }
  return url.protocol === 'https:' || url.protocol === 'http:' ? url.href :
                                                                 undefined;
}
