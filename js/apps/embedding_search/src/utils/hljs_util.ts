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
 *
 * @fileoverview highlight.js bindings for the open source build. Replaces the
 * internal js/apps/embedding_search/src/utils/hljs_util.ts and must export the
 * same API.
 */

import hljs from 'highlight.js';
import {unsafeHTML} from 'lit/directives/unsafe-html.js';

/** Highlights `code` as `language`, returning highlight.js HTML. */
export function highlight(code: string, language: string): string {
  return hljs.highlight(code, {language}).value;
}

/**
 * Guesses the language of `code` and highlights it. `relevance` measures how
 * confident the guess is; prose typically scores close to zero.
 */
export function highlightAuto(code: string):
    {value: string, language?: string, relevance: number} {
  const result = hljs.highlightAuto(code);
  return {
    value: result.value,
    language: result.language,
    relevance: result.relevance,
  };
}

/**
 * Renders highlight.js HTML output into a Lit template. highlight.js escapes
 * the source text, so its output only contains its own <span> markup.
 */
export function renderHighlighted(highlightedHtml: string) {
  return unsafeHTML(highlightedHtml);
}

/** Creates an object URL for a Blob or File. */
export function createObjectUrl(blob: Blob): string {
  return URL.createObjectURL(blob);
}
