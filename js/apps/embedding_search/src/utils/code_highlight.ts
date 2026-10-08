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
 * @fileoverview Syntax highlighting for text file previews. Decides whether a
 * file is code (and in which language) and renders it with highlight.js.
 */

import {css, html, TemplateResult} from 'lit';

import {highlight, highlightAuto, renderHighlighted} from './hljs_util.js';
import {fileExtension} from './media_types.js';

/** File extensions that are code, mapped to their highlight.js language. */
const CODE_LANGUAGES: Record<string, string> = {
  '.bash': 'bash', '.sh': 'bash',
  '.c': 'c', '.cc': 'cpp', '.cpp': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
  '.css': 'css', '.go': 'go', '.html': 'xml', '.xml': 'xml',
  '.java': 'java', '.js': 'javascript', '.jsx': 'javascript',
  '.json': 'json', '.proto': 'protobuf', '.py': 'python', '.rs': 'rust',
  '.sql': 'sql', '.ts': 'typescript', '.tsx': 'typescript',
  '.yaml': 'yaml', '.yml': 'yaml',
};

/** Extensions whose contents are prose or data rather than code. */
const PLAIN_EXTENSIONS = new Set([
  '.csv', '.ini', '.log', '.markdown', '.md', '.rst', '.tex', '.toml', '.tsv',
]);

const MIN_AUTO_RELEVANCE = 15;
const MAX_HIGHLIGHT_CHARS = 200_000;
const MAX_AUTO_DETECT_CHARS = 20_000;

let lastRender: {text: string, path: string, result: TemplateResult|string}|null =
    null;

/**
 * Renders the text of the file at `path`, syntax highlighted if it looks like
 * code. Returns the raw text when it is not code or highlighting fails.
 */
export function renderFileText(
    text: string, path: string): TemplateResult|string {
  if (lastRender?.path === path && lastRender.text === text) {
    return lastRender.result;
  }
  const result = computeFileText(text, path);
  lastRender = {text, path, result};
  return result;
}

function computeFileText(text: string, path: string): TemplateResult|string {
  const ext = fileExtension(path);
  if (PLAIN_EXTENSIONS.has(ext) || text.length > MAX_HIGHLIGHT_CHARS) {
    return text;
  }
  try {
    const language = CODE_LANGUAGES[ext];
    let highlighted: string;
    if (language) {
      highlighted = highlight(text, language);
    } else {
      if (text.length > MAX_AUTO_DETECT_CHARS) return text;
      const guess = highlightAuto(text);
      if (!guess.language || guess.relevance < MIN_AUTO_RELEVANCE) return text;
      highlighted = guess.value;
    }
    return html`<code class="hljs">${renderHighlighted(highlighted)}</code>`;
  } catch (e) {
    console.warn(`[EmbeddingSearch] Failed to highlight ${path}:`, e);
    return text;
  }
}

/** Token colors for highlighted code, tuned to the app's dark LiteRT palette. */
export const codeHighlightStyles = css`
  .hljs-comment, .hljs-quote { color: #7c8db5; font-style: italic; }
  .hljs-keyword, .hljs-selector-tag, .hljs-meta .hljs-keyword, .hljs-doctag { color: #ff7ab2; }
  .hljs-string, .hljs-regexp, .hljs-addition, .hljs-meta .hljs-string { color: #a5e075; }
  .hljs-number, .hljs-literal, .hljs-symbol, .hljs-bullet { color: #f9c859; }
  .hljs-title, .hljs-section, .hljs-selector-id, .hljs-selector-class { color: #5be3e3; }
  .hljs-built_in, .hljs-type, .hljs-class .hljs-title { color: #00c99e; }
  .hljs-attr, .hljs-attribute, .hljs-property, .hljs-variable, .hljs-template-variable, .hljs-params { color: #9ecbff; }
  .hljs-tag, .hljs-name, .hljs-meta { color: #c3a6ff; }
  .hljs-deletion { color: #ff8a8a; }
  .hljs-emphasis { font-style: italic; }
  .hljs-strong { font-weight: 700; }
`;
