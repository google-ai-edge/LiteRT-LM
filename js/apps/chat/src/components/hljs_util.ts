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

import {css, CSSResult} from 'lit';
import hljs from 'highlight.js';

type ModernHighlightFn = (code: string, options: { language: string }) => { value: string };

/**
 * Highlights the given code with the specified language using highlight.js.
 * @param code The source code to highlight.
 * @param language The language to highlight the code in.
 * @returns The highlighted code as an HTML string.
 */
export function highlight(code: string, language: string): string {
  const highlightFn = hljs.highlight as unknown as ModernHighlightFn;
  return highlightFn(code, { language }).value;
}

/**
 * Checks if the given language name or alias is supported by highlight.js.
 * @param name The language name or alias to check.
 * @returns True if the language is supported, false otherwise.
 */
export function getLanguage(name: string): boolean {
  return !!hljs.getLanguage(name);
}

/**
 * Automatically detects the language and highlights the given code.
 * @param code The source code to highlight.
 * @returns The highlight result, including the HTML string and detected
 *     language.
 */
export function highlightAuto(code: string): {value: string, language?: string} {
  return hljs.highlightAuto(code);
}

/**
 * Stylesheet hook exported for highlight.js consumers; theme-aware `.hljs*`
 * rules live in `shared_styles.ts`.
 */
export const hljsStyles: CSSResult = css``;
