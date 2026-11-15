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

import {parseHuggingFaceModelReference, parseModelReference} from './model_reference.js';

describe('parseModelReference', () => {
  const cases: Array<[string, string]> = [
    ['hf://org/repo', 'hf://org/repo'],
    ['hf://org/repo@web/model.litertlm', 'hf://org/repo@web/model.litertlm'],
    ['litert-community/SmolLM3-3B', 'hf://litert-community/SmolLM3-3B'],
    ['  org/Qwen2.5-1.5B_v2  ', 'hf://org/Qwen2.5-1.5B_v2'],
    ['https://huggingface.co/org/repo', 'hf://org/repo'],
    ['https://huggingface.co/org/repo/', 'hf://org/repo'],
    ['http://huggingface.co/org/repo', 'hf://org/repo'],
    ['https://www.huggingface.co/org/repo', 'hf://org/repo'],
    [
      'https://huggingface.co/org/repo/resolve/main/model.litertlm',
      'https://huggingface.co/org/repo/resolve/main/model.litertlm',
    ],
    [
      'https://example.com/models/model.litertlm',
      'https://example.com/models/model.litertlm',
    ],
    [
      'http://localhost:8080/model.litertlm',
      'http://localhost:8080/model.litertlm',
    ],
  ];
  for (const [text, reference] of cases) {
    it(`parses ${text}`, () => {
      expect(parseModelReference(text)).toBe(reference);
    });
  }

  const invalid = [
    '',
    'repo',
    'org/repo/extra',
    'not a model',
    'hf://org',
    'hf://datasets/org/data',
    'ftp://example.com/model.litertlm',
    'javascript:alert(1)',
  ];
  for (const text of invalid) {
    it(`rejects ${JSON.stringify(text)}`, () => {
      expect(parseModelReference(text)).toBeUndefined();
    });
  }
});

describe('parseHuggingFaceModelReference', () => {
  it('accepts Hugging Face models', () => {
    expect(parseHuggingFaceModelReference('org/repo')).toBe('hf://org/repo');
    expect(parseHuggingFaceModelReference('hf://org/repo@web'))
        .toBe('hf://org/repo@web');
    expect(parseHuggingFaceModelReference('https://huggingface.co/org/repo'))
        .toBe('hf://org/repo');
  });

  it('rejects other URLs', () => {
    expect(parseHuggingFaceModelReference(
               'https://example.com/model.litertlm'))
        .toBeUndefined();
    expect(parseHuggingFaceModelReference(
               'https://huggingface.co/org/repo/resolve/main/m.litertlm'))
        .toBeUndefined();
    expect(parseHuggingFaceModelReference('not a model')).toBeUndefined();
  });
});
