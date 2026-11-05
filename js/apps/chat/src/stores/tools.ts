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

import {JsonValue, ToolWithImplementation} from '@litert-lm/core';

import {CodeSandbox} from '../components/code_sandbox.js';

/** What tools can use to do their work. */
export interface ChatToolContext {
  codeSandbox: CodeSandbox;
}

/** A tool that the user can let the model use. */
export interface ChatTool {
  /** The function name that the model calls the tool by. */
  name: string;
  /** The name shown in the settings. */
  label: string;
  /** What the tool does, for the user. */
  summary: string;
  create(context: ChatToolContext): ToolWithImplementation;
}

/** The tools that the model can use, which the user can turn off. */
export const CHAT_TOOLS: readonly ChatTool[] = [
  {
    name: 'run_javascript',
    label: 'JavaScript sandbox',
    summary:
        'Runs JavaScript in a sandbox, e.g. to do math or manipulate text.',
    create: ({codeSandbox}) => ({
      type: 'function',
      function: {
        name: 'run_javascript',
        description:
            'Executes Javascript code in a sandbox and returns the result and console output. VERY useful for doing math, text manipulation, and other algorithmic operations.',
        parameters: {
          type: 'object',
          properties: {
            code: {
              type: 'string',
              description:
                  'The Javascript code to execute. Can use console.log to output data, or return a value.'
            }
          },
          required: ['code']
        }
      },
      execute: async (args: Record<string, JsonValue>) => {
        const code = args['code'] as string;
        if (typeof code !== 'string') {
          return {error: 'Expected string parameter "code"'};
        }
        return codeSandbox.run(code);
      }
    }),
  },
];

/** Creates the tools that aren't disabled. */
export function createEnabledTools(
    disabledTools: readonly string[],
    context: ChatToolContext): ToolWithImplementation[] {
  return CHAT_TOOLS.filter(tool => !disabledTools.includes(tool.name))
      .map(tool => tool.create(context));
}
