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

import {Mutex} from './mutex.js';
import {SessionConfig, wasmSessionConfigToSessionConfig} from './session_config.js';
import {RecursiveRequired} from './types.js';
import {Responses as WasmResponses, Session as WasmSession} from './wasm_binding_types.js';
import {consumeEmscriptenVectorToArray} from './wasm_utils.js';

const BUSY_MESSAGE = 'Session is busy. An operation is already in progress.';

/**
 * LiteRT-LM Session
 */
export class Session {
  readonly config: RecursiveRequired<SessionConfig>;
  private isBusy = false;

  constructor(private readonly session: WasmSession, private readonly mutexes: {
    executor: Mutex;
  }) {
    const wasmSessionConfig = session.getSessionConfig();
    this.config = wasmSessionConfigToSessionConfig(wasmSessionConfig);
    wasmSessionConfig.delete();
  }

  async runPrefill(inputs: string[]): Promise<void> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      await this.mutexes.executor.acquireAndRun(async () => {
        await this.session.runPrefill(inputs);
      });
    } finally {
      this.isBusy = false;
    }
  }

  async runDecode(): Promise<Responses> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      return await this.mutexes.executor.acquireAndRun(async () => {
        const wasmResponses: WasmResponses = await this.session.runDecode();
        return new Responses(wasmResponses);
      });
    } finally {
      this.isBusy = false;
    }
  }

  /**
   * Sends a signal to cancel any current generation.
   */
  cancel() {
    this.session.cancelProcess();
  }

  /**
   * Clones the session. The cloned session will have all the settings and
   * context of the original session up to the point that clone is called.
   */
  async clone(): Promise<Session> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      return await this.mutexes.executor.acquireAndRun(() => {
        const wasmClonedSession = this.session.clone();
        return new Session(wasmClonedSession, this.mutexes);
      });
    } finally {
      this.isBusy = false;
    }
  }

  async delete(): Promise<void> {
    await this.mutexes.executor.acquireAndRun(() => {
      this.session.delete();
    });
  }
}

class Responses {
  constructor(private readonly responses: WasmResponses) {}

  getTexts(): string[] {
    const texts = this.responses.getTexts();
    return consumeEmscriptenVectorToArray(texts);
  }

  delete() {
    this.responses.delete();
  }
}