/**
 * Copyright 2025 Google LLC
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

// Placeholder for internal dependency on trusted resource url type

import {getGlobalLiteRtLm, getGlobalLiteRtLmPromise, hasGlobalLiteRtLm, hasGlobalLiteRtLmPromise, setGlobalLiteRtLm, setGlobalLiteRtLmPromise} from './global_litertlm.js';
import {LiteRtLm} from './litertlm_web.js';
import {load, LoadOptions} from './load.js';

type UrlString = string;

/**
 * Options for loading LiteRT-LM.
 *
 * @property threads Whether to load the threaded version of the Wasm module.
 *     Defaults to false. Unused when specifying a .js file directly instead of
 *     a directory containing the Wasm files.
 **/
export interface LoadLiteRtLmOptions extends LoadOptions {}

/**
 * Load LiteRT-LM Wasm files from the given URL. This needs to be called before
 * calling any other LiteRT-LM functions.
 *
 * The URL can be:
 *
 * - A directory containing the LiteRT Wasm files (e.g. `.../wasm/`), or
 * - The LiteRT-LM Wasm's js file (e.g. `.../litertlm_wasm_internal.js`)
 *
 * If the URL is to a directory, LiteRT-LM will detect what WASM features are
 * available in the browser and load the compatible WASM file. If the URL is
 * to a file, it will be loaded as is.
 *
 * @param path The path to the directory containing the LiteRT-LM Wasm files, or
 *     the full URL of the LiteRT-LM Wasm .js file.
 */
export function loadLiteRtLm(
    path: UrlString, options?: LoadLiteRtLmOptions): Promise<LiteRtLm> {
  if (hasGlobalLiteRtLmPromise()) {
    throw new Error('LiteRT-LM is already loading / loaded.');
  }
  setGlobalLiteRtLmPromise(load(path, options)
                               .then(async liteRtLm => {
                                 setGlobalLiteRtLm(liteRtLm);
                                 return liteRtLm;
                               })
                               .catch(error => {
                                 setGlobalLiteRtLmPromise(undefined);
                                 throw error;
                               }));
  return getGlobalLiteRtLmPromise()!;
}

/**
 * Unload the LiteRT-LM WASM module.
 *
 * This deletes the global LiteRT-LM instance and invalidate any models
 * associated with it. You will need to call loadLiteRtLm() again to reload the
 * module.
 */
export function unloadLiteRtLm(): void {
  if (hasGlobalLiteRtLmPromise() && !hasGlobalLiteRtLm()) {
    throw new Error(
        'LiteRT-LM is loading and can not be unloaded or canceled ' +
        'until it is finished loading.');
  }

  if (hasGlobalLiteRtLm()) {
    getGlobalLiteRtLm().delete();
    setGlobalLiteRtLm(undefined);
  }
  setGlobalLiteRtLmPromise(undefined);
}

/**
 * Get the global LiteRT-LM instance, or load it if it hasn't been loaded yet.
 */
export function getOrLoadGlobalLiteRtLm(): Promise<LiteRtLm> {
  return getGlobalLiteRtLmPromise() ?? loadLiteRtLm(LiteRtLm.DEFAULT_WASM_PATH);
}
