/**
 * Copyright 2026 Google LLC
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

import {HuggingFaceFile, ModelSource, parseHuggingFaceUri, ResolvedHuggingFaceModel, resolveHuggingFaceModel, ResolveHuggingFaceModelOptions, toHuggingFaceUri} from './model-cache/index.js';
import {Backend} from './wasm_binding_types.js';

/**
 * A model to load: a URL, a `hf://` URI, a `ModelSource`, or the model's data.
 *
 * A `hf://<org>/<repo>[@<revision>][/<path>]` URI loads a model from the
 * Hugging Face Hub. Without a path, the file is chosen from the `.litertlm`
 * files at the top level of the repo by the engine's `backend`:
 *
 * - `Backend.GPU_ARTISAN`: a `-gpu.litertlm` file.
 * - Any other backend: a `.litertlm` file other than `-gpu.litertlm`.
 * - No backend: a `-gpu.litertlm` file if there is one, else any other.
 *
 * Hardware-specific files (e.g. for NPUs) are never chosen. If more than one
 * file matches, the first is chosen of: the one matching a `-gpu.litertlm`
 * file, the one named after the repo, `model.litertlm`, an int4 model, and
 * then the shortest name. A warning lists the others, which can be chosen by
 * giving their path in the URI. If no file matches, creating the engine fails
 * before anything is downloaded.
 *
 * A `ModelSource` can have a `hf://` URI as its `url`, e.g. to give an access
 * token or to check the file's SHA-256.
 */
export type ModelInput = string|ModelSource|Blob|ReadableStream<Uint8Array>;

/**
 * A model that's ready to load: a URL, a `ModelSource`, or the model's data,
 * but never a `hf://` URI.
 */
export type ResolvedModel = string|ModelSource|Blob|ReadableStream<Uint8Array>;

/** A `ModelInput` with any `hf://` URI resolved to a download URL. */
export declare interface ResolvedModelInput {
  model: ResolvedModel;
  /** The Hugging Face model it resolved to, if it was a `hf://` URI. */
  huggingFace?: ResolvedHuggingFaceModel;
  /**
   * Checks the backend that the engine resolved the model to, once it has read
   * the model's header. Throws if the model came from a `hf://` URI and isn't
   * for the requested backend (see `checkModelBackend()`).
   */
  checkBackend: (resolvedBackend: Backend) => void;
}

/**
 * Repos whose models are looked up on a `web` branch before `main`. Some
 * models' web versions are published on a temporary `web` branch until they
 * are merged into `main`. Only trusted repos are included so that other repos
 * don't start relying on this.
 */
// TODO: Remove once the web versions of models are on `main`.
const WEB_BRANCH_ORG = 'litert-community/';

/**
 * Resolves a `hf://` URI in a model input to a download URL for the file
 * chosen by `backend`, which is the backend the user asked for, if any. Other
 * inputs are returned unchanged.
 *
 * This makes network requests for `hf://` URIs, but doesn't use Wasm.
 */
export async function resolveModelInput(
    model: ModelInput, backend: Backend|undefined):
    Promise<ResolvedModelInput> {
  const source = typeof model === 'string' ? {url: model} :
      model instanceof Blob || model instanceof ReadableStream ? undefined :
                                                                 model;
  if (!source?.url.startsWith('hf://')) {
    return {model, checkBackend: () => {}};
  }
  const ref = parseHuggingFaceUri(source.url);
  if (!ref) {
    throw new Error(`Invalid Hugging Face model URI: ${
        source.url}. Expected hf://<org>/<repo>[@<revision>][/<path>]`);
  }

  const options: ResolveHuggingFaceModelOptions = {
    selectFile: (files, revision) => {
      const uri = toHuggingFaceUri({repo: ref.repo, revision});
      const choice = selectModelFile(files, ref.repo, backend);
      if (!choice) {
        throw new Error(getNoModelFileMessage(uri, files, backend));
      }
      if (choice.others.length > 0) {
        console.warn(`Loading ${uri}/${choice.file.path}. Other model files ` +
                     `in the repo, which can be loaded by URI: ${
                         choice.others.map(file => `${uri}/${file.path}`)
                             .join(', ')}`);
      }
      return choice.file;
    },
  };
  if (source.accessToken) {
    options.accessToken = source.accessToken;
  }
  if (ref.repo.toLowerCase().startsWith(WEB_BRANCH_ORG)) {
    options.defaultRevisions = ['web', 'main'];
  }
  const huggingFace = await resolveHuggingFaceModel(ref, options);

  const resolved = {...huggingFace.source};
  const uri = toHuggingFaceUri(huggingFace);
  if (source.sha256 !== undefined) {
    const sha256 = source.sha256.toLowerCase();
    if (resolved.sha256 !== undefined && resolved.sha256 !== sha256) {
      throw new Error(`${uri} has SHA-256 ${resolved.sha256}, not ${sha256}`);
    }
    resolved.sha256 = sha256;
  }
  if (source.size !== undefined && source.size !== resolved.size) {
    throw new Error(
        `${uri} is ${resolved.size} bytes, not ${source.size} bytes`);
  }
  if (source.cacheKey !== undefined) {
    resolved.cacheKey = source.cacheKey;
  }
  return {
    model: resolved,
    huggingFace,
    checkBackend: resolvedBackend => {
      checkModelBackend(huggingFace, backend, resolvedBackend);
    },
  };
}

/**
 * Hardware-specific model files, which can't run on the web: Google Tensor,
 * Intel, Qualcomm and MediaTek NPU models.
 */
const HARDWARE_SPECIFIC_FILE =
    /(^|[-_.])(google_tensor|intel|qualcomm|mediatek|npu)([-_.]|$)/i;
/** The old name for `-gpu.litertlm` files. */
const OLD_GPU_FILE = /-web\.litertlm$/i;
const GPU_FILE = /-gpu\.litertlm$/i;
/** Suffixes of repo names that aren't part of their models' names. */
const REPO_NAME_SUFFIX = /[-_](litert[-_]?lm|litert)$/i;
const INT4 = /int4/i;

/** The model file chosen from a Hugging Face repo. */
export declare interface ModelFileChoice {
  file: HuggingFaceFile;
  /** The other files that match the backend, in order of preference. */
  others: HuggingFaceFile[];
}

/**
 * Chooses the model file for a backend from the files at the top level of a
 * Hugging Face repo, as described in `ModelInput`. Returns undefined if no
 * file matches.
 */
export function selectModelFile(
    files: HuggingFaceFile[], repo: string,
    backend: Backend|undefined): ModelFileChoice|undefined {
  const candidates = files.filter(
      file => file.path.toLowerCase().endsWith('.litertlm') &&
          !HARDWARE_SPECIFIC_FILE.test(file.path) &&
          !OLD_GPU_FILE.test(file.path));
  const gpuFiles = candidates.filter(file => GPU_FILE.test(file.path));
  const otherFiles = candidates.filter(file => !GPU_FILE.test(file.path));
  const matching = backend === Backend.GPU_ARTISAN ? gpuFiles :
      backend !== undefined                        ? otherFiles :
      gpuFiles.length > 0                          ? gpuFiles :
                                                     otherFiles;
  const [file, ...others] = sortModelFiles(matching, gpuFiles, repo);
  return file ? {file, others} : undefined;
}

/** Sorts model files in order of preference, as described in `ModelInput`. */
function sortModelFiles(
    files: HuggingFaceFile[], gpuFiles: HuggingFaceFile[],
    repo: string): HuggingFaceFile[] {
  const gpuFileBases = new Set(gpuFiles.map(
      file => file.path.toLowerCase().replace(GPU_FILE, '.litertlm')));
  const repoName = repo.slice(repo.indexOf('/') + 1)
                       .replace(REPO_NAME_SUFFIX, '')
                       .toLowerCase();
  const getSortKey = (file: HuggingFaceFile): number[] => {
    const path = file.path.toLowerCase();
    const name = path.replace(GPU_FILE, '').replace(/\.litertlm$/, '');
    return [
      gpuFileBases.has(path) ? 0 : 1,
      name === repoName ? 0 : 1,
      name === 'model' ? 0 : 1,
      INT4.test(name) ? 0 : 1,
      path.length,
    ];
  };
  const sorted = files.map(file => ({file, key: getSortKey(file)}));
  sorted.sort((a, b) => {
    for (let i = 0; i < a.key.length; i++) {
      if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
    }
    return a.file.path < b.file.path ? -1 : a.file.path > b.file.path ? 1 : 0;
  });
  return sorted.map(({file}) => file);
}

function getNoModelFileMessage(
    uri: string, files: HuggingFaceFile[], backend: Backend|undefined): string {
  const modelFiles = files.map(file => file.path)
                         .filter(path => /\.litertlm$/i.test(path));
  const forBackend =
      backend === undefined ? '' : ` for ${getBackendName(backend)}`;
  if (modelFiles.length === 0) {
    return `${uri} has no .litertlm model file${forBackend}`;
  }
  return `${uri} has no suitable model file${forBackend}. Add one of ${
      modelFiles.join(', ')} to the URI, e.g. ${uri}/${modelFiles[0]}`;
}

/**
 * Throws if a model loaded from a `hf://` URI resolved to a different backend
 * than the one the user asked for. An explicit backend is binding, so that a
 * model file meant for another backend isn't silently used instead.
 */
export function checkModelBackend(
    huggingFace: ResolvedHuggingFaceModel, requested: Backend|undefined,
    resolved: Backend): void {
  if (requested === undefined || requested === resolved) return;
  throw new Error(
      `${toHuggingFaceUri(huggingFace)} is a ${getBackendName(resolved)} ` +
      `model, but ${getBackendName(requested)} was requested. Choose ` +
      'another file, or leave `backend` unset to use the model\'s backend.');
}

/** Returns a backend's name, e.g. 'Backend.GPU'. */
function getBackendName(backend: Backend): string {
  const name =
      Object.entries(Backend).find(([, value]) => value === backend)?.[0];
  return name === undefined ? `backend ${backend}` : `Backend.${name}`;
}
