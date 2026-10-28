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

// Model caching that doesn't depend on the rest of LiteRT-LM, so that it can
// be imported on its own, e.g. from a service worker. Nothing in this
// directory may import from outside it.

export {type HuggingFaceFile, type HuggingFaceModelRef, type HuggingFaceUri, parseHuggingFaceUri, type ResolvedHuggingFaceModel, resolveHuggingFaceModel, type ResolveHuggingFaceModelOptions, toHuggingFaceUri} from './hugging_face.js';
export {getModelLockName} from './locks.js';
export {type CachedModelInfo, createCacheEntryHeaders, getCacheRequestUrlForSource, type ModelCache, type ModelCachePutOptions} from './model_cache.js';
export {getCacheKey, type ModelLoadProgress, type ModelSource} from './model_source.js';
