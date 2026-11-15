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

import {getCacheEnvironment} from './environment.js';

/** Where to download a model from, and how to identify it in a cache. */
export declare interface ModelSource {
  /** The model's URL. Relative URLs are resolved against the page. */
  url: string;
  /** SHA-256 of the model file, as lowercase hex. */
  sha256?: string;
  /** Size of the model file in bytes. */
  size?: number;
  /**
   * Key that identifies the model in a cache. Defaults to `sha256` if it's
   * set, and to the URL otherwise.
   */
  cacheKey?: string;
  /**
   * Access token for the download, sent as an `Authorization: Bearer` header.
   * It is never stored in a cache.
   *
   * For Hugging Face, this is the token of the user loading the model, e.g.
   * from Sign in with Hugging Face (OAuth). Don't embed your own token in a
   * web page: everyone who loads the page can read it. An empty token is the
   * same as none.
   */
  accessToken?: string;
}

/** Progress of loading a model. */
export declare interface ModelLoadProgress {
  /** Whether the model is being read from the network or from a cache. */
  source: 'network'|'cache';
  /** Name of the cache the model is read from, if `source` is 'cache'. */
  cacheName?: string;
  loadedBytes: number;
  /** The model's size, if known. */
  totalBytes?: number;
}

/** Converts a model URL or source to a `ModelSource`. */
export function toModelSource(model: string|ModelSource): ModelSource {
  return typeof model === 'string' ? {url: model} : model;
}

/**
 * Returns the URL to download a model from, resolving relative URLs against
 * the page.
 */
export function resolveModelUrl(url: string): string {
  if (url.startsWith('hf://')) {
    throw new Error(
        `Resolve hf:// URIs with resolveHuggingFaceModel() first: ${url}`);
  }
  return new URL(url, getCacheEnvironment().baseUrl).href;
}

/** Returns the key that identifies a model in a cache. */
export function getCacheKey(source: ModelSource): string {
  return source.cacheKey ?? source.sha256 ?? resolveModelUrl(source.url);
}
