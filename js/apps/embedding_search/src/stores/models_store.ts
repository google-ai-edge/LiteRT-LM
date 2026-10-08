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

import {Backend, EmbeddingEngine, getOrLoadGlobalLiteRtLm} from '@litert-lm/core';

import {MODEL_CACHE_NAME, MODEL_URL} from '../app_config.js';

const LOCAL_MODEL_PREFIX = 'https://local-model/';
const LAST_MODEL_KEY = 'embedding-search:last-model';

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024);
  return gb >= 1 ? `${gb.toFixed(2)} GB` :
                   `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function filenameOf(cacheKey: string): string {
  const path = cacheKey.split('?')[0]!;
  return decodeURIComponent(path.split('/').pop() || path);
}

function abortError(): DOMException {
  return new DOMException('Model load cancelled.', 'AbortError');
}

function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError';
}

function withProgress(
    source: ReadableStream<Uint8Array>, onChunk: (bytes: number) => void,
    signal?: AbortSignal): ReadableStream<Uint8Array> {
  const reader = source.getReader();
  const onAbort = () => void reader.cancel(abortError()).catch(() => {});
  signal?.addEventListener('abort', onAbort, {once: true});
  const detach = () => signal?.removeEventListener('abort', onAbort);
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        if (signal?.aborted) throw abortError();
        const {done, value} = await reader.read();
        if (signal?.aborted) throw abortError();
        if (done) {
          detach();
          controller.close();
          return;
        }
        onChunk(value.length);
        controller.enqueue(value);
      } catch (e) {
        detach();
        controller.error(e);
        void reader.cancel(e).catch(() => {});
      }
    },
    cancel(reason) {
      detach();
      void reader.cancel(reason);
    },
  });
}

/**
 * Loads a single EmbeddingGemma `.litertlm` model into the LiteRT-LM
 * EmbeddingEngine and caches it in Cache Storage for subsequent visits.
 */
export class ModelsStore {
  isWasmLoaded = false;
  isLoading = false;
  loadingProgressText = '';
  loadingPercent: number|null = null;
  engine: EmbeddingEngine|null = null;
  loadedModelName: string|null = null;
  loadedModelSize = '';
  loadTimeSec: number|null = null;

  private activeCacheKey: string|null = null;
  private loadAbort: AbortController|null = null;
  private bytesFullyRead = false;

  readonly modelUrl = MODEL_URL;

  constructor(
      private readonly onUpdate: () => void,
      private readonly onStatusChange: (status: string) => void,
  ) {}

  get canCancelLoad(): boolean {
    return this.isLoading && this.loadAbort !== null &&
        !this.loadAbort.signal.aborted && !this.bytesFullyRead;
  }

  cancelLoad(): boolean {
    if (!this.canCancelLoad) return false;
    this.loadAbort!.abort();
    this.loadingProgressText = 'Cancelling…';
    this.onUpdate();
    return true;
  }

  async autoLoad(): Promise<boolean> {
    try {
      const cache = await window.caches.open(MODEL_CACHE_NAME);
      const lastKey = window.localStorage.getItem(LAST_MODEL_KEY);
      for (const key of [lastKey, this.modelUrl].filter((k): k is string => !!k)) {
        if (await cache.match(key)) {
          await this.loadFromCache(key);
          return true;
        }
      }
      if (this.modelUrl) {
        await this.loadFromUrl(this.modelUrl);
        return true;
      }
    } catch (e) {
      console.warn('[EmbeddingSearch] Automatic model load failed:', e);
    }
    return false;
  }

  async loadFromFile(file: File): Promise<void> {
    const key = LOCAL_MODEL_PREFIX + encodeURIComponent(file.name);
    await this.runLoad(key, async (signal) => {
      void this.putInCache(key, file, file.size);
      return {
        stream: this.trackProgress(file.stream(), file.size, 'Reading', signal),
        size: file.size,
      };
    });
  }

  async loadFromUrl(url: string): Promise<void> {
    await this.runLoad(url, async (signal) => {
      const fetchModel = async (label: string) => {
        this.loadingProgressText = label;
        this.onUpdate();
        const res = await fetch(url, {signal});
        if (!res.ok || !res.body) {
          throw new Error(`Download failed (${res.status} ${res.statusText}).`);
        }
        return {
          body: res.body,
          total: Number(res.headers.get('content-length') || '0'),
        };
      };

      const {body, total} = await fetchModel('Connecting…');
      const download = this.trackProgress(body, total, 'Downloading', signal, '');
      if (await this.putInCache(url, download, total, signal)) {
        return this.openCached(url, signal, 'Reading downloaded model');
      }
      const retry = await fetchModel('Reconnecting…');
      return {
        stream: this.trackProgress(retry.body, retry.total, 'Downloading', signal),
        size: retry.total,
      };
    });
  }

  async loadFromCache(key: string): Promise<void> {
    await this.runLoad(
        key, (signal) => this.openCached(key, signal, 'Reading cached model'));
  }

  async forgetModel(): Promise<void> {
    const key = this.activeCacheKey;
    await this.unloadModel();
    if (key) {
      try {
        const cache = await window.caches.open(MODEL_CACHE_NAME);
        await cache.delete(key);
      } catch (e) {
        console.warn('[EmbeddingSearch] Failed to delete cached model:', e);
      }
    }
    window.localStorage.removeItem(LAST_MODEL_KEY);
    this.activeCacheKey = null;
    this.onStatusChange('Model removed from browser storage.');
    this.onUpdate();
  }

  async unloadModel(): Promise<void> {
    this.cancelLoad();
    const engine = this.engine;
    if (!engine) return;
    this.engine = null;
    this.loadedModelName = null;
    this.loadTimeSec = null;
    try {
      await engine.delete();
    } catch (_) {}
    this.onStatusChange('Model unloaded.');
    this.onUpdate();
  }

  private async openCached(key: string, signal: AbortSignal, verb: string):
      Promise<{stream: ReadableStream<Uint8Array>, size: number}> {
    const cache = await window.caches.open(MODEL_CACHE_NAME);
    const cached = await cache.match(key);
    if (!cached || !cached.body) {
      throw new Error('Cached model not found. Please upload it again.');
    }
    const total = Number(cached.headers.get('content-length') || '0');
    return {
      stream: this.trackProgress(cached.body, total, verb, signal),
      size: total,
    };
  }

  private async putInCache(
      key: string, body: Blob|ReadableStream<Uint8Array>, size: number,
      signal?: AbortSignal): Promise<boolean> {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/octet-stream',
      };
      if (size > 0) headers['Content-Length'] = String(size);
      const cache = await window.caches.open(MODEL_CACHE_NAME);
      await cache.put(key, new Response(body, {headers}));
      return true;
    } catch (e) {
      if (signal?.aborted || isAbortError(e)) throw abortError();
      console.warn('[EmbeddingSearch] Could not cache model:', e);
      this.onStatusChange(
          'Model could not be saved for next time (storage quota?).');
      if (body instanceof ReadableStream) void body.cancel().catch(() => {});
      return false;
    }
  }

  private async runLoad(
      key: string,
      open: (signal: AbortSignal) =>
          Promise<{stream: ReadableStream<Uint8Array>, size: number}>):
      Promise<void> {
    if (this.isLoading) {
      this.onStatusChange('A model is already loading…');
      return;
    }
    const name = filenameOf(key);
    const abort = new AbortController();
    this.loadAbort = abort;
    this.bytesFullyRead = false;
    this.isLoading = true;
    this.loadingPercent = null;
    this.loadingProgressText = 'Preparing…';
    this.onStatusChange(`Loading model "${name}"…`);
    this.onUpdate();

    const startTime = performance.now();
    try {
      if (!this.isWasmLoaded) {
        this.loadingProgressText = 'Loading LiteRT WASM runtime…';
        this.onUpdate();
        // Rewritten by copy.bara.sky for the OSS Vite dev server.
        const wasmPath = (import.meta.env.DEV || import.meta.env.MODE === 'selfcontained') ? './wasm' : undefined;
        await getOrLoadGlobalLiteRtLm(wasmPath);
        this.isWasmLoaded = true;
      }

      if (this.engine) {
        try {
          await this.engine.delete();
        } catch (_) {}
        this.engine = null;
        this.loadedModelName = null;
      }

      if (abort.signal.aborted) throw abortError();
      const {stream, size} = await open(abort.signal);
      const engine = await EmbeddingEngine.create({
        model: stream,
        backend: Backend.GPU,
        visionBackend: Backend.GPU,
        audioBackend: Backend.GPU,
      });
      if (abort.signal.aborted) {
        try {
          await engine.delete();
        } catch (_) {}
        throw abortError();
      }

      this.engine = engine;
      this.loadedModelName = name.replace(/\.litertlm$/, '');
      this.loadedModelSize = size > 0 ? formatBytes(size) : '';
      this.activeCacheKey = key;
      window.localStorage.setItem(LAST_MODEL_KEY, key);
      this.loadTimeSec = (performance.now() - startTime) / 1000;
      this.onStatusChange(`Model loaded in ${this.loadTimeSec.toFixed(1)}s.`);
    } catch (e) {
      if (abort.signal.aborted || isAbortError(e)) {
        this.onStatusChange('Model load cancelled.');
        throw abortError();
      }
      console.error('[EmbeddingSearch] Failed to load model:', e);
      this.onStatusChange(`Failed to load model: ${(e as Error).message}`);
      throw e;
    } finally {
      if (this.loadAbort === abort) this.loadAbort = null;
      this.isLoading = false;
      this.loadingPercent = null;
      this.loadingProgressText = '';
      this.onUpdate();
    }
  }

  private trackProgress(
      source: ReadableStream<Uint8Array>, total: number, verb: string,
      signal: AbortSignal,
      finalText = 'Compiling WebGPU EmbeddingEngine…'):
      ReadableStream<Uint8Array> {
    let read = 0;
    let lastUpdate = 0;
    return withProgress(source, (bytes) => {
      read += bytes;
      const now = performance.now();
      const done = total > 0 && read >= total;
      if (done && finalText) this.bytesFullyRead = true;
      if (!done && now - lastUpdate < 100) return;
      lastUpdate = now;
      this.loadingPercent = total > 0 ? Math.round((read / total) * 100) : null;
      this.loadingProgressText = done && finalText ?
          finalText :
          `${verb} ${formatBytes(read)}${
              total > 0 ? ` / ${formatBytes(total)}` : ''}`;
      this.onUpdate();
    }, signal);
  }
}
