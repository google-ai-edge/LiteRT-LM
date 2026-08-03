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

import {Backend, Engine, getOrLoadGlobalLiteRtLm, GpuArtisanConfig} from '@litert-lm/core';

import {teeStream} from '../tee_stream.js';

import {SettingsStore} from './settings_store.js';

/**
 * Service responsible for downloading, caching, and building the GPU WASM
 * engine.
 */
export class ModelLoaderService {
  isWasmLoaded = false;
  isModelLoading = false;
  metricLoadTime = '-';

  cachedModels = new Map<string, number>();
  downloadProgresses = new Map<string, number>();
  downloadSpeeds = new Map<string, string>();

  engine: Engine|null = null;
  downloadAbortController: AbortController|null = null;
  isDownloadAborted = false;

  constructor(
      private readonly updateCallback: () => void,
      private readonly settings: SettingsStore,
      private readonly updateStatus: (msg: string) => void,
      private readonly createEngine: typeof Engine.create = Engine.create,
      private readonly loadWasm:
          typeof getOrLoadGlobalLiteRtLm = getOrLoadGlobalLiteRtLm) {}

  async updateCacheSize() {
    try {
      const cache = await window.caches.open('litertlm-models');
      const keys = await cache.keys();
      const newCachedModels = new Map<string, number>();

      for (const request of keys) {
        const response = await cache.match(request);
        if (response) {
          const filename = request.url.split('/').pop() || request.url;
          const contentLength = response.headers.get('content-length');
          if (contentLength) {
            newCachedModels.set(filename, Number(contentLength));
          } else {
            const blob = await response.blob();
            newCachedModels.set(filename, blob.size);
          }
        }
      }

      this.cachedModels = newCachedModels;
      this.updateCallback();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to calculate model cache size:', e);
    }
  }

  async deleteModelFromCache(modelPath: string, onDeleted?: () => void) {
    const confirmDelete = confirm(
        'Are you sure you want to remove this model\'s weights from your browser cache?');
    if (!confirmDelete) return;

    try {
      const cache = await window.caches.open('litertlm-models');
      const deleted = await cache.delete(modelPath);
      if (deleted) {
        this.updateStatus('Model cache removed successfully.');
        await this.updateCacheSize();

        if (this.settings.selectedModelPath === modelPath) {
          if (onDeleted) onDeleted();
        }
      }
    } catch (e) {
      console.error('[LiteRT-LM] Failed to delete model from Cache:', e);
    }
  }

  async clearAllCache(onDeleted?: () => void) {
    const confirmClear = confirm(
        'Are you sure you want to delete ALL cached models? This will free up significant disk space.');
    if (!confirmClear) return;

    try {
      this.updateStatus('Clearing all cache...');

      const deleted = await window.caches.delete('litertlm-models');
      if (deleted) {
        this.updateStatus('Model cache cleared successfully.');
        if (onDeleted) onDeleted();
        await this.updateCacheSize();
      }
    } catch (e) {
      console.error('[LiteRT-LM] Failed to clear cache:', e);
    }
  }

  private makeProgressStream(
      source: ReadableStream<Uint8Array>, onProgress: (bytes: number) => void) {
    const reader = source.getReader();
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const {done, value} = await reader.read();
          if (done) {
            controller.close();
            return;
          }
          onProgress(value.length);
          controller.enqueue(value);
        } catch (e) {
          controller.error(e);
        }
      },
      cancel(reason) {
        reader.cancel(reason);
      }
    });
  }

  async loadModelWeights(onModelLoaded: () => Promise<void>) {
    if (this.isModelLoading) return;

    const modelPath = this.settings.selectedModelPath;
    const modelFilename = modelPath.split('/').pop() || modelPath;

    this.isModelLoading = true;
    this.isDownloadAborted = false;
    this.updateStatus('Preparing runtime & model...');

    const startTime = performance.now();

    try {
      if (!this.isWasmLoaded) {
        this.updateStatus('Loading LiteRT WASM runtime...');
        const wasmPath = import.meta.env.DEV ? '/wasm' : undefined;
        await this.loadWasm(wasmPath);
        this.isWasmLoaded = true;
      }

      if (this.engine) {
        try {
          await this.engine.delete();
        } catch (e) {
        }
        this.engine = null;
      }

      const cache = await window.caches.open('litertlm-models');
      const cachedResponse = await cache.match(modelPath);
      let modelInput: ReadableStream<Uint8Array>;

      this.downloadProgresses.set(modelFilename, 0);
      this.downloadSpeeds.set(modelFilename, '0 MB / 0 MB');
      this.updateCallback();

      if (cachedResponse) {
        if (!cachedResponse.body) {
          throw new Error('Cached model response body is null.');
        }
        this.updateStatus(`Loading cached weights (${modelFilename})...`);

        const totalBytes =
            Number(cachedResponse.headers.get('content-length') || '0');
        let bytesRead = 0;

        const progressStream =
            this.makeProgressStream(cachedResponse.body, (chunkLength) => {
              bytesRead += chunkLength;
              const percent =
                  totalBytes > 0 ? (bytesRead / totalBytes) * 100 : 0;
              this.downloadProgresses.set(modelFilename, Math.round(percent));
              this.downloadSpeeds.set(
                  modelFilename,
                  `${(bytesRead / 1e6).toFixed(1)} MB / ${
                      (totalBytes / 1e6).toFixed(1)} MB`);
              this.updateCallback();
            });

        modelInput = progressStream;
      } else {
        this.updateStatus(`Downloading weights (${modelFilename})...`);

        this.downloadAbortController = new AbortController();

        const response = await fetch(modelPath, {
          signal: this.downloadAbortController.signal,
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch model: ${response.statusText}`);
        }

        const totalBytes =
            Number(response.headers.get('content-length') || '0');
        let bytesRead = 0;

        const progressStream =
            this.makeProgressStream(response.body!, (chunkLength) => {
              bytesRead += chunkLength;
              const percent =
                  totalBytes > 0 ? (bytesRead / totalBytes) * 100 : 0;
              this.downloadProgresses.set(modelFilename, Math.round(percent));
              this.downloadSpeeds.set(
                  modelFilename,
                  `${(bytesRead / 1e6).toFixed(1)} MB / ${
                      (totalBytes / 1e6).toFixed(1)} MB`);
              this.updateCallback();
            });

        const [loaderStream, cacheStream] = teeStream(progressStream);

        const cacheResponse = new Response(cacheStream, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': totalBytes.toString(),
          }
        });

        void cache.put(modelPath, cacheResponse)
            .then(() => {
              this.updateCacheSize();
            })
            .catch(err => {
              if (this.isDownloadAborted) {
                console.log('[LiteRT-LM] Cache write aborted.');
                return;
              }
              console.error('[LiteRT-LM] Cache write failed:', err);
              this.updateStatus(
                  '⚠ Cache Failed: Disk quota exceeded. (Running from memory)');
            });

        modelInput = loaderStream;
      }

      this.updateStatus(`Compiling Model (${modelFilename})...`);

      this.engine = await this.createEngine({
        model: modelInput,
        backend: Backend.GPU_ARTISAN,
        mainExecutorSettings: {
          maxNumTokens: this.settings.contextLength,
          backendConfig: {
            num_output_candidates: 1,
            wait_for_weight_uploads: true,
            num_decode_steps_per_sync: 1,
            sequence_batch_size: 0,
            supported_lora_ranks: [] as number[],
            max_top_k: Math.max(1, this.settings.topK),
            enable_decode_logits: false,
            enable_external_embeddings: false,
            use_submodel: true,
          } as GpuArtisanConfig
        },
        benchmarkEnabled: true
      });
      const loadTimeSec = (performance.now() - startTime) / 1000;
      this.metricLoadTime = `${loadTimeSec.toFixed(2)}s`;

      await onModelLoaded();

      this.isModelLoading = false;
      this.downloadProgresses.delete(modelFilename);
      this.downloadSpeeds.delete(modelFilename);
      this.updateCallback();

    } catch (err: unknown) {
      console.error(err);
      this.isModelLoading = false;
      this.downloadProgresses.delete(modelFilename);
      this.downloadSpeeds.delete(modelFilename);

      const error = err as Error;
      if (error.name === 'AbortError') {
        this.updateStatus('Model download cancelled by user.');
      } else {
        this.updateStatus(`Failed to load model: ${error.message || error}`);
      }
      this.updateCallback();
    } finally {
      this.downloadAbortController = null;
    }
  }

  cancelDownload() {
    if (this.downloadAbortController) {
      this.isDownloadAborted = true;
      console.log('[LiteRT-LM] Aborting active network model download...');
      this.downloadAbortController.abort();
    }
  }
}
