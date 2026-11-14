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

import {ActivationDataType, Backend, Engine, getOrLoadGlobalLiteRtLm, GpuArtisanConfig, resolveModel, SpeculativeDecodingMode} from '@litert-lm/core';

import {teeStream} from '../tee_stream.js';

import {LocalDirectoryService} from './local_directory_service.js';
import {parseModelReference} from './model_reference.js';
import {CustomModel, EXPERIMENTAL_SETTING_KEYS, getModalityConfigs, LEGACY_MODEL_PATHS, MODALITIES, ModalityBackend, ModalityConfig, ModelSettings} from './settings_store.js';

function toBackend({backend}: ModalityConfig): Backend {
  return backend === 'cpu' ? Backend.CPU : Backend.GPU;
}

/** Returns the activation precision override, which only applies on GPU. */
function toActivationDataType({backend, precision}: ModalityConfig):
    ActivationDataType|undefined {
  if (backend === 'cpu' || precision === 'default') return undefined;
  return precision === 'fp32' ? ActivationDataType.FLOAT32 :
                                ActivationDataType.FLOAT16;
}

const SUPPRESSED_WARNING_PATTERNS = [
  /mel_filterbank\.cc:.*Missing \d+ bands\s+starting at \d+ in mel-frequency design\./,
  /tasks\.cc:.*Failed to get (?:prefill|decode) profile summary: UNIMPLEMENTED: GetProfileSummary not implemented for backend: GpuArtisan/,
];

const WASM_OOM_ERROR_PATTERN =
    /out of memory|\boom\b|cannot enlarge memory|cannot allocate|allocation failed|bad_alloc|array buffer allocation|invalid (?:typed )?array (?:buffer )?length|resource_exhausted|failed to (?:allocate|reserve) memory/i;

const WASM_ABORT_TRAP_PATTERN =
    /\b(?:RuntimeError:\s*)?(?:Aborted\([^)]*\)|unreachable|memory access out of bounds)\b/i;

/**
 * Formats an error thrown while loading or compiling a model into an actionable
 * status message without dumping raw multi-line WASM stack traces.
 */
export function formatModelLoadError(
    err: unknown,
    settings: ModelSettings&
    Partial<Pick<{customModels: CustomModel[]; localDirModels: CustomModel[]},
                 'customModels'|'localDirModels'>>): string {
  const error = err as {name?: string; message?: string} | null | undefined;
  if (error?.name === 'AbortError') {
    return 'Model download cancelled by user.';
  }
  const rawMessage = (typeof error?.message === 'string' && error.message) ?
      error.message :
      String(err);
  const firstLine = rawMessage.split('\n')[0]!.trim();
  const modalities = getModalityConfigs(settings);
  const usesCpu = MODALITIES.some(m => modalities[m].backend === 'cpu');

  if (WASM_OOM_ERROR_PATTERN.test(rawMessage)) {
    if (usesCpu) {
      return 'Failed to load model: out of WASM memory on CPU. Switch backend to GPU in Experimental settings or try a smaller model (~1.5 GB or less).';
    }
    return 'Failed to load model: out of memory. Close other tabs or try a smaller model.';
  }
  if (usesCpu &&
      (error?.name === 'RuntimeError' ||
       WASM_ABORT_TRAP_PATTERN.test(rawMessage))) {
    return 'Failed to load model: WASM aborted while loading on CPU (likely out of WASM memory). Switch backend to GPU in Experimental settings or try a smaller model (~1.5 GB or less).';
  }
  return `Failed to load model: ${firstLine || rawMessage}`;
}

/**
 * Maximum number of images in a conversation. Artisan models reserve GPU memory
 * for this many images' embeddings when the model is loaded.
 */
const MAX_NUM_IMAGES = 8;

/**
 * Returns a model's key in `ModelLoaderService.cachedModels`,
 * `downloadProgresses` and `downloadSpeeds`: its normalized URL, which is
 * also its key in the cache. File names aren't unique: many Hugging Face repos
 * name their model `model.litertlm`.
 */
export function getModelKey(path: string): string {
  try {
    return new URL(path).href;
  } catch {
    return path;
  }
}

/**
 * Local storage key for the URLs that `hf://` models (keyed by `getModelKey()`)
 * were last loaded from, which are also their keys in the cache.
 */
const HF_MODEL_URLS_KEY = 'litertlm-hf-model-urls';

/** Formats a model's size for the model list. */
function formatModelSize(bytes: number): string {
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

/**
 * Wraps `console.warn` to suppress known noisy WASM runtime warnings in the
 * chat app.
 */
export function suppressKnownWasmWarnings(
    consoleObj: Pick<Console, 'warn'> = console) {
  const flaggedWarn = consoleObj.warn as unknown as {
    __litertLmWarningsSuppressed?: boolean;
  };
  if (flaggedWarn.__litertLmWarningsSuppressed) {
    return;
  }
  const originalWarn = consoleObj.warn;
  const wrappedWarn = function(this: unknown, ...args: unknown[]) {
    if (args.length > 0 && typeof args[0] === 'string') {
      const msg = args[0];
      if (SUPPRESSED_WARNING_PATTERNS.some(pattern => pattern.test(msg))) {
        return;
      }
    }
    return originalWarn.apply(this, args);
  };
  (wrappedWarn as unknown as {__litertLmWarningsSuppressed?: boolean})
      .__litertLmWarningsSuppressed = true;
  consoleObj.warn = wrappedWarn;
}

/** Options describing a non-blocking confirmation dialog request. */
export interface ConfirmDialogOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
}

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

  loadedSettings: ModelSettings|null = null;
  /** Asks the user to confirm a destructive action. */
  confirmAction = (options: ConfirmDialogOptions) =>
      Promise.resolve(confirm(options.message));
  private readonly inMemoryModels = new Map<string, File>();

  constructor(
      private readonly updateCallback: () => void,
      private readonly updateStatus: (msg: string) => void,
      private readonly localDirService?: LocalDirectoryService,
      private readonly createEngine: typeof Engine.create = Engine.create,
      private readonly loadWasm:
          typeof getOrLoadGlobalLiteRtLm = getOrLoadGlobalLiteRtLm,
      private readonly resolve: typeof resolveModel = resolveModel) {}

  async updateCacheSize() {
    if (!window.caches) {
      this.cachedModels = new Map(Array.from(
          this.inMemoryModels, ([path, file]) => [getModelKey(path), file.size]));
      this.updateCallback();
      return;
    }
    try {
      const cache = await window.caches.open('litertlm-models');
      const keys = await cache.keys();
      const newCachedModels = new Map<string, number>();
      // hf:// models are cached under the URL they were loaded from.
      const hfModels = new Map(
          Object.entries(this.getHfModelUrls())
              .map(([uri, url]) => [url, uri.replace(/#backend=cpu$/, '')]));

      for (const request of keys) {
        if (LEGACY_MODEL_PATHS.has(request.url)) {
          // A default model's old file, which was replaced by its hf:// URI.
          await cache.delete(request);
          continue;
        }
        const response = await cache.match(request);
        if (response) {
          const key = hfModels.get(request.url) ?? request.url;
          const contentLength = response.headers.get('content-length');
          const size = contentLength ? Number(contentLength) :
                                       (await response.blob()).size;
          // A model's GPU and CPU variants share a key, so add up their sizes.
          newCachedModels.set(key, (newCachedModels.get(key) ?? 0) + size);
        }
      }

      this.cachedModels = newCachedModels;
      this.updateCallback();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to calculate model cache size:', e);
    }
  }

  needsModelReload(settings: ModelSettings): boolean {
    if (!this.engine || !this.loadedSettings) return true;
    const topKChanged = settings.samplerType !== 'greedy' &&
        this.loadedSettings.topK !== settings.topK;
    return this.loadedSettings.selectedModelPath !==
        settings.selectedModelPath ||
        this.loadedSettings.contextLength !== settings.contextLength ||
        topKChanged ||
        this.loadedSettings.enableSpeculativeDecoding !==
            settings.enableSpeculativeDecoding ||
        EXPERIMENTAL_SETTING_KEYS.some(
            key => this.loadedSettings![key] !== settings[key]);
  }

  async deleteModelFromCache(modelPath: string): Promise<boolean> {
    const confirmDelete = await this.confirmAction({
      title: 'Remove cached weights?',
      message:
          'Remove this model\'s downloaded weights from your browser cache? You can re-download them anytime.',
      confirmLabel: 'Remove Weights',
      danger: true,
    });
    if (!confirmDelete) return false;

    if (!window.caches) {
      const removed = this.inMemoryModels.delete(modelPath);
      if (removed) {
        this.updateStatus('Model cache removed successfully.');
        await this.updateCacheSize();
      }
      return removed;
    }

    try {
      const cache = await window.caches.open('litertlm-models');
      const gpuUrl = this.getCacheUrl(modelPath, 'gpu');
      const cpuHfUrl = modelPath.startsWith('hf://') ?
          this.getHfModelUrls()[this.getHfStorageKey(modelPath, 'cpu')] :
          undefined;
      const deletedGpu = await cache.delete(gpuUrl);
      const deletedCpu =
          cpuHfUrl && cpuHfUrl !== gpuUrl ? await cache.delete(cpuHfUrl) : false;
      const deleted = deletedGpu || deletedCpu;
      if (deleted) {
        this.updateStatus('Model cache removed successfully.');
        await this.updateCacheSize();
      }
      return deleted;
    } catch (e) {
      console.error('[LiteRT-LM] Failed to delete model from Cache:', e);
      return false;
    }
  }

  async clearAllCache(onDeleted?: () => void) {
    const confirmClear = await this.confirmAction({
      title: 'Clear all cached models?',
      message:
          'Delete all cached model weights stored in this browser? This will free up disk space.',
      confirmLabel: 'Clear All',
      danger: true,
    });
    if (!confirmClear) return;

    if (!window.caches) {
      this.inMemoryModels.clear();
      this.updateStatus('Model cache cleared successfully.');
      if (onDeleted) onDeleted();
      await this.updateCacheSize();
      return;
    }

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

  async loadModelWeights(
      settings: ModelSettings, onModelLoaded: () => Promise<void>) {
    if (this.isModelLoading) return;

    const modelPath = settings.selectedModelPath;
    const modelFilename = modelPath.split('/').pop() || modelPath;
    const modelKey = getModelKey(modelPath);
    const modalities = getModalityConfigs(settings);
    const textBackend = modalities.text.backend;
    const useCpuText = textBackend === 'cpu';

    this.isModelLoading = true;
    this.isDownloadAborted = false;
    this.updateStatus('Preparing runtime & model...');

    const startTime = performance.now();

    try {
      if (!this.isWasmLoaded) {
        this.updateStatus('Loading LiteRT WASM runtime...');
        suppressKnownWasmWarnings();
        const wasmPath = (import.meta.env.DEV || import.meta.env.MODE === 'selfcontained') ? './wasm' : undefined;
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

      let modelInput: ReadableStream<Uint8Array>;

      if (modelPath.startsWith('local-dir://')) {
        if (!this.localDirService) {
          throw new Error('Local directory service not available.');
        }
        this.updateStatus(`Loading local directory model ${modelFilename}...`);
        const file = await this.localDirService.getFile(modelPath);
        modelInput = file.stream();
      } else if (!window.caches) {
        const memFile = this.inMemoryModels.get(modelPath);
        if (memFile) {
          this.updateStatus(`Loading imported model (${modelFilename})...`);
          modelInput = memFile.stream();
        } else if (modelPath.startsWith('https://local-model/')) {
          throw new Error(
              'Local model file not found in cache. Please re-upload the file.');
        } else {
          const modelUrl = await this.resolveModelUrl(modelPath, textBackend);
          this.downloadProgresses.set(modelKey, 0);
          this.downloadSpeeds.set(modelKey, '0 MB / 0 MB');
          this.updateCallback();
          this.updateStatus(`Downloading weights (${modelFilename})...`);

          this.downloadAbortController = new AbortController();
          const response = await fetch(modelUrl, {
            signal: this.downloadAbortController.signal,
          });
          if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
          }
          if (!response.body) {
            throw new Error('Model response body is empty.');
          }

          const totalBytes =
              Number(response.headers.get('content-length') || '0');
          let bytesRead = 0;
          modelInput =
              this.makeProgressStream(response.body, (chunkLength) => {
                bytesRead += chunkLength;
                const percent =
                    totalBytes > 0 ? (bytesRead / totalBytes) * 100 : 0;
                this.downloadProgresses.set(modelKey, Math.round(percent));
                this.downloadSpeeds.set(
                    modelKey,
                    `${(bytesRead / 1e6).toFixed(1)} MB / ${
                        (totalBytes / 1e6).toFixed(1)} MB`);
                this.updateCallback();
              });
        }
      } else {
        const cache = await window.caches.open('litertlm-models');
        const hfStorageKey = this.getHfStorageKey(modelPath, textBackend);
        const previousUrl = this.getHfModelUrls()[hfStorageKey];
        const modelUrl = await this.resolveModelUrl(modelPath, textBackend);
        const cachedResponse = await cache.match(modelUrl);
        if (cachedResponse) {
          this.setHfModelUrl(modelPath, modelUrl, textBackend);
        }

        this.downloadProgresses.set(modelKey, 0);
        this.downloadSpeeds.set(modelKey, '0 MB / 0 MB');
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
                this.downloadProgresses.set(modelKey, Math.round(percent));
                this.downloadSpeeds.set(
                    modelKey,
                    `${(bytesRead / 1e6).toFixed(1)} MB / ${
                        (totalBytes / 1e6).toFixed(1)} MB`);
                this.updateCallback();
              });

          modelInput = progressStream;
        } else if (modelPath.startsWith('https://local-model/')) {
          throw new Error('Local model file not found in cache. Please re-upload the file.');
        } else {
          this.updateStatus(`Downloading weights (${modelFilename})...`);

          this.downloadAbortController = new AbortController();

          const response = await fetch(modelUrl, {
            signal: this.downloadAbortController.signal,
          });
          if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
          }
          if (!response.body) {
            throw new Error('Model response body is empty.');
          }

          const totalBytes =
              Number(response.headers.get('content-length') || '0');
          let bytesRead = 0;

          const progressStream =
              this.makeProgressStream(response.body, (chunkLength) => {
                bytesRead += chunkLength;
                const percent =
                    totalBytes > 0 ? (bytesRead / totalBytes) * 100 : 0;
                this.downloadProgresses.set(modelKey, Math.round(percent));
                this.downloadSpeeds.set(
                    modelKey,
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

          void cache.put(modelUrl, cacheResponse)
              .then(async () => {
                this.setHfModelUrl(modelPath, modelUrl, textBackend);
                if (previousUrl && previousUrl !== modelUrl) {
                  // The hf:// model's previous file, which it was updated from.
                  await cache.delete(previousUrl);
                }
                await this.updateCacheSize();
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
      }

      this.updateStatus(`Compiling Model (${modelFilename})...`);

      this.engine = await this.createEngine({
        model: modelInput,
        // Without a backend, Engine.create() picks GPU or GPU_ARTISAN.
        ...(useCpuText ? {backend: Backend.CPU} : {}),
        activationDataType: toActivationDataType(modalities.text),
        mainExecutorSettings: useCpuText ? {
          maxNumTokens: settings.contextLength,
        } : {
          maxNumTokens: settings.contextLength,
          // Artisan (GPU_ARTISAN) models reserve GPU memory for the embeddings
          // of this many images per conversation up front, and can't take
          // images if it's unset.
          maxNumImages: MAX_NUM_IMAGES,
          backendConfig: {
            num_output_candidates: 1,
            wait_for_weight_uploads: true,
            num_decode_steps_per_sync: 3,
            sequence_batch_size: 0,
            supported_lora_ranks: [] as number[],
            max_top_k: Math.max(1, settings.topK),
            enable_decode_logits: false,
            enable_external_embeddings: false,
            use_submodel: false,
            use_autosized_ringbuffers: true,
          } as GpuArtisanConfig,
        },
        // Always enable image input. On web, only one vision signature is
        // loaded by default to fit in the wasm heap.
        visionBackend: toBackend(modalities.vision),
        visionActivationDataType: toActivationDataType(modalities.vision),
        // Always enable audio input. On GPU, the audio encoder's weights are
        // released from the wasm heap after they are uploaded.
        audioBackend: toBackend(modalities.audio),
        audioActivationDataType: toActivationDataType(modalities.audio),
        // The drafter can only be loaded when the engine is created, so this
        // can't be changed per session. ADAPTIVE only speculates while it's
        // faster than decoding normally. Models without a drafter can't use
        // speculative decoding.
        ...(settings.enableSpeculativeDecoding ? {
          speculativeDecodingConfig: {mode: SpeculativeDecodingMode.ADAPTIVE}
        } :
                                     {}),
        benchmarkEnabled: true
      });

      this.loadedSettings = structuredClone(settings);

      const loadTimeSec = (performance.now() - startTime) / 1000;
      this.metricLoadTime = `${loadTimeSec.toFixed(2)}s`;

      await onModelLoaded();

      this.isModelLoading = false;
      this.downloadProgresses.delete(modelKey);
      this.downloadSpeeds.delete(modelKey);
      this.updateCallback();

    } catch (err: unknown) {
      console.error(err);
      this.isModelLoading = false;
      this.downloadProgresses.delete(modelKey);
      this.downloadSpeeds.delete(modelKey);

      this.updateStatus(formatModelLoadError(err, settings));
      this.updateCallback();
    } finally {
      this.downloadAbortController = null;
    }
  }

  async importCustomModel(file: File): Promise<CustomModel> {
    const path = `https://local-model/${file.name}`;
    const customModel: CustomModel = {
      name: file.name.replace(/\.litertlm$/, ''),
      filename: file.name,
      path,
      size: `${(file.size / 1e9).toFixed(2)} GB`
    };

    if (!window.caches) {
      this.inMemoryModels.set(path, file);
      this.cachedModels.set(getModelKey(path), file.size);
      this.updateStatus(`Local model ${file.name} imported.`);
      this.updateCallback();
      return customModel;
    }

    const cache = await window.caches.open('litertlm-models');

    this.updateStatus(`Importing local model ${file.name}...`);
    this.isModelLoading = true;
    this.updateCallback();

    try {
      const response = new Response(file, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': file.size.toString(),
        }
      });
      await cache.put(path, response);

      await this.updateCacheSize();
      this.updateStatus(`Local model ${file.name} imported.`);
      return customModel;
    } catch (e) {
      console.error('[LiteRT-LM] Failed to import custom model:', e);
      this.updateStatus(`Failed to import model: ${(e as Error).message}`);
      throw e;
    } finally {
      this.isModelLoading = false;
      this.updateCallback();
    }
  }

  /**
   * Looks up a model that the user entered as a URL, `hf://` URI or Hugging
   * Face repo (`<org>/<repo>`), and returns it as a custom model. Nothing is
   * downloaded until the model is loaded.
   *
   * A Hugging Face model is pinned to the file and commit that it resolves to
   * now, so it keeps loading the same file (from the cache, once downloaded)
   * until it's added again.
   */
  async addRemoteModel(
      text: string,
      textBackend: ModalityBackend = 'gpu'): Promise<CustomModel> {
    try {
      const reference = parseModelReference(text);
      if (!reference) {
        throw new Error(
            `"${text.trim()}" isn't a URL or a Hugging Face repo (org/repo)`);
      }
      this.updateStatus(`Looking up ${reference}...`);
      const {huggingFace} = await this.resolveFor(reference, textBackend);
      let model: CustomModel;
      if (huggingFace) {
        const {repo, path, source} = huggingFace;
        const filename = path.split('/').pop() || path;
        const repoName = repo.split('/').pop() || repo;
        const stem = filename.replace(/\.litertlm$/, '');
        model = {
          name: stem === repoName || stem === 'model' ? repoName :
                                                        `${repoName} (${stem})`,
          filename,
          path: source.url,
          size: source.size === undefined ? '' : formatModelSize(source.size),
        };
      } else {
        const filename =
            new URL(reference).pathname.split('/').pop() || reference;
        model = {
          name: filename.replace(/\.litertlm$/, ''),
          filename,
          path: reference,
          size: '',
        };
      }
      this.updateStatus(`Added ${model.name}.`);
      return model;
    } catch (e) {
      console.error('[LiteRT-LM] Failed to add model:', e);
      this.updateStatus(`Failed to add model: ${(e as Error).message}`);
      throw e;
    }
  }

  /**
   * Returns the URL to download and cache a model from. A `hf://` model is
   * resolved to the file it points to now, e.g. on its repo's `web` branch, so
   * a new version is downloaded when the repo is updated. If that fails, e.g.
   * offline, the URL it was last loaded from is used, so that it still loads
   * from the cache.
   */
  private async resolveModelUrl(
      modelPath: string, textBackend: ModalityBackend = 'gpu'): Promise<string> {
    if (!modelPath.startsWith('hf://')) return modelPath;
    this.updateStatus(`Looking up ${modelPath}...`);
    const storageKey = this.getHfStorageKey(modelPath, textBackend);
    try {
      const {huggingFace} = await this.resolveFor(modelPath, textBackend);
      if (huggingFace) return huggingFace.source.url;
    } catch (e) {
      const lastUrl = this.getHfModelUrls()[storageKey];
      if (!lastUrl) throw e;
      console.warn(
          `[LiteRT-LM] Failed to look up ${modelPath}, loading ${lastUrl}:`, e);
      return lastUrl;
    }
    throw new Error(`${modelPath} isn't a Hugging Face model`);
  }

  /** Resolves a model to the file that `textBackend` can run. */
  private resolveFor(reference: string, textBackend: ModalityBackend) {
    // Without a backend, resolveModel prefers -gpu.litertlm and falls back to
    // .litertlm, which Engine.create() also does.
    return textBackend === 'cpu' ?
        this.resolve(reference, {backend: Backend.CPU}) :
        this.resolve(reference);
  }

  private getHfStorageKey(
      modelPath: string, textBackend: ModalityBackend = 'gpu'): string {
    const base = getModelKey(modelPath);
    return textBackend === 'cpu' ? `${base}#backend=cpu` : base;
  }

  /** Returns the URL a model is cached under. */
  private getCacheUrl(
      modelPath: string, textBackend: ModalityBackend = 'gpu'): string {
    return this.getHfModelUrls()[this.getHfStorageKey(modelPath, textBackend)] ??
        modelPath;
  }

  /** Returns the URLs that `hf://` models were last loaded from. */
  private getHfModelUrls(): Record<string, string> {
    try {
      const urls: unknown =
          JSON.parse(window.localStorage.getItem(HF_MODEL_URLS_KEY) ?? '{}');
      return typeof urls === 'object' && urls !== null ?
          urls as Record<string, string> :
          {};
    } catch {
      return {};
    }
  }

  private setHfModelUrl(
      modelPath: string, url: string, textBackend: ModalityBackend = 'gpu') {
    if (!modelPath.startsWith('hf://')) return;
    const urls = this.getHfModelUrls();
    urls[this.getHfStorageKey(modelPath, textBackend)] = url;
    window.localStorage.setItem(HF_MODEL_URLS_KEY, JSON.stringify(urls));
  }

  cancelDownload() {
    if (this.downloadAbortController) {
      this.isDownloadAborted = true;
      console.log('[LiteRT-LM] Aborting active network model download...');
      this.downloadAbortController.abort();
    }
  }
}
