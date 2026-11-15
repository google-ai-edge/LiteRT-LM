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

import {SAMPLE_IMAGE_FILES, SAMPLES_BASE_URL} from '../app_config.js';

/**
 * Stable id of the sample album in the indexer's directory list. It is also
 * the name of the Origin Private File System directory holding the downloaded
 * photos; the underscores keep it from colliding with a user folder that
 * happens to be called "Sample album".
 */
export const SAMPLE_ALBUM_DIR_ID = '__sample_album__';

/** Display name of the sample album in the folder dropdown. */
export const SAMPLE_ALBUM_DIR_NAME = 'Sample album';

const DOWNLOAD_CONCURRENCY = 8;
const MAX_FETCH_ATTEMPTS = 3;
const RETRY_BASE_DELAY_MS = 250;
const SAMPLES_URL_KEY = 'embedding-search:samples-url';

// OPFS APIs missing from the TypeScript DOM typings used here.
interface WritableFileHandle {
  createWritable(): Promise<{
    write(data: Blob): Promise<void>; close(): Promise<void>;
  }>;
}
interface RemovableDirectoryHandle {
  removeEntry?(name: string, options?: {recursive?: boolean}): Promise<void>;
}
interface StorageWithDirectory {
  getDirectory(): Promise<FileSystemDirectoryHandle>;
}

async function getOpfsRoot(): Promise<FileSystemDirectoryHandle> {
  const storage = navigator.storage as unknown as Partial<StorageWithDirectory>;
  if (!storage.getDirectory) {
    throw new Error('This browser does not support the Origin Private File System.');
  }
  return storage.getDirectory();
}

async function fetchBlobWithRetry(
    url: string, name: string, retryBaseDelayMs: number): Promise<Blob> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= MAX_FETCH_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return await response.blob();
      }
      lastError = new Error(
          `Failed to fetch ${name} (${response.status} ${response.statusText}).`);
      // Only retry transient server or rate-limit errors.
      if (response.status < 500 && response.status !== 429) break;
    } catch (e) {
      lastError = e;
    }
    if (attempt < MAX_FETCH_ATTEMPTS) {
      await new Promise(
          (resolve) => setTimeout(resolve, retryBaseDelayMs * attempt));
    }
  }
  throw lastError;
}

/** Test seams for `SampleAlbumStore`. */
export interface SampleAlbumStoreOptions {
  /** Where to create the album directory. Defaults to the OPFS root. */
  getRoot?: () => Promise<FileSystemDirectoryHandle>;
  /** Base delay between download retries. */
  retryBaseDelayMs?: number;
}

async function hasExistingFile(
    dir: FileSystemDirectoryHandle, name: string): Promise<boolean> {
  try {
    const handle = await dir.getFileHandle(name);
    const file = await handle.getFile();
    return file.size > 0;
  } catch {
    return false;
  }
}

/**
 * Installs an out-of-the-box set of photos (the AI Edge Gallery Smart Album
 * sample pack) into the Origin Private File System. The result is a regular
 * `FileSystemDirectoryHandle`, so the indexer treats it exactly like a folder
 * the user mounted, and it survives reloads without re-downloading.
 */
export class SampleAlbumStore {
  isLoading = false;
  progressText = '';
  /** Whether the most recent install left any sample photos missing. */
  hasMissingFiles = false;

  /** Base URL from which sample images are downloaded. */
  readonly samplesBaseUrl = SAMPLES_BASE_URL;

  private installPromise: Promise<FileSystemDirectoryHandle|null>|null = null;
  private readonly getRoot: () => Promise<FileSystemDirectoryHandle>;
  private readonly retryBaseDelayMs: number;

  constructor(
      private readonly onUpdate: () => void,
      private readonly onStatusChange: (status: string) => void,
      options: SampleAlbumStoreOptions = {},
  ) {
    this.getRoot = options.getRoot ?? getOpfsRoot;
    this.retryBaseDelayMs = options.retryBaseDelayMs ?? RETRY_BASE_DELAY_MS;
  }

  /** Whether the configured `samplesBaseUrl` differs from the last installed one. */
  hasBaseUrlChanged(): boolean {
    try {
      return window.localStorage.getItem(SAMPLES_URL_KEY) !==
          this.samplesBaseUrl;
    } catch {
      return false;
    }
  }

  /** Downloads the sample images from the server into OPFS. */
  async install(): Promise<FileSystemDirectoryHandle|null> {
    if (this.installPromise) return this.installPromise;
    this.isLoading = true;
    this.onUpdate();
    this.installPromise = (async () => {
      try {
        return await this.downloadToOpfs();
      } catch (e) {
        console.error('[EmbeddingSearch] Failed to install sample album:', e);
        this.onStatusChange(
            `Failed to load sample album: ${(e as Error).message}`);
        return null;
      } finally {
        this.isLoading = false;
        this.progressText = '';
        this.installPromise = null;
        this.onUpdate();
      }
    })();
    return this.installPromise;
  }

  private async downloadToOpfs(): Promise<FileSystemDirectoryHandle> {
    const total = SAMPLE_IMAGE_FILES.length;
    this.setProgress(`Downloading sample photos 0 / ${total}…`);

    const root = await this.getRoot();
    if (this.hasBaseUrlChanged()) {
      try {
        await (root as unknown as RemovableDirectoryHandle)
            .removeEntry?.(SAMPLE_ALBUM_DIR_ID, {recursive: true});
      } catch {}
      try {
        window.localStorage.setItem(SAMPLES_URL_KEY, this.samplesBaseUrl);
      } catch {}
    }
    const dir =
        await root.getDirectoryHandle(SAMPLE_ALBUM_DIR_ID, {create: true});

    let nextIndex = 0;
    let written = 0;
    let failed = 0;
    const worker = async () => {
      while (nextIndex < total) {
        const name = SAMPLE_IMAGE_FILES[nextIndex++]!;
        try {
          if (!(await hasExistingFile(dir, name))) {
            const url = `${this.samplesBaseUrl}/${name}`;
            const blob =
                await fetchBlobWithRetry(url, name, this.retryBaseDelayMs);
            const handle = await dir.getFileHandle(name, {create: true});
            const writable = await (handle as unknown as WritableFileHandle)
                                 .createWritable();
            await writable.write(blob);
            await writable.close();
          }
          written++;
        } catch (e) {
          failed++;
          console.warn(
              `[EmbeddingSearch] Skipping sample photo "${name}":`, e);
        }
        const processed = written + failed;
        if (processed % 10 === 0 || processed === total) {
          this.setProgress(
              `Downloading sample photos ${processed} / ${total}…`);
        }
      }
    };

    const workers = Array.from(
        {length: Math.min(DOWNLOAD_CONCURRENCY, total)},
        () => worker(),
    );
    await Promise.all(workers);

    this.hasMissingFiles = failed > 0;
    if (written === 0) {
      throw new Error('Could not download any sample photos.');
    }

    this.onStatusChange(
        failed > 0 ?
            `Sample album ready: ${written} photos (${failed} failed to download).` :
            `Sample album ready: ${written} photos.`);
    return dir;
  }

  private setProgress(text: string) {
    this.progressText = text;
    this.onStatusChange(text);
    this.onUpdate();
  }
}
