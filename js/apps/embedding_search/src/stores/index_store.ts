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

import {EmbeddingEngine, EmbeddingInput, EmbeddingOptions, InputOverflowStrategy} from '@litert-lm/core';

import {applyDocumentTemplate, PromptTemplateId} from '../app_config.js';
import {classifyFileName, FileCategory, toSupportedMediaBlob} from '../utils/media_types.js';
import {SAMPLE_ALBUM_DIR_NAME} from './sample_album_store.js';

export type {FileCategory} from '../utils/media_types.js';

declare global {
  interface Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
  }
  interface FileSystemHandle {
    queryPermission(descriptor?: {mode?: 'read' | 'readwrite'}): Promise<PermissionState>;
    requestPermission(descriptor?: {mode?: 'read' | 'readwrite'}): Promise<PermissionState>;
  }
  interface FileSystemDirectoryHandle {
    values(): AsyncIterable<FileSystemHandle>;
  }
}

/** A supported file found while scanning a mounted directory. */
export interface DiscoveredFile {
  name: string;
  path: string;
  category: FileCategory;
  mimeType: string;
  size: number;
  lastModified: number;
  handle: FileSystemFileHandle;
}

/** A file together with its embedding(s), as stored in the index. */
export interface IndexedDocument {
  path: string;
  name: string;
  type: FileCategory;
  mimeType: string;
  size: number;
  lastModified: number;
  /** Embedding of an image or audio file. Empty for text files. */
  embedding: Float32Array;
  /** Embeddings of a text file, keyed by the prompt template used. */
  textEmbeddings?: Partial<Record<PromptTemplateId, Float32Array>>;
  previewText?: string;
}

/**
 * Returns the embedding to compare `doc` against under `template`, or
 * undefined if `doc` has not been embedded for that template yet.
 */
export function embeddingFor(
    doc: IndexedDocument, template: PromptTemplateId): Float32Array|undefined {
  if (doc.type === 'text') return doc.textEmbeddings?.[template];
  return doc.embedding.length > 0 ? doc.embedding : undefined;
}

/** Live progress of an indexing run, used to drive the progress UI. */
export interface IndexProgress {
  isIndexing: boolean;
  current: number;
  total: number;
  currentFile: string;
  percentage: number;
  startedAt: number;
  embeddedCount: number;
  failedCount: number;
  embedMs: number;
}

/** Timing summary of the most recent indexing run for a folder. */
export interface IndexRunStats {
  embeddedCount: number;
  skippedCount: number;
  failedCount: number;
  wallMs: number;
  embedMs: number;
  finishedAt: number;
}

/** A folder the user has mounted, or the sample album unpacked into OPFS. */
export interface SavedDirectory {
  id: string;
  name: string;
  handle: FileSystemDirectoryHandle;
}

/** Constructor options for `IndexStore`. */
export interface IndexStoreOptions {
  dbName?: string;
}

const IGNORED_DIRECTORIES = new Set([
  'node_modules', '.git', '.svn', '.hg', 'dist', 'build', 'out',
  'bazel-bin', 'bazel-out', 'bazel-genfiles', '__pycache__', '.vscode',
]);

const SAVE_EVERY_N_FILES = 25;
const EMBED_BATCH_SIZE = 4;
const MAX_CONSECUTIVE_FAILURES = 5;
const PROGRESS_NOTIFY_INTERVAL_MS = 100;

/** Default IndexedDB database name. */
export const DEFAULT_INDEX_DB_NAME = 'embedding-search-index-db';

const DB_VERSION = 4;
const STORE_DIRECTORIES = 'directories';  // SavedDirectory, keyPath 'id'
const STORE_SETTINGS = 'settings';        // out-of-line; 'target-dir' -> id
const STORE_DIR_META = 'dir_indexes';     // DirectoryIndexMeta, keyPath 'dirId'
const STORE_DOCS = 'docs';                // StoredDocument, keyPath [dirId, path]
const SETTING_TARGET_DIR = 'target-dir';

interface DirectoryIndexMeta {
  dirId: string;
  lastUpdated: number;
  lastRun?: IndexRunStats;
}

interface StoredDocument extends IndexedDocument {
  dirId: string;
}

interface LoadedFile {
  item: DiscoveredFile;
  input?: EmbeddingInput;
  previewText?: string;
  error?: unknown;
}

interface EmbedResult {
  embedding?: Float32Array;
  error?: unknown;
}

function docsRange(dirId: string): IDBKeyRange {
  return IDBKeyRange.bound([dirId], [dirId, []]);
}

function newDirectoryId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `dir-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function transactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const dbPromises = new Map<string, Promise<IDBDatabase>>();

function openIndexDB(name: string): Promise<IDBDatabase> {
  const cached = dbPromises.get(name);
  if (cached) return cached;
  const promise = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(name, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      for (const storeName of Array.from(db.objectStoreNames)) {
        db.deleteObjectStore(storeName);
      }
      db.createObjectStore(STORE_DIRECTORIES, {keyPath: 'id'});
      db.createObjectStore(STORE_SETTINGS);
      db.createObjectStore(STORE_DIR_META, {keyPath: 'dirId'});
      db.createObjectStore(STORE_DOCS, {keyPath: ['dirId', 'path']});
    };
    request.onsuccess = () => {
      const db = request.result;
      const forget = () => {
        db.close();
        if (dbPromises.get(name) === promise) dbPromises.delete(name);
      };
      db.onversionchange = forget;
      db.onclose = forget;
      resolve(db);
    };
    request.onerror = () => {
      if (dbPromises.get(name) === promise) dbPromises.delete(name);
      reject(request.error);
    };
  });
  dbPromises.set(name, promise);
  return promise;
}

function emptyProgress(): IndexProgress {
  return {
    isIndexing: false,
    current: 0,
    total: 0,
    currentFile: '',
    percentage: 0,
    startedAt: 0,
    embeddedCount: 0,
    failedCount: 0,
    embedMs: 0,
  };
}

/**
 * Service managing target directory traversal, file categorization,
 * vector indexing, and IndexedDB caching.
 */
export class IndexStore {
  private dirHandle: FileSystemDirectoryHandle | null = null;
  private cancelRequested = false;
  private loadGeneration = 0;
  private scanGeneration = 0;
  private readonly dbName: string;

  private discovered: DiscoveredFile[] = [];
  private fileHandles = new Map<string, FileSystemFileHandle>();
  private dirtyPaths = new Set<string>();
  private deletedPaths = new Set<string>();
  private lastProgressNotify = 0;

  isSupported = typeof window !== 'undefined' && 'showDirectoryPicker' in window;
  isAuthorized = false;
  targetDirId = '';
  targetDirName = '';
  isScanning = false;

  savedDirectories = new Map<string, SavedDirectory>();
  indexedDocuments = new Map<string, IndexedDocument>();
  progress: IndexProgress = emptyProgress();
  lastRun: IndexRunStats | null = null;
  readonly ready: Promise<void>;

  constructor(
      private readonly onUpdate: () => void,
      private readonly onStatusChange: (status: string) => void,
      private readonly onDirectoryChanged?: () => void,
      options: IndexStoreOptions = {},
  ) {
    this.dbName = options.dbName ?? DEFAULT_INDEX_DB_NAME;
    this.ready = this.restoreLastDirectory();
  }

  get discoveredFiles(): DiscoveredFile[] {
    return this.discovered;
  }
  set discoveredFiles(files: DiscoveredFile[]) {
    this.discovered = files;
    this.fileHandles = new Map(files.map((f) => [f.path, f.handle]));
  }

  get needsReauthorization(): boolean {
    return !!this.targetDirId && !this.isAuthorized;
  }

  handleFor(path: string): FileSystemFileHandle|null {
    return this.fileHandles.get(path) ?? null;
  }

  private openDb(): Promise<IDBDatabase> {
    return openIndexDB(this.dbName);
  }

  private async restoreLastDirectory() {
    try {
      const db = await this.openDb();
      const tx = db.transaction([STORE_DIRECTORIES, STORE_SETTINGS], 'readonly');
      const dirsReq = tx.objectStore(STORE_DIRECTORIES).getAll();
      const targetReq = tx.objectStore(STORE_SETTINGS).get(SETTING_TARGET_DIR);
      await transactionDone(tx);

      this.savedDirectories = new Map(
          (dirsReq.result as SavedDirectory[]).map((dir) => [dir.id, dir]));
      const targetId = targetReq.result as string | undefined;
      const target = targetId ? this.savedDirectories.get(targetId) : undefined;
      if (!target) return;

      this.dirHandle = target.handle;
      this.targetDirId = target.id;
      this.targetDirName = target.name;
      await this.loadDirectoryIndex(target.id);
      const perm = await target.handle.queryPermission({mode: 'read'});
      this.isAuthorized = perm === 'granted';
      if (this.isAuthorized) {
        await this.scanDirectory();
      }
      this.onUpdate();
    } catch (e) {
      console.error('[EmbeddingSearch] Failed to load stored target directory handle:', e);
    }
  }

  async selectSavedDirectory(id: string): Promise<boolean> {
    const dir = this.savedDirectories.get(id);
    if (!dir) return false;
    try {
      if (!(await this.ensureReadPermission(dir.handle, dir.name))) {
        return false;
      }
      await this.useDirectory(dir.handle, {id: dir.id, name: dir.name});
      return true;
    } catch (e) {
      console.error(`[EmbeddingSearch] Failed to switch to directory "${dir.name}":`, e);
      this.onStatusChange(`Failed to open "${dir.name}": ${errorMessage(e)}`);
      return false;
    }
  }

  async reauthorize(): Promise<boolean> {
    const handle = this.dirHandle;
    if (!handle) return false;
    if (this.isAuthorized) return true;
    try {
      const granted = await this.ensureReadPermission(handle, this.targetDirName);
      if (this.dirHandle !== handle || !granted) return false;
      this.isAuthorized = true;
      await this.scanDirectory();
      this.onUpdate();
      return true;
    } catch (e) {
      console.error('[EmbeddingSearch] Failed to re-authorize directory:', e);
      this.onStatusChange(
          `Failed to open "${this.targetDirName}": ${errorMessage(e)}`);
      return false;
    }
  }

  private async ensureReadPermission(
      handle: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    let perm = await handle.queryPermission({mode: 'read'});
    if (perm !== 'granted') {
      perm = await handle.requestPermission({mode: 'read'});
    }
    if (perm !== 'granted') {
      this.onStatusChange(`Read access to "${name}" was not granted.`);
      this.onUpdate();
      return false;
    }
    return true;
  }

  async pickNewDirectory(): Promise<boolean> {
    if (!this.isSupported || !window.showDirectoryPicker) {
      this.onStatusChange(
          'File System Access API is not supported in this browser.');
      return false;
    }
    try {
      const handle = await window.showDirectoryPicker();
      await this.useDirectory(handle);
      return true;
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        this.onStatusChange('Directory selection cancelled.');
      } else {
        console.error('[EmbeddingSearch] Failed to mount directory:', e);
        this.onStatusChange(`Failed to select directory: ${errorMessage(e)}`);
      }
      return false;
    }
  }

  async useDirectory(
      handle: FileSystemDirectoryHandle,
      options: {id?: string; name?: string} = {}): Promise<void> {
    const dir = await this.registerDirectory(handle, options);
    this.scanGeneration++;
    this.dirHandle = handle;
    this.targetDirId = dir.id;
    this.targetDirName = dir.name;

    try {
      const db = await this.openDb();
      const tx = db.transaction([STORE_DIRECTORIES, STORE_SETTINGS], 'readwrite');
      tx.objectStore(STORE_DIRECTORIES).put(dir);
      tx.objectStore(STORE_SETTINGS).put(dir.id, SETTING_TARGET_DIR);
      await transactionDone(tx);
    } catch (e) {
      console.warn('[EmbeddingSearch] Could not persist directory handle:', e);
    }
    if (this.targetDirId !== dir.id) return;

    this.isAuthorized = true;
    this.discoveredFiles = [];

    await this.loadDirectoryIndex(dir.id);
    if (this.targetDirId !== dir.id) return;

    this.onDirectoryChanged?.();
    await this.scanDirectory();
    this.onUpdate();
  }

  private async registerDirectory(
      handle: FileSystemDirectoryHandle,
      options: {id?: string; name?: string}): Promise<SavedDirectory> {
    let id = options.id;
    if (!id) {
      for (const existing of this.savedDirectories.values()) {
        if (await this.isSameDirectory(existing.handle, handle)) {
          id = existing.id;
          break;
        }
      }
    }
    if (!id) id = newDirectoryId();
    const existing = this.savedDirectories.get(id);
    const name = options.name ?? existing?.name ??
        this.uniqueDirectoryName(handle.name, id);
    const dir: SavedDirectory = {id, name, handle};
    this.savedDirectories.set(id, dir);
    return dir;
  }

  private async isSameDirectory(
      a: FileSystemDirectoryHandle, b: FileSystemDirectoryHandle):
      Promise<boolean> {
    if (a === b) return true;
    if (typeof a.isSameEntry !== 'function') return false;
    try {
      return await a.isSameEntry(b);
    } catch {
      return false;
    }
  }

  private uniqueDirectoryName(base: string, id: string): string {
    const taken = new Set([SAMPLE_ALBUM_DIR_NAME]);
    for (const dir of this.savedDirectories.values()) {
      if (dir.id !== id) taken.add(dir.name);
    }
    let name = base;
    for (let n = 2; taken.has(name); n++) {
      name = `${base} (${n})`;
    }
    return name;
  }

  async removeDirectory(id: string): Promise<void> {
    const name = this.savedDirectories.get(id)?.name ?? id;
    this.savedDirectories.delete(id);
    const wasActive = this.targetDirId === id;
    if (wasActive) {
      this.scanGeneration++;
      this.loadGeneration++;
    }

    try {
      const db = await this.openDb();
      const tx = db.transaction(
          [STORE_DIRECTORIES, STORE_SETTINGS, STORE_DIR_META, STORE_DOCS],
          'readwrite');
      tx.objectStore(STORE_DIRECTORIES).delete(id);
      if (wasActive) {
        tx.objectStore(STORE_SETTINGS).delete(SETTING_TARGET_DIR);
      }
      tx.objectStore(STORE_DIR_META).delete(id);
      tx.objectStore(STORE_DOCS).delete(docsRange(id));
      await transactionDone(tx);
    } catch (e) {
      console.error(`[EmbeddingSearch] Failed to remove directory "${name}":`, e);
    }

    if (wasActive) {
      this.dirHandle = null;
      this.targetDirId = '';
      this.targetDirName = '';
      this.isAuthorized = false;
      this.discoveredFiles = [];
      this.indexedDocuments.clear();
      this.dirtyPaths.clear();
      this.deletedPaths.clear();
      this.lastRun = null;
      this.onDirectoryChanged?.();
    }
    this.onStatusChange(`Removed folder "${name}".`);
    this.onUpdate();
  }

  async scanDirectory(): Promise<void> {
    const handle = this.dirHandle;
    if (!handle || !this.isAuthorized) return;
    const generation = ++this.scanGeneration;
    const isStale = () =>
        generation !== this.scanGeneration || this.dirHandle !== handle;

    this.onStatusChange(`Scanning "${this.targetDirName}" for compatible files...`);
    this.isScanning = true;
    this.onUpdate();
    const files: DiscoveredFile[] = [];

    try {
      await this.traverseFolder(handle, '', files, isStale);
      if (isStale()) return;
      files.sort((a, b) => a.path.localeCompare(b.path));
      this.discoveredFiles = files;

      const currentPaths = new Set(files.map((f) => f.path));
      let hadPruning = false;
      for (const path of Array.from(this.indexedDocuments.keys())) {
        if (!currentPaths.has(path)) {
          this.indexedDocuments.delete(path);
          this.dirtyPaths.delete(path);
          this.deletedPaths.add(path);
          hadPruning = true;
        }
      }
      if (hadPruning) {
        await this.saveCurrentDirectoryIndex();
      }

      this.onStatusChange(
          `Discovered ${files.length} file(s) across text, images, and audio.`);
      this.onUpdate();
    } catch (e) {
      if (isStale()) return;
      console.error('[EmbeddingSearch] Traversal failed:', e);
      this.onStatusChange(`Failed to scan files: ${errorMessage(e)}`);
    } finally {
      if (generation === this.scanGeneration) {
        this.isScanning = false;
        this.onUpdate();
      }
    }
  }

  private async traverseFolder(
      dir: FileSystemDirectoryHandle,
      currentPath: string,
      outFiles: DiscoveredFile[],
      isStale: () => boolean,
  ): Promise<void> {
    for await (const entry of dir.values()) {
      if (isStale()) return;
      if (entry.name.startsWith('.')) continue;

      const relPath = currentPath ? `${currentPath}/${entry.name}` : entry.name;
      if (entry.kind === 'directory') {
        if (IGNORED_DIRECTORIES.has(entry.name.toLowerCase())) continue;
        try {
          await this.traverseFolder(
              entry as FileSystemDirectoryHandle, relPath, outFiles, isStale);
        } catch (e) {
          console.warn(
              `[EmbeddingSearch] Skipping inaccessible folder "${relPath}":`, e);
        }
      } else if (entry.kind === 'file') {
        const classified = classifyFileName(entry.name);
        if (!classified) continue;
        try {
          const fileHandle = entry as FileSystemFileHandle;
          const file = await fileHandle.getFile();
          outFiles.push({
            name: entry.name,
            path: relPath,
            category: classified.category,
            mimeType: classified.mimeType,
            size: file.size,
            lastModified: file.lastModified,
            handle: fileHandle,
          });
        } catch (e) {
          console.warn(
              `[EmbeddingSearch] Skipping inaccessible file "${relPath}":`, e);
        }
      }
    }
  }

  private isUnchanged(item: DiscoveredFile): boolean {
    const existing = this.indexedDocuments.get(item.path);
    return !!existing && existing.lastModified === item.lastModified &&
        existing.size === item.size;
  }

  isUpToDate(item: DiscoveredFile, template: PromptTemplateId): boolean {
    if (!this.isUnchanged(item)) return false;
    return embeddingFor(this.indexedDocuments.get(item.path)!, template) !==
        undefined;
  }

  countPending(template: PromptTemplateId): number {
    let count = 0;
    for (const item of this.discovered) {
      if (!this.isUpToDate(item, template)) count++;
    }
    return count;
  }

  private notifyProgress(force = false) {
    const now = performance.now();
    if (!force && now - this.lastProgressNotify < PROGRESS_NOTIFY_INTERVAL_MS) {
      return;
    }
    this.lastProgressNotify = now;
    this.onUpdate();
  }

  async startIndexing(engine: EmbeddingEngine, template: PromptTemplateId):
      Promise<void> {
    if (this.progress.isIndexing) return;

    this.cancelRequested = false;
    const files = this.discoveredFiles;
    const dirId = this.targetDirId;
    const total = files.length;
    if (total === 0) {
      this.onStatusChange('No files to index. Mount a folder with files first.');
      return;
    }

    this.progress = {
      ...emptyProgress(),
      isIndexing: true,
      total,
      currentFile: 'Preparing...',
      startedAt: performance.now(),
    };
    this.notifyProgress(true);

    const pending = files.filter((item) => !this.isUpToDate(item, template));
    const skippedCount = total - pending.length;
    this.progress.current = skippedCount;
    this.progress.percentage = Math.round((skippedCount / total) * 100);

    const batches = batchByCategory(pending, EMBED_BATCH_SIZE);
    let unsavedCount = 0;
    let failedCount = 0;
    let consecutiveFailures = 0;
    let lastError: unknown = null;
    let cancelled = false;
    let aborted = false;

    let nextBatch: Promise<LoadedFile[]> = batches.length > 0 ?
        this.loadBatch(batches[0]!, template) :
        Promise.resolve([]);
    try {
      for (let b = 0; b < batches.length; b++) {
        const loaded = await nextBatch;
        if (this.cancelRequested || this.targetDirId !== dirId) {
          cancelled = true;
          break;
        }
        if (b + 1 < batches.length) {
          nextBatch = this.loadBatch(batches[b + 1]!, template);
        }
        this.progress.currentFile = loaded[0]!.item.path;
        this.notifyProgress();

        const results = await this.embedBatch(engine, loaded);
        if (this.targetDirId !== dirId) {
          cancelled = true;
          break;
        }

        for (let i = 0; i < loaded.length; i++) {
          const {item, previewText} = loaded[i]!;
          const result = results[i]!;
          this.progress.current++;
          this.progress.percentage =
              Math.round((this.progress.current / total) * 100);
          this.progress.currentFile = item.path;
          if (!result.embedding) {
            failedCount++;
            consecutiveFailures++;
            lastError = result.error;
            this.progress.failedCount = failedCount;
            console.warn(
                `[EmbeddingSearch] Skipping failed file "${item.path}":`,
                result.error);
            continue;
          }
          consecutiveFailures = 0;
          this.storeEmbedding(item, template, result.embedding, previewText);
          this.progress.embeddedCount++;
          unsavedCount++;
        }
        this.notifyProgress();

        if (unsavedCount >= SAVE_EVERY_N_FILES) {
          unsavedCount = 0;
          await this.saveCurrentDirectoryIndex();
        }
        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          aborted = true;
          break;
        }
      }
    } finally {
      this.progress.isIndexing = false;
      this.progress.currentFile = '';
    }

    const wallMs = performance.now() - this.progress.startedAt;
    const embedded = this.progress.embeddedCount;
    if (embedded > 0) {
      this.lastRun = {
        embeddedCount: embedded,
        skippedCount,
        failedCount,
        wallMs,
        embedMs: this.progress.embedMs,
        finishedAt: Date.now(),
      };
    }

    if (this.targetDirId === dirId) {
      await this.saveCurrentDirectoryIndex();
    }

    const failedSuffix = failedCount > 0 ? ` ${failedCount} file(s) failed.` : '';
    if (aborted) {
      this.onStatusChange(`Indexing stopped: ${errorMessage(lastError)}`);
    } else if (cancelled) {
      this.onStatusChange(`Indexing stopped by user.${failedSuffix}`);
    } else if (embedded > 0) {
      this.onStatusChange(
          `Embedded ${embedded} file(s) in ${(wallMs / 1000).toFixed(1)}s.${
              failedSuffix}`);
    } else {
      this.onStatusChange(
          `Index is up to date: ${this.indexedDocuments.size} / ${total} files.${
              failedSuffix}`);
    }
    this.onUpdate();
  }

  private loadBatch(batch: DiscoveredFile[], template: PromptTemplateId):
      Promise<LoadedFile[]> {
    return Promise.all(batch.map(async (item): Promise<LoadedFile> => {
      try {
        const file = await item.handle.getFile();
        if (item.category === 'text') {
          const text = await file.text();
          return {
            item,
            input: applyDocumentTemplate(template, text),
            previewText: text.slice(0, 320).trim(),
          };
        }
        const blob =
            await toSupportedMediaBlob(file, item.category, item.mimeType);
        return {item, input: blob};
      } catch (error) {
        return {item, error};
      }
    }));
  }

  private async embedBatch(engine: EmbeddingEngine, loaded: LoadedFile[]):
      Promise<EmbedResult[]> {
    const results: EmbedResult[] = loaded.map((l) => ({error: l.error}));
    const ready: number[] = [];
    loaded.forEach((l, i) => {
      if (l.input !== undefined) ready.push(i);
    });
    if (ready.length === 0) return results;

    const isText = loaded[ready[0]!]!.item.category === 'text';
    const options: EmbeddingOptions = isText ?
        {normalize: true, inputOverflowStrategy: InputOverflowStrategy.TRUNCATE} :
        {normalize: true};
    const inputs = ready.map((i) => loaded[i]!.input!);
    const start = performance.now();
    try {
      if (inputs.length > 1 && typeof engine.computeEmbeddingBatch === 'function') {
        try {
          const responses = await engine.computeEmbeddingBatch(inputs, options);
          if (responses.length === inputs.length) {
            ready.forEach((i, k) => {
              results[i] = {embedding: new Float32Array(responses[k]!.embedding)};
            });
            return results;
          }
        } catch (e) {
          console.warn(
              '[EmbeddingSearch] Batch embedding failed; retrying individually:',
              e);
        }
      }
      for (const i of ready) {
        try {
          const resp = await engine.computeEmbedding(loaded[i]!.input!, options);
          results[i] = {embedding: new Float32Array(resp.embedding)};
        } catch (error) {
          results[i] = {error};
        }
      }
      return results;
    } finally {
      this.progress.embedMs += performance.now() - start;
    }
  }

  private storeEmbedding(
      item: DiscoveredFile, template: PromptTemplateId,
      embedding: Float32Array, previewText: string|undefined) {
    const isText = item.category === 'text';
    const previous = isText && this.isUnchanged(item) ?
        this.indexedDocuments.get(item.path)!.textEmbeddings :
        undefined;
    const doc: IndexedDocument = {
      path: item.path,
      name: item.name,
      type: item.category,
      mimeType: item.mimeType,
      size: item.size,
      lastModified: item.lastModified,
      embedding: isText ? new Float32Array(0) : embedding,
      textEmbeddings: isText ? {...previous, [template]: embedding} : undefined,
      previewText,
    };
    this.indexedDocuments.set(item.path, doc);
    this.dirtyPaths.add(item.path);
    this.deletedPaths.delete(item.path);
  }

  cancelIndexing() {
    if (this.progress.isIndexing) {
      this.cancelRequested = true;
      this.onStatusChange('Cancelling indexing...');
    }
  }

  async loadDirectoryIndex(dirId: string): Promise<void> {
    const generation = ++this.loadGeneration;
    this.indexedDocuments.clear();
    this.dirtyPaths.clear();
    this.deletedPaths.clear();
    this.lastRun = null;
    if (!dirId) return;
    try {
      const db = await this.openDb();
      const tx = db.transaction([STORE_DIR_META, STORE_DOCS], 'readonly');
      const metaReq = tx.objectStore(STORE_DIR_META).get(dirId);
      const docsReq = tx.objectStore(STORE_DOCS).getAll(docsRange(dirId));
      await transactionDone(tx);
      if (generation !== this.loadGeneration ||
          (this.targetDirId && this.targetDirId !== dirId)) {
        return;
      }
      const loaded = new Map<string, IndexedDocument>();
      for (const {dirId: _, ...doc} of docsReq.result as StoredDocument[]) {
        loaded.set(doc.path, doc);
      }
      this.indexedDocuments = loaded;
      const meta = metaReq.result as DirectoryIndexMeta | undefined;
      this.lastRun = meta?.lastRun ?? null;
    } catch (e) {
      console.warn(`[EmbeddingSearch] Failed to load index for "${dirId}":`, e);
    }
  }

  async saveCurrentDirectoryIndex(): Promise<void> {
    const dirId = this.targetDirId;
    if (!dirId) return;
    const dirty = Array.from(this.dirtyPaths);
    const deleted = Array.from(this.deletedPaths);
    this.dirtyPaths.clear();
    this.deletedPaths.clear();
    try {
      const db = await this.openDb();
      const tx = db.transaction([STORE_DIR_META, STORE_DOCS], 'readwrite');
      const docs = tx.objectStore(STORE_DOCS);
      for (const path of deleted) {
        docs.delete([dirId, path]);
      }
      for (const path of dirty) {
        const doc = this.indexedDocuments.get(path);
        if (doc) docs.put({dirId, ...doc} satisfies StoredDocument);
      }
      const meta: DirectoryIndexMeta = {
        dirId,
        lastUpdated: Date.now(),
        lastRun: this.lastRun ?? undefined,
      };
      tx.objectStore(STORE_DIR_META).put(meta);
      await transactionDone(tx);
    } catch (e) {
      console.error('[EmbeddingSearch] Failed to save directory index:', e);
      if (this.targetDirId === dirId) {
        for (const path of dirty) this.dirtyPaths.add(path);
        for (const path of deleted) this.deletedPaths.add(path);
      }
    }
  }

  async clearIndex(): Promise<void> {
    this.loadGeneration++;
    const dirId = this.targetDirId;
    if (dirId) {
      try {
        const db = await this.openDb();
        const tx = db.transaction([STORE_DIR_META, STORE_DOCS], 'readwrite');
        tx.objectStore(STORE_DIR_META).delete(dirId);
        tx.objectStore(STORE_DOCS).delete(docsRange(dirId));
        await transactionDone(tx);
      } catch (e) {
        console.error('[EmbeddingSearch] Failed to clear directory index:', e);
      }
    }

    this.indexedDocuments.clear();
    this.dirtyPaths.clear();
    this.deletedPaths.clear();
    this.lastRun = null;
    this.onStatusChange('Index cleared for current directory.');
    this.onUpdate();
  }
}

function batchByCategory(items: DiscoveredFile[], size: number):
    DiscoveredFile[][] {
  const batches: DiscoveredFile[][] = [];
  let current: DiscoveredFile[] = [];
  for (const item of items) {
    if (current.length >= size ||
        (current.length > 0 && current[0]!.category !== item.category)) {
      batches.push(current);
      current = [];
    }
    current.push(item);
  }
  if (current.length > 0) batches.push(current);
  return batches;
}
