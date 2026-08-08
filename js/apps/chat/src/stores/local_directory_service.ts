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

import {CustomModel, SettingsStore} from './settings_store.js';

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

// IndexedDB helpers to store Directory Handle
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('litertlm-local-dir', 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('handles')) {
        db.createObjectStore('handles');
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function storeDirectoryHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  const db = await openDB();
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction('handles', 'readwrite');
    const store = transaction.objectStore('handles');
    const request = store.put(handle, 'local-dir');
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function getDirectoryHandle(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('handles', 'readonly');
      const store = transaction.objectStore('handles');
      const request = store.get('local-dir');
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  } catch (e) {
    console.warn('[LiteRT-LM] IndexedDB not available or failed:', e);
    return null;
  }
}

async function* traverseDirectory(
    dirHandle: FileSystemDirectoryHandle,
    path = ''
): AsyncIterable<{ name: string; path: string; handle: FileSystemFileHandle }> {
  for await (const entry of dirHandle.values()) {
    const relativePath = path ? `${path}/${entry.name}` : entry.name;
    if (entry.kind === 'file') {
      if (entry.name.endsWith('.litertlm')) {
        yield {
          name: entry.name,
          path: relativePath,
          handle: entry as FileSystemFileHandle
        };
      }
    } else if (entry.kind === 'directory') {
      yield* traverseDirectory(entry as FileSystemDirectoryHandle, relativePath);
    }
  }
}

/**
 * Service to manage local directory access via File System Access API.
 */
export class LocalDirectoryService {
  private dirHandle: FileSystemDirectoryHandle | null = null;
  isAuthorized = false;

  get isSupported(): boolean {
    return 'showDirectoryPicker' in window;
  }

  constructor(
      private readonly settings: SettingsStore,
      private readonly updateCallback: () => void,
      private readonly updateStatus: (msg: string) => void
  ) {
    void this.tryLoadExistingHandle();
  }

  private async tryLoadExistingHandle() {
    try {
      const handle = await getDirectoryHandle();
      if (handle) {
        this.dirHandle = handle;
        const permission = await handle.queryPermission({mode: 'read'});
        this.isAuthorized = permission === 'granted';
        if (this.isAuthorized) {
          await this.scanDirectory();
        }
        this.updateCallback();
      }
    } catch (e) {
      console.error('[LiteRT-LM] Failed to load existing directory handle:', e);
    }
  }

  async mountDirectory() {
    try {
      if (this.dirHandle && !this.isAuthorized) {
        const permission = await this.dirHandle.requestPermission({mode: 'read'});
        this.isAuthorized = permission === 'granted';
        if (this.isAuthorized) {
          await this.scanDirectory();
        }
        this.updateCallback();
        return;
      }

      if (this.dirHandle && this.isAuthorized) {
        const change = confirm('Do you want to select a different local directory? (Cancel will just rescan the current one)');
        if (!change) {
          await this.scanDirectory();
          return;
        }
      }

      const showDirectoryPicker = window.showDirectoryPicker;
      if (!showDirectoryPicker) {
        throw new Error('File System Access API is not supported in this browser.');
      }

      const handle = await showDirectoryPicker();
      this.dirHandle = handle;
      await storeDirectoryHandle(handle);
      this.isAuthorized = true;
      await this.scanDirectory();
      this.updateCallback();
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        this.updateStatus('Directory selection cancelled.');
      } else {
        console.error('[LiteRT-LM] Failed to mount directory:', e);
        this.updateStatus(`Failed to mount directory: ${(e as Error).message}`);
      }
    }
  }

  async scanDirectory() {
    if (!this.dirHandle || !this.isAuthorized) return;

    this.updateStatus('Scanning local directory...');
    const models: CustomModel[] = [];
    try {
      for await (const entry of traverseDirectory(this.dirHandle)) {
        const file = await entry.handle.getFile();
        models.push({
          name: entry.name.replace(/\.litertlm$/, ''),
          filename: entry.name,
          path: `local-dir://${entry.path}`,
          size: `${(file.size / 1e9).toFixed(2)} GB`
        });
      }
      this.settings.localDirModels = models;
      this.settings.saveSettings();
      this.updateStatus(`Scanned ${models.length} local models.`);
    } catch (e) {
      console.error('[LiteRT-LM] Failed to scan directory:', e);
      this.updateStatus(`Failed to scan directory: ${(e as Error).message}`);
    }
  }

  async getFile(path: string): Promise<File> {
    if (!this.dirHandle) {
      throw new Error('No local directory mounted.');
    }
    if (!this.isAuthorized) {
      const permission = await this.dirHandle.requestPermission({mode: 'read'});
      this.isAuthorized = permission === 'granted';
      this.updateCallback();
      if (!this.isAuthorized) {
        throw new Error('Permission denied to access local directory.');
      }
    }

    const relativePath = path.replace(/^local-dir:\/\//, '');
    const parts = relativePath.split('/');
    let currentDir = this.dirHandle;

    for (let i = 0; i < parts.length - 1; i++) {
      currentDir = await currentDir.getDirectoryHandle(parts[i]!);
    }

    const fileHandle = await currentDir.getFileHandle(parts[parts.length - 1]!);
    return await fileHandle.getFile();
  }
}
