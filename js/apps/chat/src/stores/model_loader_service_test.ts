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

import {Engine, getOrLoadGlobalLiteRtLm, LiteRtLm} from '@litert-lm/core';
import {EngineFake} from '@litert-lm/core/testing';

import {ModelLoaderService} from './model_loader_service.js';
import {SettingsStore} from './settings_store.js';

describe('ModelLoaderService', () => {
  let settingsStore: SettingsStore;
  let modelLoader: ModelLoaderService;

  let fakeEngineCreate: typeof Engine.create;
  let fakeLoadWasm: typeof getOrLoadGlobalLiteRtLm;

  beforeEach(() => {
    if (!(window as unknown as Record<string, unknown>)["caches"]) {
      (window as unknown as Record<string, unknown>)["caches"] = {
        open: async () => {},
        delete: async () => {},
      };
    }

    settingsStore = new SettingsStore(() => {});
    fakeEngineCreate = async (settings, hint) => {
      return await EngineFake.create(settings, hint);
    };
    fakeLoadWasm = async () => ({} as LiteRtLm);
    modelLoader = new ModelLoaderService(
        () => {}, (msg: string) => {}, undefined, fakeEngineCreate,
        fakeLoadWasm);
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('initializes neatly', () => {
    expect(modelLoader.isModelLoading).toBeFalse();
    expect(modelLoader.cachedModels.size).toBe(0);
  });

  it('aborts active downloads neatly', () => {
    const mockAbort = jasmine.createSpy('abort');
    const MOCK_ABORT_CONTROLLER = {abort: mockAbort} as unknown as
        AbortController;

    modelLoader.downloadAbortController = MOCK_ABORT_CONTROLLER;
    modelLoader.cancelDownload();

    expect(modelLoader.isDownloadAborted).toBeTrue();
    expect(mockAbort).toHaveBeenCalled();
  });

  it('updates cache size correctly', async () => {
    let updateCalled = false;
    modelLoader = new ModelLoaderService(() => {
      updateCalled = true;
    }, (msg: string) => {}, undefined, fakeEngineCreate, fakeLoadWasm);

    const mockCache = {
      keys: jasmine.createSpy().and.resolveTo([
        {url: 'https://example.com/model1.litertlm'},
        {url: 'https://example.com/model2.litertlm'}
      ]),
      match: jasmine.createSpy().and.callFake(async (req: {url: string}) => {
        if (req.url.includes('model1')) {
          return {headers: new Headers({'content-length': '1024'})};
        } else {
          return {
            headers: new Headers(),
            blob: async () => ({size: 2048}),
          };
        }
      })
    };
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

    await modelLoader.updateCacheSize();

    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(modelLoader.cachedModels.get('model1.litertlm')).toBe(1024);
    expect(modelLoader.cachedModels.get('model2.litertlm')).toBe(2048);
    expect(updateCalled).toBeTrue();
  });

  it('deletes model from cache when confirmed', async () => {
    spyOn(window, 'confirm').and.returnValue(true);
    const mockCache = {delete: jasmine.createSpy().and.resolveTo(true)};
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);
    spyOn(modelLoader, 'updateCacheSize').and.resolveTo();

    const deleted = await modelLoader.deleteModelFromCache('model1.litertlm');

    expect(deleted).toBeTrue();
    expect(window.confirm).toHaveBeenCalled();
    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(mockCache.delete).toHaveBeenCalledWith('model1.litertlm');
    expect(modelLoader.updateCacheSize).toHaveBeenCalled();
  });

  it('does not delete model if user cancels confirmation', async () => {
    spyOn(window, 'confirm').and.returnValue(false);
    spyOn(window.caches, 'open');

    const deleted = await modelLoader.deleteModelFromCache('model1.litertlm');

    expect(deleted).toBeFalse();
    expect(window.caches.open).not.toHaveBeenCalled();
  });

  it('clears all cache when confirmed', async () => {
    spyOn(window, 'confirm').and.returnValue(true);
    spyOn(window.caches, 'delete').and.resolveTo(true);
    spyOn(modelLoader, 'updateCacheSize').and.resolveTo();

    const onDeleted = jasmine.createSpy('onDeleted');
    await modelLoader.clearAllCache(onDeleted);

    expect(window.confirm).toHaveBeenCalled();
    expect(window.caches.delete).toHaveBeenCalledWith('litertlm-models');
    expect(onDeleted).toHaveBeenCalled();
    expect(modelLoader.updateCacheSize).toHaveBeenCalled();
  });

  it('does not clear all cache if user cancels confirmation', async () => {
    spyOn(window, 'confirm').and.returnValue(false);
    spyOn(window.caches, 'delete');

    const onDeleted = jasmine.createSpy('onDeleted');
    await modelLoader.clearAllCache(onDeleted);

    expect(window.caches.delete).not.toHaveBeenCalled();
    expect(onDeleted).not.toHaveBeenCalled();
  });

  it('loadModelWeights ignores overlapping calls', async () => {
    modelLoader.isModelLoading = true;
    spyOn(window.caches, 'open');
    await modelLoader.loadModelWeights(settingsStore.modelSettings, async () => {});
    expect(window.caches.open).not.toHaveBeenCalled();
  });

  it('loadModelWeights loads from cache if available', async () => {
    settingsStore.selectedModelPath = 'https://example.com/model.litertlm';

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3]));
        controller.close();
      }
    });

    const mockCache = {
      match: jasmine.createSpy().and.resolveTo({
        headers: new Headers({'content-length': '3'}),
        body: mockStream,
      })
    };
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);
    spyOn(window, 'fetch');

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(mockCache.match)
        .toHaveBeenCalledWith('https://example.com/model.litertlm');
    expect(window.fetch).not.toHaveBeenCalled();
    expect(modelLoader.engine).toBeDefined();
    expect(onModelLoaded).toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
  });

  it('loadModelWeights downloads model if not in cache', async () => {
    settingsStore.selectedModelPath = 'https://example.com/model.litertlm';

    const mockCache = {
      match: jasmine.createSpy().and.resolveTo(undefined),
      put: jasmine.createSpy().and.resolveTo()
    };
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3]));
        controller.close();
      }
    });

    spyOn(window, 'fetch').and.resolveTo({
      ok: true,
      headers: new Headers({'content-length': '3'}),
      body: mockStream,
    } as unknown as Response);

    spyOn(modelLoader, 'updateCacheSize').and.resolveTo();

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(mockCache.match)
        .toHaveBeenCalledWith('https://example.com/model.litertlm');
    expect(window.fetch).toHaveBeenCalled();
    expect(mockCache.put).toHaveBeenCalled();
    expect(modelLoader.engine).toBeDefined();
    expect(onModelLoaded).toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
  });

  it('loadModelWeights handles fetch failure', async () => {
    settingsStore.selectedModelPath = 'https://example.com/model.litertlm';

    const mockCache = {match: jasmine.createSpy().and.resolveTo(undefined)};
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

    spyOn(window, 'fetch')
        .and.resolveTo({ok: false, statusText: 'Not Found'} as unknown as Response);

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(window.fetch).toHaveBeenCalled();
    expect(modelLoader.engine).toBeNull();
    expect(onModelLoaded).not.toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
  });

  it('loadModelWeights handles user cancellation', async () => {
    settingsStore.selectedModelPath = 'https://example.com/model.litertlm';

    const mockCache = {match: jasmine.createSpy().and.resolveTo(undefined)};
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

    spyOn(window, 'fetch').and.rejectWith({name: 'AbortError'});

    let latestStatus = '';
    modelLoader =
        new ModelLoaderService(() => {}, (msg: string) => {
          latestStatus = msg;
        }, undefined, fakeEngineCreate, fakeLoadWasm);

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(window.fetch).toHaveBeenCalled();
    expect(modelLoader.engine).toBeNull();
    expect(onModelLoaded).not.toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
    expect(latestStatus).toBe('Model download cancelled by user.');
  });

  it('imports custom model correctly', async () => {
    const mockCache = {
      put: jasmine.createSpy().and.resolveTo(true)
    };
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);
    spyOn(modelLoader, 'updateCacheSize').and.resolveTo();

    const file = new File(['model data'], 'custom-model.litertlm', {type: 'application/octet-stream'});
    const customModel = await modelLoader.importCustomModel(file);

    expect(customModel.path).toBe('https://local-model/custom-model.litertlm');
    expect(customModel.name).toBe('custom-model');
    expect(customModel.filename).toBe('custom-model.litertlm');
    expect(customModel.size).toBe('0.00 GB');
    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(mockCache.put).toHaveBeenCalled();
    expect(modelLoader.updateCacheSize).toHaveBeenCalled();
  });

  it('loadModelWeights fails if local model is not in cache', async () => {
    settingsStore.selectedModelPath = 'https://local-model/missing-model.litertlm';

    const mockCache = {
      match: jasmine.createSpy().and.resolveTo(undefined)
    };
    spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);
    spyOn(window, 'fetch');

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(window.caches.open).toHaveBeenCalledWith('litertlm-models');
    expect(mockCache.match).toHaveBeenCalledWith('https://local-model/missing-model.litertlm');
    expect(window.fetch).not.toHaveBeenCalled();
    expect(modelLoader.engine).toBeNull();
    expect(onModelLoaded).not.toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
  });



  it('loadModelWeights loads from local directory', async () => {
    settingsStore.selectedModelPath = 'local-dir://gemma.litertlm';

    const mockFile = new File(['model data'], 'gemma.litertlm', {type: 'application/octet-stream'});
    const mockLocalDirService = jasmine.createSpyObj('LocalDirectoryService', ['getFile']);
    mockLocalDirService.getFile.and.resolveTo(mockFile);

    modelLoader = new ModelLoaderService(
        () => {}, (msg: string) => {}, mockLocalDirService,
        fakeEngineCreate, fakeLoadWasm);

    const onModelLoaded = jasmine.createSpy('onModelLoaded').and.resolveTo();

    await modelLoader.loadModelWeights(settingsStore.modelSettings, onModelLoaded);

    expect(mockLocalDirService.getFile).toHaveBeenCalledWith('local-dir://gemma.litertlm');
    expect(modelLoader.engine).toBeDefined();
    expect(onModelLoaded).toHaveBeenCalled();
    expect(modelLoader.isModelLoading).toBeFalse();
  });

  describe('needsModelReload', () => {
    it('returns true when engine is null', () => {
      modelLoader.engine = null;
      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeTrue();
    });

    it('returns false after successful load', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          controller.close();
        }
      });
      const mockCache = {
        match: jasmine.createSpy().and.resolveTo({
          headers: new Headers({'content-length': '3'}),
          body: mockStream,
        })
      };
      spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

      settingsStore.selectedModelPath = 'https://example.com/model.litertlm';
      settingsStore.contextLength = 2048;
      settingsStore.topK = 32;

      await modelLoader.loadModelWeights(settingsStore.modelSettings, async () => {});

      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeFalse();
    });

    it('returns true when selectedModelPath changes', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          controller.close();
        }
      });
      const mockCache = {
        match: jasmine.createSpy().and.resolveTo({
          headers: new Headers({'content-length': '3'}),
          body: mockStream,
        })
      };
      spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

      settingsStore.selectedModelPath = 'https://example.com/model.litertlm';
      await modelLoader.loadModelWeights(settingsStore.modelSettings, async () => {});

      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeFalse();

      settingsStore.selectedModelPath = 'https://example.com/different.litertlm';
      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeTrue();
    });

    it('returns true when contextLength changes', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          controller.close();
        }
      });
      const mockCache = {
        match: jasmine.createSpy().and.resolveTo({
          headers: new Headers({'content-length': '3'}),
          body: mockStream,
        })
      };
      spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

      settingsStore.selectedModelPath = 'https://example.com/model.litertlm';
      settingsStore.contextLength = 2048;
      await modelLoader.loadModelWeights(settingsStore.modelSettings, async () => {});

      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeFalse();

      settingsStore.contextLength = 4096;
      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeTrue();
    });

    it('returns true when topK changes', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          controller.close();
        }
      });
      const mockCache = {
        match: jasmine.createSpy().and.resolveTo({
          headers: new Headers({'content-length': '3'}),
          body: mockStream,
        })
      };
      spyOn(window.caches, 'open').and.resolveTo(mockCache as unknown as Cache);

      settingsStore.selectedModelPath = 'https://example.com/model.litertlm';
      settingsStore.topK = 32;
      await modelLoader.loadModelWeights(settingsStore.modelSettings, async () => {});

      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeFalse();

      settingsStore.topK = 64;
      expect(modelLoader.needsModelReload(settingsStore.modelSettings)).toBeTrue();
    });
  });
});
