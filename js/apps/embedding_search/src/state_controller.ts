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

import {LitElement, ReactiveController, ReactiveControllerHost} from 'lit';
import {property} from 'lit/decorators.js';

import {ExampleQuery, PROMPT_TEMPLATE_LABELS, PromptTemplateId, SAMPLE_IMAGE_FILES} from './app_config.js';
import {DiscoveredFile, FileCategory, IndexStore} from './stores/index_store.js';
import {ModelsStore} from './stores/models_store.js';
import {SAMPLE_ALBUM_DIR_ID, SAMPLE_ALBUM_DIR_NAME, SampleAlbumStore, SampleAlbumStoreOptions} from './stores/sample_album_store.js';
import {QueryPart, SearchStore} from './stores/search_store.js';
import {AudioRecorder} from './utils/audio_recorder.js';

/* tslint:disable:no-new-decorators */

const PROMPT_TEMPLATE_KEY = 'embedding-search:prompt-template';

function loadPromptTemplate(): PromptTemplateId {
  const stored = window.localStorage.getItem(PROMPT_TEMPLATE_KEY);
  return stored && stored in PROMPT_TEMPLATE_LABELS ?
      stored as PromptTemplateId :
      'none';
}

/** The element that currently has focus, descending through open shadow roots. */
export function deepActiveElement(): HTMLElement|null {
  let active: Element|null = document.activeElement;
  while (active?.shadowRoot?.activeElement) {
    active = active.shadowRoot.activeElement;
  }
  return active instanceof HTMLElement ? active : null;
}

/** Returns focus to `element` if it is still attached to the document. */
export function restoreFocus(element: HTMLElement|null) {
  if (element?.isConnected) element.focus();
}

/** A folder offered in the folder dropdown. */
export interface DirectoryOption {
  id: string;
  name: string;
}

/** Constructor options for `EmbeddingSearchStateController`. */
export interface StateControllerOptions {
  autoStart?: boolean;
  indexDbName?: string;
  sampleAlbum?: SampleAlbumStoreOptions;
}

/**
 * Main reactive controller coordinating model loading, indexing,
 * searching, and audio recording across the application.
 */
export class EmbeddingSearchStateController implements ReactiveController {
  private hosts: ReactiveControllerHost[] = [];

  readonly models: ModelsStore;
  readonly indexer: IndexStore;
  readonly search: SearchStore;
  readonly samples: SampleAlbumStore;
  readonly recorder: AudioRecorder;

  statusMessage = 'Welcome to LiteRT-LM EmbeddingGemma Multimodal Search.';
  promptTemplate: PromptTemplateId = loadPromptTemplate();

  private indexingRun: Promise<void>|null = null;

  isModelModalOpen = false;
  isCameraModalOpen = false;

  constructor(host: ReactiveControllerHost, options: StateControllerOptions = {}) {
    const update = () => this.requestUpdate();
    const status = (msg: string) => this.setStatus(msg);

    this.models = new ModelsStore(update, status);
    this.indexer = new IndexStore(
        update,
        status,
        () => {
          this.search.clearQuery();
          this.requestUpdate();
        },
        {dbName: options.indexDbName},
    );
    this.search = new SearchStore(update, status);
    this.samples = new SampleAlbumStore(update, status, options.sampleAlbum);
    this.recorder = new AudioRecorder();

    this.addHost(host, true);
    if (options.autoStart ?? true) {
      void this.startUp();
    }
  }

  private async startUp() {
    const modelPromise = this.models.autoLoad().catch(() => false);
    await this.indexer.ready;
    if (!this.indexer.targetDirId ||
        (this.indexer.targetDirId === SAMPLE_ALBUM_DIR_ID &&
         (this.samples.hasBaseUrlChanged() ||
          this.indexer.discoveredFiles.length < SAMPLE_IMAGE_FILES.length))) {
      const dir = await this.samples.install();
      if (dir) await this.useSampleAlbum(dir);
    }

    await modelPromise;
    if (!this.models.engine) {
      this.isModelModalOpen = true;
      this.requestUpdate();
      return;
    }
    await this.indexIfPending();
  }

  private useSampleAlbum(dir: FileSystemDirectoryHandle): Promise<void> {
    return this.indexer.useDirectory(
        dir, {id: SAMPLE_ALBUM_DIR_ID, name: SAMPLE_ALBUM_DIR_NAME});
  }

  private async indexIfPending() {
    if (this.models.engine && this.indexer.isAuthorized &&
        this.indexer.countPending(this.promptTemplate) > 0) {
      await this.startIndexing();
    }
  }

  addHost(host: ReactiveControllerHost, isPrimary = false) {
    if (!this.hosts.includes(host)) {
      this.hosts.push(host);
      if (isPrimary) host.addController(this);
    }
  }

  removeHost(host: ReactiveControllerHost) {
    const idx = this.hosts.indexOf(host);
    if (idx !== -1) this.hosts.splice(idx, 1);
  }

  requestUpdate() {
    for (const host of this.hosts) {
      try {
        host.requestUpdate();
      } catch (_) {}
    }
  }

  setStatus(msg: string) {
    this.statusMessage = msg;
    this.requestUpdate();
  }

  hostDisconnected() {
    if (this.models.engine) {
      void this.models.unloadModel();
    }
  }

  private async stopIndexingRun() {
    if (this.indexingRun) {
      this.indexer.cancelIndexing();
      await this.indexingRun;
    }
  }

  async setPromptTemplate(template: PromptTemplateId) {
    if (template === this.promptTemplate) return;
    this.promptTemplate = template;
    window.localStorage.setItem(PROMPT_TEMPLATE_KEY, template);
    this.requestUpdate();

    if (this.indexingRun) {
      await this.stopIndexingRun();
      if (template !== this.promptTemplate) return;
    }

    const pending = this.indexer.countPending(template);
    if (pending > 0 && this.models.engine) {
      this.setStatus(
          `Prompt template changed. Embedding ${pending} file(s) for it…`);
      await this.startIndexing();
    } else {
      this.setStatus(pending > 0 ?
          `Prompt template changed. ${pending} file(s) need indexing.` :
          'Prompt template changed.');
    }
    if (template === this.promptTemplate && this.search.hasSearched &&
        this.models.engine) {
      await this.runSearch();
    }
  }

  private requireEngine() {
    if (this.models.engine) return this.models.engine;
    this.setStatus('Please load an EmbeddingGemma model first.');
    this.isModelModalOpen = true;
    this.requestUpdate();
    return null;
  }

  get needsReauthorization(): boolean {
    return this.indexer.needsReauthorization;
  }

  async reauthorize(): Promise<boolean> {
    if (!this.indexer.needsReauthorization) return true;
    return this.indexer.reauthorize();
  }

  async startIndexing() {
    const engine = this.requireEngine();
    if (!engine) return;
    if (this.indexingRun) return this.indexingRun;
    if (!(await this.reauthorize())) return;
    const run = this.indexer.startIndexing(engine, this.promptTemplate);
    this.indexingRun = run;
    try {
      await run;
    } finally {
      if (this.indexingRun === run) this.indexingRun = null;
    }
  }

  get availableDirectories(): DirectoryOption[] {
    const dirs: DirectoryOption[] =
        [{id: SAMPLE_ALBUM_DIR_ID, name: SAMPLE_ALBUM_DIR_NAME}];
    for (const dir of this.indexer.savedDirectories.values()) {
      if (dir.id !== SAMPLE_ALBUM_DIR_ID) {
        dirs.push({id: dir.id, name: dir.name});
      }
    }
    return dirs;
  }

  async loadSampleAlbum() {
    await this.stopIndexingRun();
    const dir = await this.samples.install();
    if (!dir) return;
    await this.useSampleAlbum(dir);
    await this.indexIfPending();
  }

  async selectDirectory(dirId: string) {
    const sampleAlbumIncomplete =
        dirId === SAMPLE_ALBUM_DIR_ID &&
        (this.samples.hasMissingFiles ||
         (this.indexer.targetDirId === SAMPLE_ALBUM_DIR_ID &&
          this.indexer.discoveredFiles.length < SAMPLE_IMAGE_FILES.length));
    if (dirId === this.indexer.targetDirId && this.indexer.isAuthorized &&
        !sampleAlbumIncomplete) {
      return;
    }
    await this.stopIndexingRun();
    if (dirId === SAMPLE_ALBUM_DIR_ID &&
        (!this.indexer.savedDirectories.has(SAMPLE_ALBUM_DIR_ID) ||
         sampleAlbumIncomplete)) {
      await this.loadSampleAlbum();
      return;
    }
    if (await this.indexer.selectSavedDirectory(dirId)) {
      await this.indexIfPending();
    }
  }

  async addNewDirectory() {
    await this.stopIndexingRun();
    if (await this.indexer.pickNewDirectory()) {
      await this.indexIfPending();
    }
  }

  async removeDirectory(dirId: string) {
    if (dirId === SAMPLE_ALBUM_DIR_ID) return;
    const wasActive = this.indexer.targetDirId === dirId;
    if (wasActive) await this.stopIndexingRun();
    await this.indexer.removeDirectory(dirId);
    if (wasActive && this.availableDirectories[0]) {
      await this.selectDirectory(this.availableDirectories[0].id);
    }
  }

  async runSearch() {
    const engine = this.requireEngine();
    if (!engine) return;
    if (!(await this.reauthorize())) return;
    await this.search.executeSearch(engine, this.indexer, this.promptTemplate);
  }

  setCategoryFilter(category: 'all'|FileCategory) {
    this.search.setSelectedCategory(category);
    if (this.search.hasSearched) {
      this.search.rerank(this.indexer, this.promptTemplate);
    }
  }

  get discoveredImages(): DiscoveredFile[] {
    return this.indexer.discoveredFiles.filter((f) => f.category === 'image');
  }

  resolveExampleImage(slot: {hint?: string; fallbackIndex: number}):
      DiscoveredFile|null {
    const images = this.discoveredImages;
    if (slot.hint) {
      const hint = slot.hint.toLowerCase();
      const match = images.find((d) => d.path.toLowerCase().includes(hint));
      if (match) return match;
    }
    return images[slot.fallbackIndex] ?? null;
  }

  canRunExample(example: ExampleQuery): boolean {
    return example.parts.every(
        (p) => !('image' in p) || this.resolveExampleImage(p.image) !== null);
  }

  async runExample(example: ExampleQuery) {
    if (!(await this.reauthorize())) return;
    const parts: QueryPart[] = [];
    for (const part of example.parts) {
      if ('text' in part) {
        parts.push({type: 'text', text: ` ${part.text} `});
        continue;
      }
      const doc = this.resolveExampleImage(part.image);
      const handle = doc && this.indexer.handleFor(doc.path);
      if (!doc || !handle) return;
      const file = await handle.getFile();
      parts.push({
        type: 'chip',
        chip: this.search.createChip('image', doc.name, file),
      });
    }
    this.search.setParts(parts);
    await this.runSearch();
  }
}

/**
 * Base element for components that observe EmbeddingSearchStateController.
 */
export class StatefulElement extends LitElement {
  private _state?: EmbeddingSearchStateController;

  @property({attribute: false, noAccessor: true})
  get state(): EmbeddingSearchStateController {
    if (!this._state) {
      throw new Error(
          `<${this.localName}> used before its "state" property was set.`);
    }
    return this._state;
  }
  set state(controller: EmbeddingSearchStateController) {
    const old = this._state;
    if (old !== controller) {
      old?.removeHost(this);
      this._state = controller;
      if (controller && this.isConnected) controller.addHost(this);
      this.requestUpdate('state', old);
    }
  }

  get hasState(): boolean {
    return this._state !== undefined;
  }

  override connectedCallback() {
    super.connectedCallback();
    this._state?.addHost(this);
  }

  override disconnectedCallback() {
    this._state?.removeHost(this);
    super.disconnectedCallback();
  }
}
