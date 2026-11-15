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

import {ReactiveController, ReactiveControllerHost} from 'lit';

import {ChatSessionStore, ConversationMeta, getIdleStatus, StoredMessage} from './stores/chat_session_store.js';
import {LocalDirectoryService} from './stores/local_directory_service.js';
import {ConfirmDialogOptions, ModelLoaderService} from './stores/model_loader_service.js';
import {loadModelFromUrl} from './stores/model_url_parameter.js';
import {SettingsStore} from './stores/settings_store.js';

export type {ConfirmDialogOptions, ConversationMeta, StoredMessage};

function isErrorStatusMessage(msg: string): boolean {
  return /^(Failed|Could not|Error|Generation failed|⚠|Only images and audio|The "model" URL parameter)/i
      .test(msg.trim());
}

/**
 * Overrides installed by dev-only entry points (`src/dev/`) before the app is
 * created. Production code never sets them.
 */
export interface DevHooks {
  /** Replaces `localStorage`, e.g. to keep demo state in memory. */
  storage?: Storage;
  /** Creates the model loader, e.g. one with a simulated engine. */
  createModelLoader?:
      (updateCallback: () => void, updateStatus: (msg: string) => void,
       localDirService: LocalDirectoryService) => ModelLoaderService;
  /** Runs when the app connects. Returning true skips the normal startup. */
  onConnected?: (controller: LlmChatStateController) => boolean;
}

/** See `DevHooks`. */
export const devHooks: DevHooks = {};

/**
 * UI specific orchestrator state controller for the LiteRT-LM Chat
 * application.
 */
export class LlmChatStateController implements ReactiveController {
  private hosts: ReactiveControllerHost[] = [];

  readonly storage: Storage = devHooks.storage ?? window.localStorage;
  readonly settings: SettingsStore;
  readonly localDirService: LocalDirectoryService;
  readonly modelLoader: ModelLoaderService;
  readonly chatSession: ChatSessionStore;

  // Global UI status string
  statusText = getIdleStatus(false);

  // Dismissible inline error banner message when an operation fails
  errorMessage: string|null = null;

  // Optional non-blocking UI confirmation handler registered by the app shell
  confirmHandler: ((options: ConfirmDialogOptions) => Promise<boolean>)|null =
      null;

  constructor(host: ReactiveControllerHost) {
    this.settings = new SettingsStore(() => this.requestUpdate(), this.storage);
    this.localDirService = new LocalDirectoryService(
        this.settings,
        () => this.requestUpdate(),
        this.handleStatus,
    );
    const createModelLoader: NonNullable<DevHooks['createModelLoader']> =
        devHooks.createModelLoader ??
        ((...args) => new ModelLoaderService(...args));
    this.modelLoader = createModelLoader(
        () => this.requestUpdate(), this.handleStatus, this.localDirService);
    this.modelLoader.confirmAction = (options: ConfirmDialogOptions) =>
        this.requestConfirm(options);
    this.chatSession = new ChatSessionStore(
        () => this.requestUpdate(),
        this.settings,
        this.modelLoader,
        this.handleStatus,
        this.storage,
    );

    this.addHost(host, true);
  }

  private readonly handleStatus = (msg: string) => {
    this.statusText = msg;
    if (isErrorStatusMessage(msg)) {
      this.errorMessage = msg;
    } else if (
        msg.startsWith('Preparing') || msg.startsWith('Loading') ||
        msg.startsWith('Downloading') || msg.startsWith('Model loaded') ||
        msg.startsWith('Thinking') || msg.startsWith('Idle') ||
        msg.startsWith('Restored conversation')) {
      this.errorMessage = null;
    }
    this.requestUpdate();
  };

  requestConfirm(options: ConfirmDialogOptions): Promise<boolean> {
    if (this.confirmHandler) {
      return this.confirmHandler(options);
    }
    return Promise.resolve(confirm(options.message));
  }

  dismissError() {
    this.errorMessage = null;
    if (isErrorStatusMessage(this.statusText)) {
      this.statusText = getIdleStatus(Boolean(this.modelLoader.engine));
    }
    this.requestUpdate();
  }

  addHost(host: ReactiveControllerHost, isPrimary = false) {
    if (!this.hosts.includes(host)) {
      this.hosts.push(host);
      if (isPrimary) {
        host.addController(this);
      }
    }
  }

  removeHost(host: ReactiveControllerHost) {
    const idx = this.hosts.indexOf(host);
    if (idx !== -1) {
      this.hosts.splice(idx, 1);
    }
  }

  requestUpdate() {
    for (const host of this.hosts) {
      try {
        host.requestUpdate();
      } catch (e) {}
    }
  }

  hostConnected() {
    // Triggered automatically when the Lit Element mounts.
    const handledByDevHook = devHooks.onConnected?.(this) ?? false;
    void this.modelLoader.updateCacheSize();
    if (handledByDevHook) return;

    const loadsModelFromUrl = loadModelFromUrl(window.location.search, {
      settings: this.settings,
      modelLoader: this.modelLoader,
      chatSession: this.chatSession,
      updateStatus: this.handleStatus,
    });
    if (loadsModelFromUrl) return;

    const activeId = this.storage.getItem('litertlm-active-conv-id');
    if (activeId &&
        this.chatSession.conversationsList.some(c => c.id === activeId)) {
      void this.chatSession.selectConversation(activeId);
    } else {
      void this.chatSession.startNewConversation();
    }
  }

  hostDisconnected() {
    if (this.chatSession.activeConversation) {
      this.chatSession.activeConversation.delete();
      this.chatSession.activeConversation = null;
    }
    if (this.modelLoader.engine) {
      try {
        this.modelLoader.engine.delete();
      } catch (e) {
        console.error('[LiteRT-LM] Failed to delete engine:', e);
      }
      this.modelLoader.engine = null;
    }
  }
}
