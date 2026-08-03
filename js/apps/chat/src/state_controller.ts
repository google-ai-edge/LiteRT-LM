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

import {ChatSessionStore, ConversationMeta, StoredMessage} from './stores/chat_session_store.js';
import {ModelLoaderService} from './stores/model_loader_service.js';
import {SettingsStore} from './stores/settings_store.js';

export type {ConversationMeta, StoredMessage};

/**
 * UI specific orchestrator state controller for the LiteRT-LM Chat
 * application.
 */
export class LlmChatStateController implements ReactiveController {
  private hosts: ReactiveControllerHost[] = [];

  readonly settings: SettingsStore;
  readonly modelLoader: ModelLoaderService;
  readonly chatSession: ChatSessionStore;

  // Global UI status string
  statusText = 'Idle. Select settings to start.';

  constructor(host: ReactiveControllerHost) {
    this.settings = new SettingsStore(() => this.requestUpdate());
    this.modelLoader = new ModelLoaderService(
        () => this.requestUpdate(), this.settings, (msg: string) => {
          this.statusText = msg;
          this.requestUpdate();
        });
    this.chatSession = new ChatSessionStore(
        () => this.requestUpdate(), this.settings, this.modelLoader,
        (msg: string) => {
          this.statusText = msg;
          this.requestUpdate();
        });

    this.addHost(host, true);
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
    void this.modelLoader.updateCacheSize();

    const activeId = window.localStorage.getItem('litertlm-active-conv-id');
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
