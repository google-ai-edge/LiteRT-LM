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

import {devHooks, LlmChatStateController} from '../state_controller.js';

import {
  createDemoEngine,
  DemoModelLoader,
  installDemoMode,
} from './demo_mode';

function createConnectedController(): LlmChatStateController {
  const host = {
    addController() {},
    removeController() {},
    requestUpdate() {},
    updateComplete: Promise.resolve(true),
  };
  const controller = new LlmChatStateController(host);
  controller.hostConnected();
  return controller;
}

describe('demo_mode', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    delete devHooks.storage;
    delete devHooks.createModelLoader;
    delete devHooks.onConnected;
    window.localStorage.clear();
  });

  it('ignores URLs without the demo query parameter', () => {
    expect(installDemoMode('?foo=bar')).toBeFalse();
    expect(devHooks.onConnected).toBeUndefined();
  });

  it('shows a cancellable download with ?demo=loading', () => {
    expect(installDemoMode('?demo=loading')).toBeTrue();
    const controller = createConnectedController();
    expect(controller.modelLoader.isModelLoading).toBeTrue();
    expect(controller.modelLoader.downloadAbortController).not.toBeNull();

    controller.modelLoader.cancelDownload();
    expect(controller.modelLoader.isModelLoading).toBeFalse();
    expect(controller.statusText).toBe('Model download cancelled by user.');
  });

  it('leaves the ?demo=loading download when weights load', async () => {
    installDemoMode('?demo=loading');
    const controller = createConnectedController();

    await controller.modelLoader.loadModelWeights(
        controller.settings.modelSettings, async () => {});

    expect(controller.modelLoader.isModelLoading).toBeFalse();
    expect(controller.modelLoader.downloadProgresses.size).toBe(0);
    expect(controller.modelLoader.downloadAbortController).toBeNull();
  });

  it('shows the error banner with ?demo=error', () => {
    installDemoMode('?demo=error');
    const controller = createConnectedController();
    expect(controller.errorMessage).toContain('WebGPU adapter unavailable');
  });

  it('streams thought and reply chunks from createDemoEngine', async () => {
    const engine = createDemoEngine(0);
    const conv = await engine.createConversation({
      preface: {extra_context: {'enable_thinking': true}},
    });
    const stream = conv.sendMessageStreaming('Explain WebGPU');
    const reader = stream.getReader();
    let content = '';
    let thought = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (typeof value.content === 'string') content += value.content;
      if (typeof value.channels?.['thought'] === 'string') {
        thought += value.channels['thought'];
      }
    }
    expect(thought.length).toBeGreaterThan(0);
    expect(content).toContain('WebGPU');
  });

  it('keeps ?demo=1 conversations and settings out of localStorage',
     async () => {
       window.localStorage.setItem(
           'litertlm-conversations-list',
           JSON.stringify([{
             id: 'real-conv',
             title: 'Saved User Chat',
             createdAt: 100,
             modelPath: 'path/to/model.bin',
           }]));
       window.localStorage.setItem('litertlm-active-conv-id', 'real-conv');

       installDemoMode('?demo=1');
       const controller = createConnectedController();
       await (controller.modelLoader as DemoModelLoader).ready;
       expect(controller.chatSession.conversationsList.map(c => c.title))
           .toEqual(['On-device WebGPU & interactive preview']);
       expect(controller.chatSession.messages.length).toBe(4);

       controller.settings.systemPrompt = 'You are terse.';
       controller.settings.saveSettings();
       expect(window.localStorage.getItem('litertlm-chat-settings')).toBeNull();
       expect(window.localStorage.getItem('litertlm-conversations-list'))
           .toContain('Saved User Chat');
       expect(window.localStorage.getItem('litertlm-active-conv-id'))
           .toBe('real-conv');
     });
});
