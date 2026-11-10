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


import {devHooks, LlmChatStateController} from './state_controller.js';
import {ModelLoaderService} from './stores/model_loader_service.js';

describe('LlmChatStateController', () => {
  let emitLoaderStatus: (msg: string) => void;
  let controller: LlmChatStateController;

  beforeEach(() => {
    window.localStorage.clear();
    devHooks.createModelLoader = (onUpdate, updateStatus, localDirService) => {
      emitLoaderStatus = updateStatus;
      return new ModelLoaderService(onUpdate, updateStatus, localDirService);
    };
    controller = createController();
  });

  function createController(): LlmChatStateController {
    return new LlmChatStateController({
      addController() {},
      removeController() {},
      requestUpdate() {},
      updateComplete: Promise.resolve(true),
    });
  }

  afterEach(() => {
    delete devHooks.createModelLoader;
    window.localStorage.clear();
  });

  it('asks the confirm handler for model loader confirmations', async () => {
    const confirmHandler = jasmine.createSpy().and.resolveTo(false);
    controller.confirmHandler = confirmHandler;
    const options = {title: 'Load model?', message: 'It is large.'};

    await expectAsync(controller.modelLoader.confirmAction(options))
        .toBeResolvedTo(false);
    expect(confirmHandler).toHaveBeenCalledWith(options);
  });

  it('shows an error status until the next load starts', () => {
    emitLoaderStatus('Failed to load model: out of memory');
    expect(controller.statusText).toBe('Failed to load model: out of memory');
    expect(controller.errorMessage).toBe('Failed to load model: out of memory');

    emitLoaderStatus('Loading cached weights (model.litertlm)...');
    expect(controller.statusText)
        .toBe('Loading cached weights (model.litertlm)...');
    expect(controller.errorMessage).toBeNull();
  });

  it('clears the error when the user starts a new conversation', async () => {
    emitLoaderStatus('Failed to load model: out of memory');

    await controller.chatSession.startNewConversation();

    expect(controller.statusText).toBe('Idle · Weights load on first message');
    expect(controller.errorMessage).toBeNull();
  });

  it('reopens the saved active conversation when connected', () => {
    window.localStorage.setItem(
        'litertlm-conversations-list',
        JSON.stringify(
            [{id: 'saved', title: 'Saved', createdAt: 1, modelPath: 'm'}]));
    window.localStorage.setItem(
        'litertlm-chat-history-saved',
        JSON.stringify([{role: 'user', text: 'Hi', senderName: 'User'}]));
    window.localStorage.setItem('litertlm-active-conv-id', 'saved');
    controller = createController();

    controller.hostConnected();

    expect(controller.chatSession.activeSavedConvId).toBe('saved');
  });
});
