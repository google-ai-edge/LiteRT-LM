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

import './chat_bubble';

import {ChatSessionStore} from '../stores/chat_session_store.js';
import {LlmChatStateController} from '../state_controller.js';
import {LitertChatBubble} from './chat_bubble.js';


describe('litert-chat-bubble', () => {
  let element: LitertChatBubble;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockClipboardWriteText: jasmine.Spy;

  beforeEach(async () => {
    // Mock clipboard
    if (!navigator.clipboard) {
      Object.assign(navigator, {
        clipboard: {
          writeText: () => Promise.resolve(),
        },
      });
    }
    mockClipboardWriteText = spyOn(navigator.clipboard, 'writeText')
                                 .and.returnValue(Promise.resolve());

    // Mock LlmChatStateController and its nested chatSession
    mockChatSession = jasmine.createSpyObj('ChatSessionStore', [
      'rewindAndEdit',
      'redoResponse',
    ]);
    mockState = {
      chatSession: mockChatSession,
    } as unknown as jasmine.SpyObj<LlmChatStateController>;

    element = document.createElement('litert-chat-bubble');
    element.state = mockState;
    element.message = {role: 'user', text: '', senderName: ''};
    document.body.appendChild(element);
  });

  afterEach(() => {
    if (element) {
      element.remove();
    }
  });

  it('renders a user message', async () => {
    element.message = {
      role: 'user',
      text: 'Hello, this is a user message.',
      senderName: 'User',
    };
    element.index = 0;
    await element.updateComplete;

    const bubble = element.shadowRoot!.querySelector('.message-bubble');
    expect(bubble).toBeTruthy();
    expect(bubble!.classList.contains('user')).toBeTrue();

    const sender = element.shadowRoot!.querySelector('.message-sender');
    expect(sender!.textContent!.trim()).toBe('User');

    const content = element.shadowRoot!.querySelector('.message-user-text');
    expect(content!.textContent!.trim()).toBe('Hello, this is a user message.');
  });

  it('renders an assistant message with markdown', async () => {
    element.message = {
      role: 'assistant',
      text: 'Hello *world*, this is **bold** and a [link](http://example.com).',
      senderName: 'Assistant (1.2 GB)',
    };
    element.index = 1;
    await element.updateComplete;

    const bubble = element.shadowRoot!.querySelector('.message-bubble');
    expect(bubble).toBeTruthy();
    expect(bubble!.classList.contains('assistant')).toBeTrue();

    // Check sender name is cleaned
    const sender = element.shadowRoot!.querySelector('.message-sender');
    expect(sender!.textContent!.trim()).toBe('Assistant');

    const content = element.shadowRoot!.querySelector('.message-content');
    expect(content).toBeTruthy();

    // Check markdown rendering
    const em = content!.querySelector('em');
    expect(em!.textContent).toBe('world');

    const strong = content!.querySelector('strong');
    expect(strong!.textContent).toBe('bold');

    const link = content!.querySelector('a') as HTMLAnchorElement;
    expect(link!.href).toBe('http://example.com/');
    expect(link!.textContent).toBe('link');
  });

  it('renders thought process CoT blocks', async () => {
    element.message = {
      role: 'assistant',
      text: 'Final answer.',
      thoughtText: 'Thinking about the answer...',
      senderName: 'Assistant',
    };
    element.index = 2;
    await element.updateComplete;

    const details = element.shadowRoot!.querySelector('.thought-details');
    expect(details).toBeTruthy();

    const summary = details!.querySelector('.thought-summary');
    expect(summary!.textContent!.trim()).toBe('Thought Process');

    const thoughtContent = details!.querySelector('.thought-content');
    expect(thoughtContent!.textContent!.trim()).toBe('Thinking about the answer...');
  });

  it('renders code blocks with headers and copy buttons', async () => {
    element.message = {
      role: 'assistant',
      text: 'Here is some code:\n```typescript\nconst x = 5;\n```',
      senderName: 'Assistant',
    };
    element.index = 3;
    await element.updateComplete;

    const block = element.shadowRoot!.querySelector('litert-code-block');
    expect(block).toBeTruthy();

    const container = block!.shadowRoot!.querySelector('.code-container');
    expect(container).toBeTruthy();

    const header = container!.querySelector('.code-header');
    expect(header).toBeTruthy();

    const lang = header!.querySelector('.code-lang');
    expect(lang!.textContent!.trim()).toBe('typescript');

    const copyBtn = header!.querySelector('.btn-copy-code');
    expect(copyBtn).toBeTruthy();
    expect(copyBtn!.textContent!.trim()).toBe('Copy');

    const code = block!.querySelector('code');
    expect(code!.textContent!.trim()).toBe('const x = 5;');
  });

  it('triggers rewindAndEdit when user clicks Edit', async () => {
    element.message = {
      role: 'user',
      text: 'Edit me',
      senderName: 'User',
    };
    element.index = 4;
    await element.updateComplete;

    const editBtn = element.shadowRoot!.querySelector('.btn-action') as HTMLButtonElement;
    expect(editBtn.textContent!.trim()).toBe('✎ Edit');

    // Setup spy return
    mockChatSession.rewindAndEdit.and.returnValue(Promise.resolve('Edit me'));

    let eventDetail: {prompt: string} | null = null;
    element.addEventListener('edit-prompt', (e: Event) => {
      eventDetail = (e as CustomEvent<{prompt: string}>).detail;
    });

    editBtn.click();
    
    // Wait for async handler
    await new Promise(resolve => setTimeout(resolve, 50));

    expect(mockState.chatSession.rewindAndEdit).toHaveBeenCalledWith(4);
    expect(eventDetail as {prompt: string} | null).toEqual({prompt: 'Edit me'});
  });

  it('triggers redoResponse when user clicks Retry', async () => {
    element.message = {
      role: 'assistant',
      text: 'Retry me',
      senderName: 'Assistant',
    };
    element.index = 5;
    await element.updateComplete;

    // The copy button is first, retry is second
    const actionBtns = element.shadowRoot!.querySelectorAll('.btn-action');
    const retryBtn = actionBtns[1] as HTMLButtonElement;
    expect(retryBtn.textContent!.trim()).toBe('⟲ Retry');

    retryBtn.click();

    expect(mockState.chatSession.redoResponse).toHaveBeenCalledWith(5);
  });

  it('copies code to clipboard when copy button is clicked in code header', async () => {
    element.message = {
      role: 'assistant',
      text: 'Here is some code:\n```typescript\nconst x = 5;\n```',
      senderName: 'Assistant',
    };
    element.index = 6;
    await element.updateComplete;

    const block = element.shadowRoot!.querySelector('litert-code-block');
    const copyBtn =
        block!.shadowRoot!.querySelector('.btn-copy-code') as HTMLButtonElement;
    expect(copyBtn).toBeTruthy();

    copyBtn.click();
    
    // Wait for async copy handler
    await new Promise(resolve => setTimeout(resolve, 50));

    expect(mockClipboardWriteText).toHaveBeenCalledWith('const x = 5;');
    expect(copyBtn.textContent!.trim()).toBe('Copied!');
  });

  it('dispatches preview-html event when preview button is clicked in HTML code header', async () => {
    element.message = {
      role: 'assistant',
      text: 'Here is HTML:\n```html\n<h1>Hello</h1>\n```',
      senderName: 'Assistant',
    };
    element.index = 7;
    await element.updateComplete;

    const block = element.shadowRoot!.querySelector('litert-code-block');
    const previewBtn = block!.shadowRoot!.querySelector('.btn-preview-code') as
        HTMLButtonElement;
    expect(previewBtn).toBeTruthy();

    let eventDetail: { base64Code: string } | null = null;
    element.addEventListener('preview-html', (e: Event) => {
      eventDetail = (e as CustomEvent<{base64Code: string}>).detail;
    });

    previewBtn.click();

    expect(eventDetail as {base64Code: string} | null).toEqual({
      base64Code: jasmine.any(String),
    });
    
    expect(decodeURIComponent(escape(atob(eventDetail!.base64Code))).trim()).toBe('<h1>Hello</h1>');
  });

  it('renders inline and display LaTeX math', async () => {
    element.message = {
      role: 'assistant',
      text: 'Inline math $e^{i\\pi} + 1 = 0$ and display math:\n$$f(x) = \\int_{-\\infty}^{\\infty} e^{-x^2} dx$$',
      senderName: 'Assistant',
    };
    element.index = 8;
    await element.updateComplete;

    const content = element.shadowRoot!.querySelector('.message-content');
    expect(content).toBeTruthy();

    // Check KaTeX element exists
    const katexElements = content!.querySelectorAll('.katex');
    expect(katexElements.length).toBe(2);

    // One of them should be in display mode (wrapped in .katex-display)
    const displayKatex = content!.querySelector('.katex-display');
    expect(displayKatex).toBeTruthy();
    expect(displayKatex!.querySelector('.katex')).toBeTruthy();
  });

  it('does not render LaTeX math inside code blocks', async () => {
    element.message = {
      role: 'assistant',
      text: 'Here is code with dollar: `const x = $5;` and a block:\n```\n$y = 6$\n```',
      senderName: 'Assistant',
    };
    element.index = 9;
    await element.updateComplete;

    const content = element.shadowRoot!.querySelector('.message-content');
    expect(content).toBeTruthy();

    // There should be no KaTeX elements because they are in code blocks
    const katexElements = content!.querySelectorAll('.katex');
    expect(katexElements.length).toBe(0);
  });
});
