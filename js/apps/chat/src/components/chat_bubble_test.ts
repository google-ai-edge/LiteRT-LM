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

import {ChatSessionStore, PromptMedia, PromptPart} from '../stores/chat_session_store.js';
import {LlmChatStateController} from '../state_controller.js';
import {LitertChatBubble, trimAmbiguousTail} from './chat_bubble.js';


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
      'getMedia',
    ]);
    mockState = {
      chatSession: mockChatSession,
      requestConfirm: jasmine.createSpy('requestConfirm').and.resolveTo(true),
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

  it('registers with the state controller once and unregisters on removal',
     async () => {
       const state = {
         chatSession: mockChatSession,
         addHost: jasmine.createSpy('addHost'),
         removeHost: jasmine.createSpy('removeHost'),
       };
       const bubble = document.createElement('litert-chat-bubble');
       bubble.state = state as unknown as LlmChatStateController;
       bubble.message = {role: 'user', text: 'Hi', senderName: 'User'};
       document.body.appendChild(bubble);
       await bubble.updateComplete;

       bubble.remove();

       expect(state.addHost).toHaveBeenCalledOnceWith(bubble);
       expect(state.removeHost).toHaveBeenCalledOnceWith(bubble);
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

    // Check sender name is cleaned and no assistant logo avatar is rendered
    const sender = element.shadowRoot!.querySelector('.message-sender');
    expect(sender!.textContent!.trim()).toBe('Assistant');
    expect(element.shadowRoot!.querySelector('.assistant-avatar')).toBeNull();

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

  it('renders thought process CoT blocks collapsed once answer arrives and toggles on click', async () => {
    element.message = {
      role: 'assistant',
      text: 'Final answer.',
      thoughtText: 'Thinking about the answer...',
      senderName: 'Assistant',
    };
    element.index = 2;
    await element.updateComplete;

    const details = element.shadowRoot!.querySelector('.thought-details') as HTMLDetailsElement;
    expect(details).toBeTruthy();
    expect(details.open).toBeFalse();

    const summary = details.querySelector('.thought-summary') as HTMLElement;
    expect(summary.textContent!.trim()).toBe('Thought Process');
    const labelEl =
        summary.querySelector('.thought-summary-label') as HTMLElement;
    const beforeRect = labelEl.getBoundingClientRect();

    const thoughtContent = details.querySelector('.thought-content');
    expect(thoughtContent!.textContent!.trim()).toBe('Thinking about the answer...');

    summary.click();
    await element.updateComplete;
    expect(details.open).toBeTrue();
    const afterRect = labelEl.getBoundingClientRect();
    expect(afterRect.left).toBeCloseTo(beforeRect.left, 0);
    expect(afterRect.top).toBeCloseTo(beforeRect.top, 0);

    summary.click();
    await element.updateComplete;
    expect(details.open).toBeFalse();
  });

  it('hides assistant Copy and Retry actions while streaming', async () => {
    element.message = {
      role: 'assistant',
      text: 'Partial stream...',
      senderName: 'Assistant',
    };
    element.index = 2;
    element.streaming = true;
    await element.updateComplete;

    expect(element.shadowRoot!.querySelector('.message-actions')).toBeNull();

    element.streaming = false;
    await element.updateComplete;
    expect(element.shadowRoot!.querySelector('.message-actions')).toBeTruthy();
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

  it('keeps nested code blocks in their list and the text around them',
     async () => {
       element.message = {
         role: 'assistant',
         text: 'Intro:\n\n```js\nfirst();\n```\n\nThen:\n\n' +
             '- Step one:\n\n  ```js\n  second();\n  ```\n\n- Step two\n',
         senderName: 'Assistant',
       };
       await element.updateComplete;

       const blocks = element.shadowRoot!.querySelectorAll('litert-code-block');
       expect(blocks.length).toBe(2);
       expect(blocks[0]!.closest('li')).toBeNull();
       const text = blocks[0]!.parentElement!.textContent!;
       expect(text).toContain('Intro:');
       expect(text).toContain('Then:');
       expect(text).not.toContain('>');
       expect(blocks[1]!.closest('li')).not.toBeNull();
       expect(blocks[1]!.closest('ul')!.querySelectorAll('li').length).toBe(2);
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
    expect(editBtn.textContent!.trim()).toBe('Edit');
    expect(editBtn.querySelector('svg')).toBeTruthy();

    // Setup spy return
    const parts: PromptPart[] = [{type: 'text', text: 'Edit me'}];
    mockChatSession.rewindAndEdit.and.returnValue(Promise.resolve(parts));

    let eventDetail: {parts: PromptPart[]}|null = null;
    element.addEventListener('edit-prompt', (e: Event) => {
      eventDetail = (e as CustomEvent<{parts: PromptPart[]}>).detail;
    });

    editBtn.click();
    
    // Wait for async handler
    await new Promise(resolve => setTimeout(resolve, 50));

    expect(mockState.chatSession.rewindAndEdit).toHaveBeenCalledWith(4);
    expect(eventDetail as {parts: PromptPart[]} | null).toEqual({parts});
  });

  it('renders images and audio inline in a user message', async () => {
    const image: PromptMedia = {
      id: 'img1',
      kind: 'image',
      name: 'cat.png',
      blob: new Blob(),
      url: 'blob:fake-url',
    };
    const audio: PromptMedia = {
      id: 'aud1',
      kind: 'audio',
      name: 'meow.wav',
      blob: new Blob(),
      url: 'blob:fake-audio-url',
    };
    const media = new Map([[image.id, image], [audio.id, audio]]);
    mockChatSession.getMedia.and.callFake((id: string) => media.get(id));
    element.message = {
      role: 'user',
      text: 'Before [image: cat.png] after [image: dog.png] ' +
          '[audio: meow.wav] [audio: bark.wav]',
      parts: [
        {type: 'text', text: 'Before '},
        {type: 'image', id: 'img1', name: 'cat.png'},
        {type: 'text', text: ' after '},
        {type: 'image', id: 'img2', name: 'dog.png'},
        {type: 'text', text: ' '},
        {type: 'audio', id: 'aud1', name: 'meow.wav'},
        {type: 'text', text: ' '},
        {type: 'audio', id: 'aud2', name: 'bark.wav'},
      ],
      senderName: 'User',
    };
    element.index = 0;
    await element.updateComplete;

    const content = element.shadowRoot!.querySelector('.message-user-text')!;
    const img = content.querySelector('img.message-image') as HTMLImageElement;
    expect(img).toBeTruthy();
    expect(img.getAttribute('src')).toBe('blob:fake-url');
    expect(img.alt).toBe('cat.png');

    const audioEl =
        content.querySelector('audio.message-audio') as HTMLAudioElement;
    expect(audioEl).toBeTruthy();
    expect(audioEl.getAttribute('src')).toBe('blob:fake-audio-url');

    // Media from an earlier page load are shown as placeholders.
    const missing = content.querySelectorAll('.message-media-missing');
    expect(missing.length).toBe(2);
    expect(missing[0]!.textContent).toContain('dog.png');
    expect(missing[1]!.textContent).toContain('bark.wav');

    expect(content.textContent).toContain('Before');
    expect(content.textContent).toContain('after');
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
    expect(retryBtn.textContent!.trim()).toBe('Retry');
    expect(retryBtn.querySelector('svg')).toBeTruthy();

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

  it('hides a partial bullet while streaming instead of a heading',
     async () => {
       element.message = {
         role: 'assistant',
         text: 'Some intro text\n-',
         senderName: 'Assistant',
       };
       element.index = 1;
       element.streaming = true;
       await element.updateComplete;

       // Without trimming, the `-` would turn the intro into an <h2>.
       const content = element.shadowRoot!.querySelector('.message-content')!;
       expect(content.querySelector('h2')).toBeNull();
       expect(content.textContent).toContain('Some intro text');

       // Once the bullet's text arrives, it renders as a list.
       element.message = {...element.message, text: 'Some intro text\n- item'};
       await element.updateComplete;
       expect(content.querySelector('li')?.textContent).toBe('item');
     });

  it('renders the full text once streaming ends', async () => {
    element.message = {
      role: 'assistant',
      text: 'Intro\n\n---',
      senderName: 'Assistant',
    };
    element.index = 1;
    element.streaming = true;
    await element.updateComplete;
    const content = element.shadowRoot!.querySelector('.message-content')!;
    expect(content.querySelector('hr')).toBeNull();

    element.streaming = false;
    await element.updateComplete;
    expect(content.querySelector('hr')).toBeTruthy();
  });

  it('never gets shorter while streaming lists and emphasis', async () => {
    const replies = [
      'Steps:\n\n1. **First**: one.\n2. **Second**: two.\n3. Third.\n\nDone.',
      'Steps:\n\n1. First.\n\n2. Second.\n\n10. Tenth.\n\nDone.',
      'Intro.\n- Apple\n- **Banana**\n* Cherry\n\n***Note:*** ok.',
      'Table:\n\n| Stage | Bound |\n| :--- | :--- |\n| Prefill | Compute |\n\nOk.',
    ];
    element.index = 1;
    element.streaming = true;
    const drops: string[] = [];
    for (const reply of replies) {
      let lastHeight = 0;
      for (let i = 1; i <= reply.length; i++) {
        const text = reply.slice(0, i);
        element.message = {role: 'assistant', text, senderName: 'Assistant'};
        await element.updateComplete;
        const height = element.offsetHeight;
        if (height < lastHeight) {
          drops.push(`${JSON.stringify(text)}: ${lastHeight} -> ${height}`);
        }
        lastHeight = height;
      }
    }
    expect(drops).toEqual([]);
  });

  it('re-enables Edit button immediately when generation stops', async () => {
    element.message = {
      role: 'user',
      text: 'Prompt while generating',
      senderName: 'User',
    };
    element.index = 0;
    element.disabled = true;
    await element.updateComplete;

    const editBtn =
        element.shadowRoot!.querySelector('.btn-action') as HTMLButtonElement;
    expect(editBtn.disabled).toBeTrue();

    element.disabled = false;
    await element.updateComplete;
    expect(editBtn.disabled).toBeFalse();
  });

  it('renders Stopped chip and partial tokens outside copyable text',
     async () => {
       element.message = {
         role: 'assistant',
         text: 'Partial output',
         senderName: 'Assistant',
         stopped: true,
         tokensCount: '12',
       };
       element.index = 1;
       await element.updateComplete;

       const stoppedChip = element.shadowRoot!.querySelector('.stopped-chip');
       expect(stoppedChip).toBeTruthy();
       expect(stoppedChip!.textContent).toContain('Stopped');

       const stats = element.shadowRoot!.querySelector('.message-stats');
       expect(stats!.textContent).toContain('12');

       const copyBtn = element.shadowRoot!.querySelector('.btn-action') as
           HTMLButtonElement;
       copyBtn.click();
       await new Promise(resolve => setTimeout(resolve, 20));
       expect(mockClipboardWriteText).toHaveBeenCalledWith('Partial output');
     });

  it('renders completed tool calls as a collapsible disclosure', async () => {
    element.message = {
      role: 'assistant',
      text: 'The sum is 650.',
      senderName: 'Assistant',
      toolCalls: [{
        id: 1,
        name: 'run_javascript',
        argsText: '1 + 2',
        status: 'completed',
        outputText: '3',
      }],
    };
    element.index = 1;
    await element.updateComplete;

    const toolDetails =
        element.shadowRoot!.querySelector('.tool-details') as HTMLDetailsElement;
    expect(toolDetails).toBeTruthy();
    expect(toolDetails.open).toBeFalse();
    expect(toolDetails.querySelector('.tool-summary')!.textContent)
        .toContain('Ran run_javascript');
    expect(toolDetails.textContent).toContain('1 + 2');
    expect(toolDetails.textContent).toContain('3');
  });

  it('preserves live litert-code-block instance and user scroll position across streamed tokens',
     async () => {
       const lines = Array.from({length: 40}, (_, i) => `const line${i} = ${i};`)
                         .join('\n');
       element.index = 1;
       element.streaming = true;
       element.message = {
         role: 'assistant',
         text: `Here is code:\n\`\`\`js\n${lines}`,
         senderName: 'Assistant',
       };
       await element.updateComplete;

       const initialBlock =
           element.shadowRoot!.querySelector('litert-code-block')!;
       await (initialBlock as unknown as {updateComplete: Promise<boolean>})
           .updateComplete;
       const scrollRegion = initialBlock.shadowRoot!.querySelector(
                                '.code-scroll-region') as HTMLElement;
       scrollRegion.dispatchEvent(
           new WheelEvent('wheel', {deltaY: -80, bubbles: true}));
       scrollRegion.scrollTop = 42;
       scrollRegion.dispatchEvent(new Event('scroll'));

       element.message = {
         role: 'assistant',
         text: `Here is code:\n\`\`\`js\n${lines}\nconst nextToken = 999;`,
         senderName: 'Assistant',
       };
       await element.updateComplete;
       await new Promise<void>(
           resolve => requestAnimationFrame(() => resolve()));

       const updatedBlock =
           element.shadowRoot!.querySelector('litert-code-block')!;
       expect(updatedBlock).toBe(initialBlock);
       expect(scrollRegion.scrollTop).toBe(42);
     });
});

describe('trimAmbiguousTail', () => {
  it('drops a trailing line of bare markdown punctuation', () => {
    for (const tail of ['-', '- ', '---', '*', '**', '***', '+', '=', '___',
                        '#', '## ', '>', '1', '10', '1.', '2)', '`', '``',
                        '~~']) {
      expect(trimAmbiguousTail(`Intro\n${tail}`))
          .withContext(JSON.stringify(tail))
          .toBe('Intro\n');
    }
    expect(trimAmbiguousTail('***')).toBe('');
  });

  it('keeps lines that have content or are complete', () => {
    for (const text of ['Intro\n- item', 'Intro\n**Bold', 'Intro\n---\n',
                        'Intro\n## Title', 'Intro\n1. one', 'Intro\n```py',
                        'Price: 5-', 'Intro']) {
      expect(trimAmbiguousTail(text)).withContext(text).toBe(text);
    }
  });

  it('holds back incomplete GFM pipe-table headers until delimiter cells arrive',
     () => {
       expect(trimAmbiguousTail('Intro\n\n| Stage | Bound |')).toBe('Intro\n\n');
       expect(trimAmbiguousTail('Intro\n\n| Stage | Bound |\n'))
           .toBe('Intro\n\n');
       expect(trimAmbiguousTail('Intro\n\n| Stage | Bound |\n| :--- |'))
           .toBe('Intro\n\n');

       const completeHeader =
           'Intro\n\n| Stage | Bound |\n| :--- | :--- |';
       expect(trimAmbiguousTail(completeHeader)).toBe(completeHeader);

       const withDataRow =
           'Intro\n\n| Stage | Bound |\n| :--- | :--- |\n| Prefill |';
       expect(trimAmbiguousTail(withDataRow)).toBe(withDataRow);
     });
});
