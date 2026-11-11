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

import {getDeepActiveElement, getFocusableElementsDeep, trapTabKey} from './focus_util.js';

describe('focus_util', () => {
  let container: HTMLElement;
  let first: HTMLElement;
  let inner: HTMLElement;

  beforeEach(() => {
    container = document.createElement('div');
    container.innerHTML = '<button id="first">1</button>' +
        '<input id="hidden" hidden><button id="disabled" disabled>2</button>' +
        '<span id="host"></span>';
    document.body.append(container);
    container.querySelector('#host')!.attachShadow({mode: 'open'}).innerHTML =
        '<button id="inner">3</button><div inert><button>4</button></div>';
    first = container.querySelector<HTMLElement>('#first')!;
    inner = container.querySelector('#host')!.shadowRoot!.querySelector<
        HTMLElement>('#inner')!;
  });

  afterEach(() => {
    container.remove();
  });

  function pressTab(shiftKey = false): KeyboardEvent {
    const event =
        new KeyboardEvent('keydown', {key: 'Tab', shiftKey, cancelable: true});
    trapTabKey(event, container);
    return event;
  }

  it('finds visible, enabled, non-inert elements in shadow roots too', () => {
    expect(getFocusableElementsDeep(container)).toEqual([first, inner]);
  });

  it('wraps Tab from the last element to the first', () => {
    inner.focus();
    expect(pressTab().defaultPrevented).toBeTrue();
    expect(getDeepActiveElement()).toBe(first);
  });

  it('wraps Shift+Tab from the first element to the last', () => {
    first.focus();
    expect(pressTab(true).defaultPrevented).toBeTrue();
    expect(getDeepActiveElement()).toBe(inner);
  });

  it('leaves Tab alone between the ends', () => {
    first.focus();
    expect(pressTab().defaultPrevented).toBeFalse();
    expect(getDeepActiveElement()).toBe(first);
  });
});
