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

import './custom_dropdown';

import {CustomDropdown} from './custom_dropdown.js';

describe('custom-dropdown', () => {
  let element: CustomDropdown;
  let trigger: HTMLButtonElement;
  let content: HTMLElement;

  beforeEach(async () => {
    element = document.createElement('custom-dropdown');
    element.innerHTML = `
      <button slot="trigger" type="button">Model</button>
      <div class="dropdown-item" data-value="option1">Option 1</div>
      <div class="dropdown-item" data-value="option2" disabled>Option 2</div>
      <div class="dropdown-row">
        <div class="dropdown-item" data-value="option3">Option 3</div>
        <button class="delete-cache-btn">X</button>
      </div>
    `;
    document.body.appendChild(element);
    await element.updateComplete;
    trigger = element.querySelector('[slot="trigger"]')!;
    content = element.shadowRoot!.querySelector('.dropdown-content')!;
  });

  afterEach(() => {
    element.remove();
  });

  function item(value: string): HTMLElement {
    return element.querySelector(`[data-value="${value}"]`)!;
  }

  it('toggles open when the trigger is clicked', async () => {
    expect(content.classList.contains('show')).toBeFalse();
    expect(trigger.getAttribute('aria-expanded')).toBe('false');

    trigger.click();
    await element.updateComplete;
    expect(content.classList.contains('show')).toBeTrue();
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(trigger.classList.contains('open')).toBeTrue();

    trigger.click();
    await element.updateComplete;
    expect(content.classList.contains('show')).toBeFalse();
  });

  it('selects an item and fires a change event', async () => {
    const changeSpy = jasmine.createSpy('change');
    element.addEventListener('change', changeSpy);
    trigger.click();
    await element.updateComplete;

    item('option1').click();
    await element.updateComplete;

    expect(element.value).toBe('option1');
    expect(changeSpy).toHaveBeenCalledTimes(1);
    expect(content.classList.contains('show')).toBeFalse();
  });

  it('does not select disabled items', async () => {
    trigger.click();
    await element.updateComplete;

    item('option2').click();
    await element.updateComplete;

    expect(element.value).toBe('');
    expect(content.classList.contains('show')).toBeTrue();
  });

  it('does not select or close when another button in the menu is clicked',
     async () => {
       const changeSpy = jasmine.createSpy('change');
       element.addEventListener('change', changeSpy);
       trigger.click();
       await element.updateComplete;

       element.querySelector<HTMLButtonElement>('.delete-cache-btn')!.click();
       await element.updateComplete;

       expect(changeSpy).not.toHaveBeenCalled();
       expect(content.classList.contains('show')).toBeTrue();
     });

  it('closes when clicking outside', async () => {
    trigger.click();
    await element.updateComplete;
    // The outside-click listener is added after the opening click.
    await new Promise(resolve => setTimeout(resolve));

    document.body.click();
    await element.updateComplete;

    expect(content.classList.contains('show')).toBeFalse();
  });

  it('closes on Escape and returns focus to the trigger', async () => {
    trigger.click();
    await element.updateComplete;

    window.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
    await element.updateComplete;

    expect(element.isOpen).toBeFalse();
    expect(document.activeElement).toBe(trigger);
  });

  it('exposes listbox and option ARIA roles with aria-selected state',
     async () => {
       element.value = 'option1';
       await element.updateComplete;

       expect(trigger.getAttribute('aria-haspopup')).toBe('listbox');
       expect(content.getAttribute('role')).toBe('listbox');
       expect(item('option1').getAttribute('role')).toBe('option');
       expect(item('option1').getAttribute('aria-selected')).toBe('true');
       expect(item('option2').getAttribute('aria-selected')).toBe('false');
       expect(item('option2').getAttribute('aria-disabled')).toBe('true');
     });

  it('navigates enabled items with the arrow, Home, and End keys', async () => {
    element.value = 'option3';
    await element.updateComplete;
    const press = async (key: string) => {
      trigger.dispatchEvent(new KeyboardEvent(
          'keydown', {key, bubbles: true, cancelable: true}));
      await element.updateComplete;
    };

    // ArrowUp opens the menu on the selected item.
    await press('ArrowUp');
    expect(element.isOpen).toBeTrue();
    expect(item('option3').hasAttribute('data-focused')).toBeTrue();

    // Disabled items are skipped.
    await press('ArrowUp');
    expect(item('option1').hasAttribute('data-focused')).toBeTrue();
    await press('End');
    expect(item('option3').hasAttribute('data-focused')).toBeTrue();
    await press('Home');
    expect(item('option1').hasAttribute('data-focused')).toBeTrue();

    await press('Enter');
    expect(element.value).toBe('option1');
    expect(element.isOpen).toBeFalse();
  });
});
