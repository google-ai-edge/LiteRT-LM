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

import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
/* tslint:disable:no-new-decorators */

/**
 * A dropdown menu that opens upward from a slotted `slot="trigger"` button.
 * Light DOM children with class `dropdown-item` and `data-value` are options.
 */
@customElement('custom-dropdown')
export class CustomDropdown extends LitElement {
  @property({type: String})
  value = '';

  @state()
  isOpen = false;

  @state()
  private activeIndex = -1;

  override connectedCallback(): void {
    super.connectedCallback();
    this.addEventListener('focusout', this.handleFocusOut);
  }

  override disconnectedCallback(): void {
    this.removeEventListener('focusout', this.handleFocusOut);
    super.disconnectedCallback();
    document.removeEventListener('click', this.handleOutsideClick);
    window.removeEventListener('keydown', this.handleGlobalKeyDown, true);
  }

  private getTriggerButton(): HTMLElement|null {
    return this.querySelector<HTMLElement>('[slot="trigger"]');
  }

  private syncTriggerAttributes() {
    const trigger = this.getTriggerButton();
    if (!trigger) return;
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', this.isOpen ? 'true' : 'false');
    trigger.classList.toggle('open', this.isOpen);
  }

  /** Opens the dropdown menu programmatically. */
  open() {
    this.setOpen(true);
  }

  /** Closes the dropdown menu programmatically. */
  close() {
    this.setOpen(false);
  }

  /** Focuses the dropdown trigger button. */
  focusButton() {
    this.getTriggerButton()?.focus();
  }

  private getEnabledItems(): HTMLElement[] {
    return Array.from(this.querySelectorAll('.dropdown-item'))
        .filter((el): el is HTMLElement =>
            el instanceof HTMLElement && !el.hasAttribute('disabled'));
  }

  private syncSelectedAttributes() {
    const enabled = this.getEnabledItems();
    for (const item of Array.from(this.querySelectorAll('.dropdown-item'))) {
      if (!(item instanceof HTMLElement)) continue;
      const isSelected = item.dataset['value'] === this.value;
      item.toggleAttribute('data-selected', isSelected);
      item.setAttribute('role', 'option');
      item.setAttribute('tabindex', '-1');
      item.setAttribute('aria-selected', isSelected ? 'true' : 'false');
      if (item.hasAttribute('disabled')) {
        item.setAttribute('aria-disabled', 'true');
      } else {
        item.removeAttribute('aria-disabled');
      }
    }
    enabled.forEach((item, idx) => {
      item.toggleAttribute('data-focused', this.isOpen && idx === this.activeIndex);
    });
  }

  private setOpen(isOpen: boolean) {
    if (this.isOpen === isOpen) {
      return;
    }
    this.isOpen = isOpen;
    this.syncTriggerAttributes();
    if (this.isOpen) {
      // Fit the menu in the space above the trigger.
      const rect = this.getBoundingClientRect();
      if (rect.top > 0) {
        const availableAbove = Math.max(120, Math.floor(rect.top - 60));
        this.style.setProperty('--dropdown-up-max-h', `${availableAbove}px`);
      }
      const enabled = this.getEnabledItems();
      const selectedIdx =
          enabled.findIndex(el => el.dataset['value'] === this.value);
      this.activeIndex = selectedIdx >= 0 ? selectedIdx : 0;
      this.syncSelectedAttributes();
      window.addEventListener('keydown', this.handleGlobalKeyDown, true);
      setTimeout(() => {
        if (this.isConnected && this.isOpen) {
          document.addEventListener('click', this.handleOutsideClick);
        }
      }, 0);
    } else {
      this.style.removeProperty('--dropdown-up-max-h');
      this.activeIndex = -1;
      this.syncSelectedAttributes();
      document.removeEventListener('click', this.handleOutsideClick);
      window.removeEventListener('keydown', this.handleGlobalKeyDown, true);
    }
    this.dispatchEvent(
        new CustomEvent('toggle-open', {detail: {open: this.isOpen}}));
  }

  private readonly handleFocusOut = (e: FocusEvent) => {
    if (!this.isOpen) return;
    const next = e.relatedTarget;
    if (!(next instanceof Node)) return;
    if (!this.contains(next) && !this.renderRoot.contains(next)) {
      this.setOpen(false);
    }
  };

  /** Closes on Escape before anything else on the page sees it. */
  private readonly handleGlobalKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && this.isOpen) {
      e.preventDefault();
      e.stopPropagation();
      this.setOpen(false);
      this.focusButton();
    }
  };

  private readonly handleOutsideClick = (e: MouseEvent) => {
    if (!e.composedPath().includes(this)) {
      this.setOpen(false);
    }
  };

  static override styles = css`
    :host {
      position: relative;
      font-family: inherit;
      display: inline-flex;
      align-items: center;
      min-width: 0;
      max-width: 100%;
    }

    .dropdown-content {
      display: none;
      position: absolute;
      bottom: calc(100% + 8px);
      left: 0;
      background-color: var(--bg-card, #171c26);
      min-width: 316px;
      width: max-content;
      max-width: min(376px, calc(100vw - 32px));
      box-shadow: var(--shadow-md, 0 10px 28px rgba(0, 0, 0, 0.48));
      z-index: 100000;
      border-radius: 12px;
      border: 1px solid var(--border-strong, #364156);
      max-height: min(390px, var(--dropdown-up-max-h, calc(100dvh - 156px)));
      overflow-y: auto;
      padding: 5px;
      box-sizing: border-box;
    }

    @media (max-width: 560px) {
      .dropdown-content {
        min-width: 0;
        width: calc(100vw - 44px);
        max-width: calc(100vw - 24px);
        max-height: min(380px, var(--dropdown-up-max-h, calc(100dvh - 160px)));
      }
    }

    .dropdown-content.show {
      display: block;
    }
  `;

  private selectElement(target: HTMLElement) {
    if (target.dataset['value'] && !target.hasAttribute('disabled')) {
      this.value = target.dataset['value'];
      this.setOpen(false);
      this.focusButton();
      this.dispatchEvent(new CustomEvent('change', {detail: this.value}));
    }
  }

  private moveActiveIndex(index: number) {
    const enabled = this.getEnabledItems();
    if (enabled.length === 0) return;
    this.activeIndex = (index + enabled.length) % enabled.length;
    this.syncSelectedAttributes();
    const target = enabled[this.activeIndex];
    target?.scrollIntoView?.({block: 'nearest'});
    target?.focus({preventScroll: true});
  }

  /** Whether `target` is in a control other than an item or the trigger. */
  private isAuxiliaryControl(target: EventTarget|null): boolean {
    const control = (target as Element | null)?.closest?.('button, input, a');
    return Boolean(
        control && !control.matches('.dropdown-item, [slot="trigger"]'));
  }

  private handleKeyDown(e: KeyboardEvent) {
    // Let other buttons in the menu (e.g. remove a model) handle their keys.
    if (this.isAuxiliaryControl(e.target)) return;
    const lastIndex = this.getEnabledItems().length - 1;
    const step = {ArrowDown: 1, ArrowUp: -1}[e.key];
    if (step !== undefined) {
      e.preventDefault();
      if (this.isOpen) {
        this.moveActiveIndex(this.activeIndex + step);
      } else {
        this.setOpen(true);
        void this.updateComplete.then(() => {
          if (this.isOpen) this.moveActiveIndex(this.activeIndex);
        });
      }
    } else if ((e.key === 'Home' || e.key === 'End') && this.isOpen) {
      e.preventDefault();
      this.moveActiveIndex(e.key === 'Home' ? 0 : lastIndex);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!this.isOpen) {
        this.setOpen(true);
        return;
      }
      const target = this.getEnabledItems()[this.activeIndex];
      if (target) {
        this.selectElement(target);
      }
    }
  }

  private handleItemClick(e: Event) {
    if (this.isAuxiliaryControl(e.target)) return;
    const target =
        (e.target as HTMLElement | null)?.closest<HTMLElement>('.dropdown-item');
    if (target) {
      this.selectElement(target);
    }
  }

  override updated() {
    this.syncSelectedAttributes();
    this.syncTriggerAttributes();
  }

  override render() {
    // When the items supply their own listbox, the menu is just a group.
    const hasLightListbox = Boolean(this.querySelector('[role="listbox"]'));
    return html`
      <slot
        name="trigger"
        @slotchange=${() => this.syncTriggerAttributes()}
        @click=${() => this.setOpen(!this.isOpen)}
        @keydown=${this.handleKeyDown}
      ></slot>
      <div
        id="dropdown-options"
        class="dropdown-content ${this.isOpen ? 'show' : ''}"
        role=${hasLightListbox ? 'group' : 'listbox'}
        aria-label=${hasLightListbox ? 'Model selector menu' : 'Available models'}
        @keydown=${this.handleKeyDown}
      >
        <slot @slotchange=${() => {
          this.syncSelectedAttributes();
          this.requestUpdate();
        }} @click=${this.handleItemClick}></slot>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'custom-dropdown': CustomDropdown;
  }
}
