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

/** Matches elements that can take keyboard focus. */
export const FOCUSABLE_SELECTOR =
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Returns the focused element, looking inside shadow roots. */
export function getDeepActiveElement(): HTMLElement|null {
  let active: Element|null = document.activeElement;
  while (active?.shadowRoot?.activeElement) {
    active = active.shadowRoot.activeElement;
  }
  return active instanceof HTMLElement ? active : null;
}

/**
 * Returns the visible, non-inert focusable elements in `root`, including those
 * in shadow roots, in document order.
 */
export function getFocusableElementsDeep(root: ParentNode): HTMLElement[] {
  const result: HTMLElement[] = [];
  const collect = (scope: ParentNode) => {
    for (const el of Array.from(scope.querySelectorAll<HTMLElement>('*'))) {
      if (el.closest('[inert]')) continue;
      if (el.matches(FOCUSABLE_SELECTOR) &&
          (el.offsetParent !== null || el.getClientRects().length > 0)) {
        result.push(el);
      }
      if (el.shadowRoot) {
        collect(el.shadowRoot);
      }
    }
  };
  collect(root);
  return result;
}

/** Keeps Tab and Shift+Tab focus within `roots`, wrapping at the ends. */
export function trapTabKey(
    e: KeyboardEvent, ...roots: Array<HTMLElement|null>) {
  if (e.key !== 'Tab' || e.defaultPrevented) return;
  const focusables = roots.flatMap(
      root => root ? getFocusableElementsDeep(root) : []);
  if (focusables.length === 0) return;
  const target =
      e.shiftKey ? focusables[focusables.length - 1]! : focusables[0]!;
  const edge = e.shiftKey ? focusables[0] : focusables[focusables.length - 1];
  const active = getDeepActiveElement();
  if (!active || active === edge || !focusables.includes(active)) {
    e.preventDefault();
    target.focus();
  }
}

/**
 * Runs `focus` now and again in the next frame, in case its target wasn't
 * focusable yet (e.g. it was still hidden or inert).
 */
export function focusNowAndNextFrame(focus: () => void) {
  focus();
  requestAnimationFrame(focus);
}
