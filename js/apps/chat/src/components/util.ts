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
import {unsafeHTML} from 'lit/directives/unsafe-html.js';
import {unsafeCSS} from 'lit';
import katexStylesText from 'katex/dist/katex.min.css?inline';

/** KaTeX styles to be used in Lit components. */
export const katexStyles = unsafeCSS(katexStylesText);


/** Sets the HTML content of an iframe using standard srcdoc. */
export function setIframeHtml(iframe: HTMLIFrameElement, html: string) {
  iframe.srcdoc = html;
}

/** Sets the HTML content of a sandboxed iframe using standard srcdoc and sandbox attribute. */
export function setSandboxIframeHtml(iframe: HTMLIFrameElement, html: string) {
  iframe.setAttribute('sandbox', 'allow-scripts');
  iframe.srcdoc = html;
}

/** Registers the app service worker using standard navigator.serviceWorker. */
export function registerAppServiceWorker(container: ServiceWorkerContainer): Promise<ServiceWorkerRegistration> {
  return container.register('./sw.js', {scope: './'});
}

/** Renders HTML using standard unsafeHTML. */
export function renderHtml(htmlText: string) {
  return unsafeHTML(htmlText);
}
