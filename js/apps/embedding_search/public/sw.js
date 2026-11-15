// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview PWA service worker for the embedding search demo.
 *
 * Caches the app shell (HTML, bundle, manifest, icons) and the Material
 * Symbols icon font with a network-first strategy so the installed app still
 * renders when offline. Model weights and the sample album are deliberately
 * left alone: the app already persists those in Cache Storage / OPFS.
 */

const CACHE_NAME = 'litertlm-embedding-search-shell-v1';

/**
 * Cross-origin hosts whose responses are safe and useful to cache.
 * @const {!Set<string>}
 */
const CACHEABLE_FONT_HOSTS = new Set([
  'fonts.googleapis.com',
  'fonts.gstatic.com',
]);

const scope = /** @type {!ServiceWorkerGlobalScope} */ (self);

scope.addEventListener('install', (e) => {
  // Claim clients instantly without waiting for a refresh.
  scope.skipWaiting();
});

scope.addEventListener('activate', (e) => {
  const event = /** @type {!ExtendableEvent} */ (e);
  event.waitUntil(
      scope.caches.keys()
          .then(
              (keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME)
                                        .map((key) => scope.caches.delete(key))))
          .then(() => scope.clients.claim()));
});

/**
 * @param {!URL} url
 * @return {boolean} Whether the request should be handled by this worker.
 */
function shouldHandle(url) {
  if (CACHEABLE_FONT_HOSTS.has(url.host)) {
    return true;
  }
  if (url.origin !== scope.location.origin) {
    // Model downloads (e.g. Hugging Face) stream gigabytes; never buffer them
    // through the service worker.
    return false;
  }
  // Model weights are cached by the app in a dedicated Cache Storage bucket,
  // and the sample album is copied into OPFS on first use.
  if (url.pathname.endsWith('.litertlm') || url.pathname.includes('/samples/')) {
    return false;
  }
  return true;
}

scope.addEventListener('fetch', (e) => {
  const event = /** @type {!FetchEvent} */ (e);
  if (event.request.method !== 'GET') {
    return;
  }
  const url = new URL(event.request.url);
  if (!shouldHandle(url)) {
    return;
  }

  // Network-first so developers and users pick up updates on reload, falling
  // back to the cached shell only when offline.
  event.respondWith(
      fetch(event.request)
          .then((response) => {
            // Font CSS loaded via <link> without `crossorigin` is an opaque
            // response (status 0); it is still safe to cache for offline use.
            if (response.status === 200 || response.type === 'opaque') {
              const cacheCopy = response.clone();
              scope.caches.open(CACHE_NAME).then((cache) => {
                cache.put(event.request, cacheCopy);
              });
            }
            return response;
          })
          .catch((err) => {
            return scope.caches.match(event.request).then((cachedResponse) => {
              if (cachedResponse) {
                return cachedResponse;
              }
              console.error(
                  '[PWA SW] Network fetch failed and no cache match:',
                  url.pathname, err);
              return new Response('Network error occurred.', {
                status: 408,
                headers: {'Content-Type': 'text/plain'},
              });
            });
          }));
});
