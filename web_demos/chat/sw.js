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

const CACHE_NAME = 'litertlm-chat-shell-v1';

const scope = /** @type {!ServiceWorkerGlobalScope} */ (self);

scope.addEventListener('install', (e) => {
  // Claim client instantly without waiting for refresh
  scope.skipWaiting();
});

scope.addEventListener('activate', (e) => {
  const event = /** @type {!ExtendableEvent} */ (e);
  event.waitUntil(scope.clients.claim());
});

scope.addEventListener('fetch', (e) => {
  const event = /** @type {!FetchEvent} */ (e);
  const url = new URL(event.request.url);

  // Only intercept GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // 1. Ignore giant model weight files (.litertlm)
  // Caching gigabytes inside Service Worker Cache causes performance bottlenecks and storage exhaustion.
  // Model caching is already managed efficiently in main.ts inside a distinct Chrome Cache Storage bucket.
  if (url.pathname.endsWith('.litertlm')) {
    return;
  }

  // 2. Intercept local UI shell requests with Network-First strategy
  // Prioritizes network fetches so developers and users receive updates instantly on reload,
  // falling back strictly to local shell caches when offline.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache local origin resources dynamically on the fly
        if (response.status === 200 && url.origin === scope.location.origin) {
          const cacheCopy = response.clone();
          scope.caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, cacheCopy);
          });
        }
        return response;
      })
      .catch((err) => {
        console.log('[PWA SW] Network failed or offline, falling back to cache for:', url.pathname);
        return scope.caches.match(event.request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          console.error('[PWA SW] Network fetch failed and no cache match:', err);
          return new Response('Network error occurred.', { status: 408, headers: { 'Content-Type': 'text/plain' } });
        });
      })
  );
});
