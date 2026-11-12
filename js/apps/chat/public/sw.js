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

const CACHE_PREFIX = 'litertlm-chat-shell-';
const CACHE_NAME = `${CACHE_PREFIX}v2`;

/** @type {!Array<string>} */
const PRECACHE_URLS = [
  './',
  './index.html',
  './manifest.json',
  './assets/LiteRT_App_1024x1024px_Symbol-only_1.svg',
  './assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg',
  './assets/LiteRT_Logo_Symbol-only_RGB_Color_Blue.svg',
  './icons/icon-192.png',
  './icons/icon-maskable-192.png',
];

const scope = /** @type {!ServiceWorkerGlobalScope} */ (self);

/**
 * Returns whether a URL should be excluded from shell caching (e.g. cross-origin
 * model hosts or `.litertlm` / `/models/` weight files).
 * @param {!URL} url
 * @return {boolean}
 */
function isExcludedAssetUrl(url) {
  const isSameOrigin = url.origin === scope.location.origin;
  const isWasmCdn = url.host === 'cdn.jsdelivr.net';
  return (
    (!isSameOrigin && !isWasmCdn) ||
    url.pathname.endsWith('.litertlm') ||
    url.pathname.includes('/models/')
  );
}

/**
 * Strips the `redirected` flag from a fetched Response so cached navigation
 * fallbacks satisfy the Fetch specification (`redirect !== 'follow'`).
 * @param {!Response} response
 * @return {!Promise<!Response>}
 */
function sanitizeResponse(response) {
  if (!response.redirected) {
    return Promise.resolve(response);
  }
  return response.blob().then(
    (body) =>
      new Response(body, {
        status: response.status,
        statusText: response.statusText,
        headers: new Headers(response.headers),
      })
  );
}

/**
 * Returns the cacheable scripts and stylesheets that an HTML page references.
 * @param {string} html
 * @return {!Array<string>}
 */
function findShellAssets(html) {
  /** @type {!Set<string>} */
  const urls = new Set();
  const pattern =
    /<(?:script|link)\b[^>]*?\b(?:src|href)\s*=\s*["']([^"']+)["']/gi;
  for (const match of html.matchAll(pattern)) {
    try {
      const url = new URL(match[1], scope.registration.scope);
      if (url.origin === scope.location.origin && !isExcludedAssetUrl(url)) {
        urls.add(url.href);
      }
    } catch (err) {
      // Ignore malformed references.
    }
  }
  return Array.from(urls);
}

/**
 * Fetches and stores a list of URLs in the shell cache with redirect stripping.
 * @param {!Cache} cache
 * @param {!Array<string>} urls
 * @return {!Promise<!Array<void>>}
 */
function cacheUrlList(cache, urls) {
  return Promise.all(
    urls.map((url) =>
      fetch(url, {cache: 'reload'})
        .then((response) => {
          if (!response.ok) {
            return undefined;
          }
          return sanitizeResponse(response).then((clean) =>
            cache.put(url, clean)
          );
        })
        .catch(() => undefined)
    )
  );
}

scope.addEventListener('install', (e) => {
  const event = /** @type {!ExtendableEvent} */ (e);
  const isFirstInstall = !scope.registration.active;
  event.waitUntil(
    scope.caches
      .open(CACHE_NAME)
      .then((cache) =>
        cacheUrlList(cache, PRECACHE_URLS)
          // Also cache the new page's hashed bundles, so that the new shell
          // works offline before the user reloads into it.
          .then(() => cache.match('./index.html'))
          .then((response) => response || cache.match('./'))
          .then((response) => (response ? response.text() : ''))
          .then((html) => cacheUrlList(cache, findShellAssets(html)))
      )
      .then(() => {
        // Activate immediately on first install; on subsequent updates, stay in
        // waiting state until the UI prompts the user to reload cleanly.
        if (isFirstInstall) {
          return scope.skipWaiting();
        }
      })
  );
});

scope.addEventListener('message', (e) => {
  const event = /** @type {!ExtendableMessageEvent} */ (e);
  const data =
    /** @type {?{type: (string|undefined), urls: (!Array<string>|undefined)}} */ (
      event.data
    );
  if (!data) {
    return;
  }
  if (data.type === 'SKIP_WAITING') {
    scope.skipWaiting();
  } else if (data.type === 'CACHE_URLS' && Array.isArray(data.urls)) {
    const validUrls = data.urls.filter((raw) => {
      try {
        const parsed = new URL(raw, scope.location.origin);
        return (
          parsed.origin === scope.location.origin && !isExcludedAssetUrl(parsed)
        );
      } catch (err) {
        return false;
      }
    });
    event.waitUntil(
      scope.caches
        .open(CACHE_NAME)
        .then((cache) => cacheUrlList(cache, validUrls))
    );
  }
});

scope.addEventListener('activate', (e) => {
  const event = /** @type {!ExtendableEvent} */ (e);
  event.waitUntil(
    scope.caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            // Only prune outdated shell caches; NEVER touch 'litertlm-models'.
            .filter(
              (key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME
            )
            .map((key) => scope.caches.delete(key))
        )
      )
      .then(() => scope.clients.claim())
  );
});

scope.addEventListener('fetch', (e) => {
  const event = /** @type {!FetchEvent} */ (e);
  if (event.request.method !== 'GET') {
    return;
  }

  const url = new URL(event.request.url);
  if (isExcludedAssetUrl(url)) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.status === 200) {
          const copyForCache = response.clone();
          event.waitUntil(
            Promise.all([
              sanitizeResponse(copyForCache),
              scope.caches.open(CACHE_NAME),
            ]).then(([clean, cache]) => cache.put(event.request, clean))
          );
        }
        return response;
      })
      .catch(() =>
        scope.caches
          .match(event.request)
          .then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            if (event.request.mode === 'navigate') {
              return scope.caches
                .match(event.request, {ignoreSearch: true})
                .then(
                  (navMatch) =>
                    navMatch || scope.caches.match('./index.html')
                )
                .then((indexMatch) => indexMatch || scope.caches.match('./'));
            }
            return undefined;
          })
          .then((fallback) => {
            if (fallback) {
              return fallback;
            }
            return new Response('Offline — resource not cached.', {
              status: 503,
              headers: {'Content-Type': 'text/plain; charset=utf-8'},
            });
          })
      )
  );
});
