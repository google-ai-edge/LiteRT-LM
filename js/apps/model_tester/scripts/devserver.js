/**
 * Copyright 2025 Google LLC
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

/** @suppress {lintChecks} */

import * as esbuild from 'esbuild';
import http from 'node:http';

// Start esbuild's server on port 3000
let ctx = await esbuild.context({
  bundle: true,
  target: 'es2020',
  outfile: 'dist/bundle.js',
  sourcemap: true,
  entryPoints: ['src/index.ts'],
});

// The return value tells us where esbuild's local server is. It might
// not be 3000 if that port was busy.
let { host, port } = await ctx.serve({ servedir: 'dist', port: 3000});

// Then start a proxy server on port 8000
http.createServer((req, res) => {
  const options = {
    hostname: host,
    port: port,
    path: req.url,
    method: req.method,
    headers: req.headers,
  };

  // Forward each incoming request to esbuild
  const proxyReq = http.request(options, proxyRes => {
    // If esbuild returns "not found", send a custom 404 page
    if (proxyRes.statusCode === 404) {
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end('<h1>Not found</h1>');
      return;
    }

    // Otherwise, forward the response from esbuild to the client.
    // Include cross origin headers for Wasm threaded support.
    const headers = {
        ...proxyRes.headers,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
    };
    res.writeHead(proxyRes.statusCode, headers);
    proxyRes.pipe(res, { end: true });
  });

  // Forward the body of the request to esbuild
  req.pipe(proxyReq, { end: true });
}).listen(8000);

console.log('Server listening on http://localhost:8000');