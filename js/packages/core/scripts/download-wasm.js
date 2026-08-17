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

import fs from 'fs';
import path from 'path';
import https from 'https';
import { fileURLToPath } from 'url';
import { HttpsProxyAgent } from 'https-proxy-agent';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (process.env.SKIP_WASM_DOWNLOAD) {
  console.log('[Prebuild] SKIP_WASM_DOWNLOAD is set, skipping download.');
  process.exit(0);
}

const packageJsonPath = path.resolve(__dirname, '../package.json');
if (!fs.existsSync(packageJsonPath)) {
  console.error('[Prebuild] Could not find package.json');
  process.exit(1);
}

const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
const version = packageJson.version;
const cdnBaseUrl = `https://cdn.jsdelivr.net/npm/@litert-lm/core@${version}/wasm`;

const files = [
  'litertlm_wasm_internal.js',
  'litertlm_wasm_internal.wasm',
  'litertlm_wasm_compat_internal.js',
  'litertlm_wasm_compat_internal.wasm'
];

const destDir = path.resolve(__dirname, '../wasm');
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Node's built-in `https` module ignores HTTP_PROXY/HTTPS_PROXY env vars, so
// corporate-proxy networks need an explicit proxy agent to reach the CDN.
function getProxyAgent(hostname) {
  const noProxy = (process.env.NO_PROXY || process.env.no_proxy || '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
  const isBypassed = noProxy.some((pattern) =>
    pattern === '*' ||
    hostname === pattern ||
    hostname.endsWith(pattern.startsWith('.') ? pattern : `.${pattern}`)
  );
  if (isBypassed) {
    return undefined;
  }

  const proxyUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy;
  return proxyUrl ? new HttpsProxyAgent(proxyUrl) : undefined;
}

function downloadFile(url, destPath, filename, redirectCount = 0) {
  if (redirectCount > 5) {
    console.error(`[Prebuild] Too many redirects for ${filename}`);
    return;
  }

  const agent = getProxyAgent(new URL(url).hostname);
  https.get(url, { agent }, (response) => {
    if (response.statusCode === 301 || response.statusCode === 302) {
      const redirectUrl = response.headers.location;
      downloadFile(redirectUrl, destPath, filename, redirectCount + 1);
      return;
    }

    if (response.statusCode !== 200) {
      console.error(`[Prebuild] Failed to download ${filename}: HTTP ${response.statusCode}`);
      return;
    }

    const fileStream = fs.createWriteStream(destPath);
    response.pipe(fileStream);
    fileStream.on('finish', () => {
      fileStream.close();
      console.log(`[Prebuild] Successfully downloaded ${filename}`);
    });
  }).on('error', (err) => {
    try {
      fs.unlinkSync(destPath);
    } catch (e) {}
    console.error(`[Prebuild] Error downloading ${filename}:`, err);
  });
}

const checkUrl = `${cdnBaseUrl}/${files[0]}`;
const req = https.request(checkUrl, { method: 'HEAD', agent: getProxyAgent(new URL(checkUrl).hostname) }, (res) => {
  res.resume();
  if (res.statusCode === 404) {
    console.log(`[Prebuild] Version ${version} not available on CDN yet, skipping download.`);
    return;
  }

  for (const file of files) {
    const url = `${cdnBaseUrl}/${file}`;
    const destPath = path.resolve(destDir, file);
    console.log(`[Prebuild] Downloading ${file} from CDN...`);
    downloadFile(url, destPath, file);
  }
});

req.on('error', (err) => {
  console.error(`[Prebuild] Error checking CDN availability:`, err);
});

req.end();
