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
import {createRequire} from 'module';
import path from 'path';
import {fileURLToPath} from 'url';
import {defineConfig} from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  base: './',
  build: {
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      allow: ['.'],
    }
  },
  plugins: [
    {
      name: 'copy-assets-on-build',
      closeBundle() {
        const outDir = path.resolve(__dirname, 'dist');
        if (!fs.existsSync(outDir)) return;

        // Symlink all model files from models/ to dist/models/ to save disk space during local builds
        const modelSrcDir = path.resolve(__dirname, 'models');
        const modelDstDir = path.resolve(outDir, 'models');
        if (fs.existsSync(modelSrcDir)) {
          if (!fs.existsSync(modelDstDir)) {
            fs.mkdirSync(modelDstDir, { recursive: true });
          }
          const files = fs.readdirSync(modelSrcDir);
          for (const file of files) {
            if (file.endsWith('.litertlm')) {
              const srcPath = path.resolve(modelSrcDir, file);
              const dstPath = path.resolve(modelDstDir, file);
              
              // Remove existing file or symlink if present to avoid EEXIST conflicts
              try {
                fs.unlinkSync(dstPath);
              } catch (e: unknown) {
                if ((e as {code?: string}).code !== 'ENOENT') throw e;
              }
              
              console.log(`[Vite] Creating symlink for model ${file} in dist/models/`);
              fs.symlinkSync(srcPath, dstPath, 'file');
            }
          }
          console.log('[Vite] Symlinked all .litertlm models to dist/models/');
        }
      }
    },
    {
      name: 'serve-static-assets-in-place',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url ? req.url.split('?')[0] : '';
          
          // Serve any .litertlm model file in-place from the workspace (root or subdirs)
          if (url.endsWith('.litertlm')) {
            const filePath = path.resolve(__dirname, url.slice(1));
            if (fs.existsSync(filePath)) {
              const stat = fs.statSync(filePath);
              res.writeHead(200, {
                'Content-Type': 'application/octet-stream',
                'Content-Length': stat.size,
                'Cross-Origin-Resource-Policy': 'cross-origin',
              });
              fs.createReadStream(filePath).pipe(res);
              return;
            } else {
              console.warn(`[Vite Middleware] Model file not found at ${filePath}`);
            }
          }

          // Serve the WASM files from node_modules/@litert-lm/core/wasm/
          if (url.startsWith('/wasm/')) {
            const filename = url.substring(6); // remove '/wasm/'
            const require = createRequire(import.meta.url);
            const coreDir = path.dirname(require.resolve('@litert-lm/core/package.json'));
            const filePath = path.resolve(coreDir, 'wasm', filename);
            if (fs.existsSync(filePath)) {
              let contentType = 'application/octet-stream';
              if (filename.endsWith('.js')) {
                contentType = 'application/javascript';
              } else if (filename.endsWith('.wasm')) {
                contentType = 'application/wasm';
              }
              res.writeHead(200, {
                'Content-Type': contentType,
                'Cross-Origin-Resource-Policy': 'cross-origin',
              });
              fs.createReadStream(filePath).pipe(res);
              return;
            }
          }

          next();
        });
      }
    }
  ]
});
