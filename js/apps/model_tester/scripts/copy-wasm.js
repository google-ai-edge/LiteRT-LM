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
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const coreDir = path.dirname(require.resolve('@litert-lm/core/package.json'));
const wasmSrcDir = path.resolve(coreDir, 'wasm');
const wasmDstDir = path.resolve('dist', 'wasm');

fs.mkdirSync(wasmDstDir, { recursive: true });
for (const file of fs.readdirSync(wasmSrcDir)) {
  fs.copyFileSync(path.resolve(wasmSrcDir, file), path.resolve(wasmDstDir, file));
}
