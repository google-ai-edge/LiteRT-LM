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

// Configuration for running the tests in a browser with jasmine-browser-runner.
// See https://github.com/jasmine/jasmine-browser-runner/blob/main/lib/types.js
// for options.
//
// Use `npm test` or `npm run test:package` (see `scripts/browser-tests.mjs`)
// instead of running jasmine-browser-runner directly, since they bundle the
// tests first.
//
// The following environment variables override where files are served from:
//
//   LITERTLM_TEST_SPEC_DIR: Directory containing the bundled tests in
//       `spec_dist/`. Defaults to this package.
//   LITERTLM_TEST_WASM_DIR: Directory of Wasm files served at `/wasm`.
//       Defaults to this package's `wasm/` directory.
//   LITERTLM_TEST_DATA_DIR: Directory of test files (e.g. models) served at
//       `/testdata`. Defaults to the repo's `runtime/testdata/` directory.

import express from 'express';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const packageDir =
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const repoRoot = path.resolve(packageDir, '../../..');

const specDir = process.env.LITERTLM_TEST_SPEC_DIR || packageDir;
const wasmDir =
    process.env.LITERTLM_TEST_WASM_DIR || path.join(packageDir, 'wasm');
const testDataDir = process.env.LITERTLM_TEST_DATA_DIR ||
    path.join(repoRoot, 'runtime', 'testdata');

export default {
  srcDir: path.join(packageDir, 'src'),
  // Sources are imported by the bundled specs, so they don't need to be listed
  // here.
  srcFiles: [],
  specDir,
  specFiles: ['spec_dist/**/*_test.js'],
  helpers: [],
  esmFilenameExtension: '.js',
  modulesWithSideEffectsInSrcFiles: false,
  enableTopLevelAwait: false,
  env: {
    stopSpecOnExpectationFailure: false,
    stopOnSpecFailure: false,
    random: true,
    // Fail if a suite contains multiple suites or specs with the same name.
    forbidDuplicateNames: true,
  },
  middleware: {
    '/wasm': express.static(wasmDir),
    '/testdata': express.static(testDataDir),
  },
  // For security, listen only to localhost.
  listenAddress: 'localhost',
  hostname: 'localhost',
  browser: {
    name: process.env.LITERTLM_TEST_BROWSER || 'headlessChrome',
  },
};
