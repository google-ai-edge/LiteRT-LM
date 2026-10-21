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

// Bundles and runs the browser tests for @litert-lm/core with
// jasmine-browser-runner.
//
// Usage:
//
//   node scripts/browser-tests.mjs unit
//       Runs the unit tests in `src/**/*_test.ts` against the sources in
//       `src/` and the Wasm files in `wasm/`.
//
//   node scripts/browser-tests.mjs package [path/to/litert-lm-core-*.tgz]
//       Runs the tests in `package_test/` against a packed npm tarball, using
//       the Wasm files that ship in it. This is what users get from npm. If no
//       tarball is given, this package is packed with `npm pack` first, so
//       `npm run build` and the Wasm files in `wasm/` must be up to date.
//
// Options:
//
//   --keep  Don't delete the temporary consumer project created by `package`.
//
// See `spec/support/jasmine-browser.mjs` for the environment variables that
// control which browser is used and where files are served from.

import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import * as esbuild from 'esbuild';

const require = createRequire(import.meta.url);
const packageDir =
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const jasmineConfig =
    path.join(packageDir, 'spec', 'support', 'jasmine-browser.mjs');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

/**
 * Runs an npm command.
 *
 * When this script runs from an npm lifecycle script (e.g. `prepublishOnly`
 * during `npm publish --dry-run`), npm passes its config to child processes
 * through `npm_config_*` environment variables. Drop `dry_run` so that the
 * tarball is actually packed and installed.
 */
function runNpm(args, options) {
  const env = {...process.env};
  delete env['npm_config_dry_run'];
  return execFileSync(
      npm, args,
      {...options, env, shell: process.platform === 'win32'});
}

function usage() {
  console.error(
      'Usage: node scripts/browser-tests.mjs unit\n' +
      '       node scripts/browser-tests.mjs package [tarball] [--keep]');
  process.exit(2);
}

/**
 * Bundles the test entry points matching `pattern` (relative to `workingDir`)
 * into `workingDir/spec_dist/`.
 *
 * Code splitting keeps a single copy of the library that all tests share, as
 * they would when compiled together internally.
 */
async function bundleTests(workingDir, pattern, alias = {}) {
  const outdir = path.join(workingDir, 'spec_dist');
  fs.rmSync(outdir, {recursive: true, force: true});
  await esbuild.build({
    absWorkingDir: workingDir,
    entryPoints: [pattern],
    outdir,
    bundle: true,
    splitting: true,
    format: 'esm',
    platform: 'browser',
    target: 'esnext',
    sourcemap: true,
    alias,
    logLevel: 'warning',
  });
}

/** Runs the bundled tests in `specDir/spec_dist/` in a browser. */
function runTests(env) {
  // The package's `exports` hide `package.json`, so find it next to the main
  // module.
  const runnerDir = path.dirname(require.resolve('jasmine-browser-runner'));
  const {bin} = JSON.parse(
      fs.readFileSync(path.join(runnerDir, 'package.json'), 'utf8'));
  const runnerBin = path.join(
      runnerDir, typeof bin === 'string' ? bin : bin['jasmine-browser-runner']);
  execFileSync(
      process.execPath,
      [runnerBin, 'runSpecs', `--config=${jasmineConfig}`],
      {cwd: packageDir, stdio: 'inherit', env: {...process.env, ...env}});
}

async function runUnitTests() {
  if (!fs.existsSync(path.join(packageDir, 'wasm'))) {
    throw new Error(
        `No Wasm files found in ${path.join(packageDir, 'wasm')}. Build or ` +
        'download them first.');
  }
  // Tests import the library as `@litert-lm/core`. Resolve it to the sources
  // so that tests and sources share the same modules.
  await bundleTests(
      packageDir, 'src/**/*_test.ts', {'@litert-lm/core': './src'});
  runTests({});
}

async function runPackageTests(tarball, keep) {
  const tempDir =
      fs.mkdtempSync(path.join(os.tmpdir(), 'litert-lm-package-test-'));
  try {
    if (tarball) {
      tarball = path.resolve(tarball);
    } else {
      console.log('Packing @litert-lm/core...');
      const output = runNpm(
          ['pack', '--json', '--pack-destination', tempDir], {
            cwd: packageDir,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'inherit'],
          });
      tarball = path.join(tempDir, JSON.parse(output)[0].filename);
    }
    console.log(`Testing ${tarball}`);

    // Install the tarball into a fresh project, like a user would.
    const consumerDir = path.join(tempDir, 'consumer');
    fs.mkdirSync(consumerDir);
    fs.writeFileSync(
        path.join(consumerDir, 'package.json'),
        JSON.stringify(
            {name: 'litert-lm-package-test', private: true, type: 'module'},
            null, 2));
    runNpm(
        ['install', '--no-audit', '--no-fund', '--no-package-lock', tarball],
        {cwd: consumerDir, stdio: 'inherit'});

    // Copy the tests into the consumer project so that `@litert-lm/core`
    // resolves to the installed package instead of this workspace.
    fs.cpSync(
        path.join(packageDir, 'package_test'),
        path.join(consumerDir, 'package_test'), {recursive: true});
    await bundleTests(consumerDir, 'package_test/**/*_test.ts');

    const installedPackageDir =
        path.join(consumerDir, 'node_modules', '@litert-lm', 'core');
    runTests({
      LITERTLM_TEST_SPEC_DIR: consumerDir,
      LITERTLM_TEST_WASM_DIR: path.join(installedPackageDir, 'wasm'),
    });
  } finally {
    if (keep) {
      console.log(`Kept temporary files in ${tempDir}`);
    } else {
      fs.rmSync(tempDir, {recursive: true, force: true});
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const keep = args.includes('--keep');
  const [mode, ...rest] = args.filter((arg) => arg !== '--keep');
  if (mode === 'unit' && rest.length === 0) {
    await runUnitTests();
  } else if (mode === 'package' && rest.length <= 1) {
    await runPackageTests(rest[0], keep);
  } else {
    usage();
  }
}

main().catch((e) => {
  console.error(e.message ?? e);
  process.exit(1);
});
