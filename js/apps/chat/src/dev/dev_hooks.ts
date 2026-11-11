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

// Installs URL-driven review states for the devserver.
//
// - `?demo=<mode>`: deterministic states with a simulated engine (see
//   `installDemoMode`).
// - `?theme=light|dark`: forces a theme without saving it.

import {THEME_STORAGE_KEY} from '../components/chat_app.js';
import {devHooks} from '../state_controller.js';

import {installDemoMode} from './demo_mode.js';
import {MemoryStorage} from './memory_storage.js';

/** Installs the review states that `search` (a URL query) asks for. */
export function installDevHooks(search: string) {
  installDemoMode(search);

  const theme = new URLSearchParams(search).get('theme');
  if (theme === 'light' || theme === 'dark') {
    devHooks.storage ??= new MemoryStorage(
        Object.entries(window.localStorage) as Array<[string, string]>);
    devHooks.storage.setItem(THEME_STORAGE_KEY, theme);
  }
}
