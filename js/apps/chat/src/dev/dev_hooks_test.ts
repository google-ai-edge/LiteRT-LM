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


import {THEME_STORAGE_KEY} from '../components/chat_app.js';
import {devHooks} from '../state_controller.js';

import {installDevHooks} from './dev_hooks.js';

describe('installDevHooks', () => {
  afterEach(() => {
    delete devHooks.storage;
    delete devHooks.createModelLoader;
    delete devHooks.onConnected;
    window.localStorage.clear();
  });

  it('installs nothing without review-state parameters', () => {
    installDevHooks('?foo=bar');

    expect(devHooks.storage).toBeUndefined();
    expect(devHooks.onConnected).toBeUndefined();
  });

  it('forces ?theme= without saving it', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'system');

    installDevHooks('?theme=dark');

    expect(devHooks.storage?.getItem(THEME_STORAGE_KEY)).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('system');
  });
});
