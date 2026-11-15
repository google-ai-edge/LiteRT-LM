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

// Devserver entry point: `../index.ts` with URL-driven review states (see
// `dev_hooks.ts`) installed before the app is mounted.

import '../components/chat_app';
import '../components/learn_more_drawer';
import {installDevHooks} from './dev_hooks.js';

installDevHooks(window.location.search);
document.body.prepend(document.createElement('litert-lm-chat-app'));
