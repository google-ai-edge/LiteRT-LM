/* Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Version script for liblitert-lm.so on Linux and Android.
 *
 * Exports only the LiteRT-LM C API. Everything else, including the symbols of
 * statically linked dependencies such as LiteRT, TFLite, Abseil and protobuf,
 * stays local, as -exported_symbol does on Apple platforms.
 *
 * Every C API 1.x symbol is in LITERT_LM_C_API_1.0. Keep it that way for new
 * 1.x functions. A C API 2.0.0 can add a new version node for the symbols
 * whose ABI it changes.
 */
LITERT_LM_C_API_1.0 {
  global:
    litert_lm_*;
  local:
    *;
};
