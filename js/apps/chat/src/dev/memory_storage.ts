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

/** A `Storage` kept in memory, so that dev-only states never touch the real one. */
export class MemoryStorage implements Storage {
  [name: string]: unknown;
  private readonly items: Map<string, string>;

  constructor(initialItems: Iterable<[string, string]> = []) {
    this.items = new Map(initialItems);
  }

  get length(): number {
    return this.items.size;
  }

  clear() {
    this.items.clear();
  }

  getItem(key: string): string|null {
    return this.items.get(key) ?? null;
  }

  key(index: number): string|null {
    return Array.from(this.items.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.items.delete(key);
  }

  setItem(key: string, value: string) {
    this.items.set(key, String(value));
  }
}
