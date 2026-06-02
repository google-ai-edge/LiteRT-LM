/**
 * Copyright 2026 Google LLC
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

/**
 * Recursively makes all properties of T required.
 */
export type RecursiveRequired<T> = 
  T extends Set<infer U> ? Set<RecursiveRequired<U>> :
  T extends object ? {
  [K in keyof T] -?: RecursiveRequired<T[K]>;
} : T;

/**
 * Primitive JSON value.
 *
 * This includes 'null' in order to match the JSON spec.
 */
// tslint:disable-next-line:no-undefined-type-alias
export type JsonPrimitive = string|number|boolean|null;

/**
 * Array of JSON values.
 */
export type JsonArray = JsonValue[];

/**
 * Object of JSON values.
 */
export interface JsonObject {
  [key: string]: JsonValue|undefined;
}

/**
 * Any valid JSON value.
 */
export type JsonValue = JsonPrimitive|JsonArray|JsonObject;
