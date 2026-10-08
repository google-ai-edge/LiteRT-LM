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

import {EmbeddingEngine, EmbeddingInput, SingleEmbeddingInput} from '@litert-lm/core';

import {applyQueryTemplate, PromptTemplateId} from '../app_config.js';
import {createObjectUrl} from '../utils/hljs_util.js';
import {embeddingFor, FileCategory, IndexedDocument, IndexStore} from './index_store.js';

const TOP_K = 50;

/**
 * Computes cosine similarity between two L2-normalized vectors (equivalent to
 * their dot product when produced with `normalize: true`).
 */
export function cosineSimilarity(
    a: number[] | Float32Array, b: number[] | Float32Array): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
  }
  return dot;
}

/** An image or audio attachment in a search query. */
export interface QueryChip {
  id: string;
  type: 'image' | 'audio';
  name: string;
  blob: Blob;
  previewUrl: string;
}

/**
 * A single element of the query, in the order the user arranged it. Media
 * chips are interleaved with text so the user controls where each image or
 * audio clip sits relative to the surrounding words.
 */
export type QueryPart =
    | {type: 'text'; text: string}
    | {type: 'chip'; chip: QueryChip};

/** A document matched by a search, with its similarity score. */
export interface SearchResultItem {
  doc: IndexedDocument;
  score: number;
  scorePercent: string;
}

/**
 * Service managing search queries, embedding generation for queries,
 * vector similarity computation, and result ranking.
 */
export class SearchStore {
  /** The query, as an ordered sequence of text runs and media chips. */
  parts: QueryPart[] = [];

  selectedCategory: 'all'|FileCategory = 'all';

  isSearching = false;
  results: SearchResultItem[] = [];
  searchLatencyMs = 0;
  queryEmbedMs = 0;
  rankMs = 0;
  comparedCount = 0;
  hasSearched = false;
  selectedDoc: IndexedDocument|null = null;

  /** Cached query vector so category filter changes re-rank without a GPU call. */
  private lastQueryVector: number[]|null = null;

  constructor(
      private readonly onUpdate: () => void,
      private readonly onStatusChange: (status: string) => void,
  ) {}

  /** All media chips in the query, in order. */
  get chips(): QueryChip[] {
    return this.parts.filter((p): p is {type: 'chip'; chip: QueryChip} => p.type === 'chip')
        .map((p) => p.chip);
  }

  /** Whether the query has anything worth searching for. */
  get hasContent(): boolean {
    return this.parts.some(
        (p) => p.type === 'chip' || p.text.trim().length > 0);
  }

  createChip(type: 'image'|'audio', name: string, blob: Blob): QueryChip {
    return {
      id: Math.random().toString(36).substring(2) + Date.now().toString(36),
      type,
      name,
      blob,
      previewUrl: createObjectUrl(blob),
    };
  }

  /** Appends a chip to the end of the query. */
  appendChip(type: 'image'|'audio', name: string, blob: Blob): QueryChip {
    const chip = this.createChip(type, name, blob);
    this.parts = [...this.parts, {type: 'chip', chip}];
    this.onUpdate();
    return chip;
  }

  /**
   * Replaces the query contents. Object URLs belonging to chips that are no
   * longer present are revoked.
   */
  setParts(parts: QueryPart[]) {
    const retained = new Set(
        parts.filter((p): p is {type: 'chip'; chip: QueryChip} => p.type === 'chip')
            .map((p) => p.chip.id));
    for (const part of this.parts) {
      if (part.type === 'chip' && !retained.has(part.chip.id)) {
        URL.revokeObjectURL(part.chip.previewUrl);
      }
    }
    this.parts = parts;
    this.onUpdate();
  }

  removeChip(id: string) {
    this.setParts(
        this.parts.filter((p) => p.type !== 'chip' || p.chip.id !== id));
  }

  clearQuery() {
    this.setParts([]);
    this.selectedCategory = 'all';
    this.results = [];
    this.selectedDoc = null;
    this.hasSearched = false;
    this.lastQueryVector = null;
    this.onUpdate();
  }

  setSelectedCategory(cat: 'all'|FileCategory) {
    this.selectedCategory = cat;
    this.onUpdate();
  }

  setSelectedDoc(doc: IndexedDocument|null) {
    this.selectedDoc = doc;
    this.onUpdate();
  }

  /**
   * Moves the selected result by `delta` (`-1` or `1`), returning true if the
   * selection changed.
   */
  selectAdjacent(delta: -1|1): boolean {
    if (this.results.length === 0) return false;
    const current =
        this.results.findIndex((r) => r.doc.path === this.selectedDoc?.path);
    const next = current === -1 ?
        0 :
        Math.min(this.results.length - 1, Math.max(0, current + delta));
    if (next === current) return false;
    this.setSelectedDoc(this.results[next]!.doc);
    return true;
  }

  async executeSearch(
      engine: EmbeddingEngine,
      indexStore: IndexStore,
      template: PromptTemplateId,
  ): Promise<void> {
    if (this.isSearching) return;
    if (indexStore.indexedDocuments.size === 0) {
      this.onStatusChange('Cannot search: No files are indexed yet.');
      return;
    }

    const inputs: SingleEmbeddingInput[] = [];
    for (const part of this.parts) {
      if (part.type === 'chip') {
        inputs.push(part.chip.blob);
      } else {
        const trimmed = part.text.replace(/\u00a0/g, ' ').trim();
        if (trimmed) inputs.push(trimmed);
      }
    }

    if (inputs.length === 0) {
      this.onStatusChange('Please enter text, upload an image/audio file, or record audio.');
      return;
    }

    // Prompt templates are text-only; multimodal queries stay raw.
    const isTextOnly = inputs.every((input) => typeof input === 'string');
    const embeddingInput: EmbeddingInput = isTextOnly ?
        applyQueryTemplate(template, (inputs as string[]).join(' ')) :
        (inputs.length === 1 ? inputs[0]! : inputs);

    this.isSearching = true;
    this.onStatusChange('Generating multimodal query embedding...');
    this.onUpdate();

    const startTime = performance.now();
    try {
      const queryResp = await engine.computeEmbedding(embeddingInput, {normalize: true});
      const rankStart = performance.now();
      this.queryEmbedMs = rankStart - startTime;

      this.onStatusChange('Ranking matching files...');
      this.lastQueryVector = queryResp.embedding;
      const matchedCount = this.rankDocuments(queryResp.embedding, indexStore, template);
      const endTime = performance.now();
      this.rankMs = endTime - rankStart;
      this.searchLatencyMs = Math.round(endTime - startTime);
      this.hasSearched = true;

      this.onStatusChange(
          `Found ${matchedCount} match(es) in ${this.searchLatencyMs}ms.`);
    } catch (e) {
      console.error('[EmbeddingSearch] Search query failed:', e);
      this.onStatusChange(`Search failed: ${(e as Error).message}`);
    } finally {
      this.isSearching = false;
      this.onUpdate();
    }
  }

  /** Re-filters and re-sorts the index against the last query embedding. */
  rerank(indexStore: IndexStore, template: PromptTemplateId): boolean {
    if (!this.lastQueryVector || this.isSearching) return false;
    const rankStart = performance.now();
    const matchedCount =
        this.rankDocuments(this.lastQueryVector, indexStore, template);
    this.rankMs = performance.now() - rankStart;
    this.queryEmbedMs = 0;
    this.searchLatencyMs = Math.round(this.rankMs);
    this.hasSearched = true;
    this.onStatusChange(`Found ${matchedCount} match(es).`);
    this.onUpdate();
    return true;
  }

  private rankDocuments(
      queryVector: number[], indexStore: IndexStore,
      template: PromptTemplateId): number {
    const matched: SearchResultItem[] = [];
    let compared = 0;
    for (const doc of indexStore.indexedDocuments.values()) {
      if (!indexStore.handleFor(doc.path)) continue;
      if (this.selectedCategory !== 'all' && doc.type !== this.selectedCategory) {
        continue;
      }
      const docEmbedding = embeddingFor(doc, template);
      if (!docEmbedding) continue;

      compared++;
      const score = cosineSimilarity(queryVector, docEmbedding);
      matched.push({
        doc,
        score,
        scorePercent: `${(Math.max(0, score) * 100).toFixed(1)}%`,
      });
    }

    matched.sort((a, b) => b.score - a.score);
    this.results = matched.slice(0, TOP_K);
    this.comparedCount = compared;
    this.selectedDoc = this.results[0]?.doc ?? null;
    return matched.length;
  }
}
