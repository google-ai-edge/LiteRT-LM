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

import {Backend, EmbeddingEngine, InputOverflowStrategy, loadLiteRtLm, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

const TEST_EMBEDDING_MODEL_PATH = '/testdata/test_embedding.litertlm';
const MAX_INPUT_LENGTH = 128;

describe('EmbeddingEngine', () => {
  let engine: EmbeddingEngine;
  // Exceeds MAX_INPUT_LENGTH (128) tokens for test_embedding.litertlm.
  const longText = '\'s '.repeat(200);

  beforeAll(async () => {
    unloadLiteRtLm();
    await loadLiteRtLm('/wasm');
    engine = await EmbeddingEngine.create({
      model: TEST_EMBEDDING_MODEL_PATH,
      backend: Backend.CPU,
      maxInputLength: MAX_INPUT_LENGTH,
    });
  }, 300_000);

  afterAll(async () => {
    await engine?.delete();
  });

  describe('inputOverflowStrategy', () => {
    it('defaults to ERROR when inputOverflowStrategy is omitted', async () => {
      await expectAsync(engine.computeEmbedding(longText))
          .toBeRejectedWithError(/exceeds maximum supported signature length/i);
    });

    it('rejects over-length input with InputOverflowStrategy.ERROR',
       async () => {
         await expectAsync(engine.computeEmbedding(longText, {
           inputOverflowStrategy: InputOverflowStrategy.ERROR,
         }))
             .toBeRejectedWithError(
                 /exceeds maximum supported signature length/i);
       });

    it('truncates over-length input with InputOverflowStrategy.TRUNCATE',
       async () => {
         const response = await engine.computeEmbedding(longText, {
           inputOverflowStrategy: InputOverflowStrategy.TRUNCATE,
         });
         expect(response.embedding.length).toBeGreaterThan(0);
         expect(response.inputLength).toBeGreaterThan(MAX_INPUT_LENGTH);
         expect(response.truncatedLength).toBe(MAX_INPUT_LENGTH);
         expect(response.numChunks).toBe(1);
       });

    it('chunks and averages over-length input with InputOverflowStrategy.CHUNK_AND_AVERAGE',
       async () => {
         const response = await engine.computeEmbedding(longText, {
           inputOverflowStrategy: InputOverflowStrategy.CHUNK_AND_AVERAGE,
         });
         expect(response.embedding.length).toBeGreaterThan(0);
         expect(response.inputLength).toBeGreaterThan(MAX_INPUT_LENGTH);
         expect(response.truncatedLength).toBeUndefined();
         expect(response.numChunks).toBeGreaterThan(1);
       });

    it('applies inputOverflowStrategy in computeEmbeddingBatch', async () => {
      await expectAsync(engine.computeEmbeddingBatch([longText]))
          .toBeRejectedWithError(/exceeds maximum supported signature length/i);

      await expectAsync(engine.computeEmbeddingBatch([longText], {
        inputOverflowStrategy: InputOverflowStrategy.ERROR,
      }))
          .toBeRejectedWithError(/exceeds maximum supported signature length/i);

      const truncated = await engine.computeEmbeddingBatch([longText], {
        inputOverflowStrategy: InputOverflowStrategy.TRUNCATE,
      });
      expect(truncated.length).toBe(1);
      expect(truncated[0].inputLength).toBeGreaterThan(MAX_INPUT_LENGTH);
      expect(truncated[0].truncatedLength).toBe(MAX_INPUT_LENGTH);
      expect(truncated[0].numChunks).toBe(1);

      const chunked = await engine.computeEmbeddingBatch([longText], {
        inputOverflowStrategy: InputOverflowStrategy.CHUNK_AND_AVERAGE,
      });
      expect(chunked.length).toBe(1);
      expect(chunked[0].inputLength).toBeGreaterThan(MAX_INPUT_LENGTH);
      expect(chunked[0].truncatedLength).toBeUndefined();
      expect(chunked[0].numChunks).toBeGreaterThan(1);
    });

    it('rejects unknown inputOverflowStrategy values', async () => {
      await expectAsync(engine.computeEmbedding(longText, {
        inputOverflowStrategy: 99 as InputOverflowStrategy,
      })).toBeRejectedWithError(/Unknown InputOverflowStrategy/i);
    });
  });
});
