// Standalone .tflite text-embedding runner C API.
//
// Runs a self-contained embedding graph (e.g. EmbeddingGemma's published
// `*_mixed-precision.tflite`, signature `text_batch i32[1,S] -> encodings
// f32[1,D]`) through LiteRT directly — independent of the LiteRT-LM
// EmbeddingEngine pipeline (which requires a split embedder/text-encoder
// bundle that is not publicly distributed).
//
// Tokenization uses a SentencePiece model file (Gemma family), BOS-prefixed,
// right-padded with 0 to the model's static input length.

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_TFLITE_EMBED_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_TFLITE_EMBED_H_

#include <stddef.h>

#if defined(__APPLE__)
#include "engine.h"  // NOLINT
#else
#include "c/engine.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LiteRtLmTfliteEmbed LiteRtLmTfliteEmbed;

// Loads the tflite embedding graph and the SentencePiece tokenizer. Returns
// NULL on failure (engine logs carry the reason).
LITERT_LM_C_API_EXPORT LiteRtLmTfliteEmbed* litert_lm_tflite_embed_create(
    const char* tflite_path, const char* sentencepiece_path);

// Output dimension of the loaded graph (e.g. 768).
LITERT_LM_C_API_EXPORT int litert_lm_tflite_embed_dim(
    const LiteRtLmTfliteEmbed* embed);

// Embeds one text; `out` must hold at least dim floats (L2-normalized by the
// graph's projection head). Returns 0 on success, non-zero on failure.
LITERT_LM_C_API_EXPORT int litert_lm_tflite_embed_text(
    LiteRtLmTfliteEmbed* embed, const char* text, float* out);

LITERT_LM_C_API_EXPORT void litert_lm_tflite_embed_delete(
    LiteRtLmTfliteEmbed* embed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_TFLITE_EMBED_H_
