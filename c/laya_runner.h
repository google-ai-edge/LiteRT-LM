// C API for the Laya zero-shot-classifier embedding-input graphs
// (litert-community/Laya-Multilingual-LiteRT). Runs the two supplied graphs
// (main encoder + act head) through LiteRT on CPU. Tokenization, prompt
// building and the fp16 token-table lookup stay on the host (see the package's
// HOST_CONTRACT.md); this wrapper only moves tensors in and out.
//
// All tensors are float32, row-major. Buffer order follows the runtime's
// serving_default signature order; each tensor is identified by its exact
// byte size, which is unique within its graph.
#ifndef LITERT_LM_C_LAYA_RUNNER_H_
#define LITERT_LM_C_LAYA_RUNNER_H_

#include "api_export.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LiteRtLmLaya LiteRtLmLaya;

// Loads both graphs on CPU. Blocking. Returns null on failure.
LITERT_LM_C_API_EXPORT LiteRtLmLaya* litert_lm_laya_create(
    const char* main_graph_path, const char* act_graph_path);

LITERT_LM_C_API_EXPORT void litert_lm_laya_delete(LiteRtLmLaya* laya);

// Static window N of the main graph (256 or 512), 0 on error.
LITERT_LM_C_API_EXPORT int litert_lm_laya_window(const LiteRtLmLaya* laya);

// Embedding width of the main graph (768), 0 on error.
LITERT_LM_C_API_EXPORT int litert_lm_laya_embed_dim(const LiteRtLmLaya* laya);

// One question row through the main graph.
//   inputs_embeds:   [N * dim] float32 (padded rows use the PAD table row)
//   attention_mask:  [N] float32, 1.0 real / 0.0 padding
//   qtype_onehot:    [3] float32 (choice/score/noul)
//   token_logits_out:[N] float32 — logits at every position; the host gathers
//                    its marker positions.
//   pooled_out:      [dim] float32 — pooled_cls = h[:,0].
// Returns 0 on success, nonzero on failure.
LITERT_LM_C_API_EXPORT int litert_lm_laya_run_main(
    LiteRtLmLaya* laya, const float* inputs_embeds, const float* attention_mask,
    const float* qtype_onehot, float* token_logits_out, float* pooled_out);

// The act head. pooled: [dim], feats: [4]
// (top1, top1-top2, normalized entropy, k/255) -> act_logits_out: [2].
// Returns 0 on success, nonzero on failure.
LITERT_LM_C_API_EXPORT int litert_lm_laya_run_act(LiteRtLmLaya* laya,
                                                  const float* pooled,
                                                  const float* feats,
                                                  float* act_logits_out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LITERT_LM_C_LAYA_RUNNER_H_
