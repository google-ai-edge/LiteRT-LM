---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
tags:
  - litert
  - text-to-image
  - on-device
  - diffusion-transformer
  - int8
pipeline_tag: text-to-image
library_name: litert
base_model_relation: quantized
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

LiteRT is Google's on-device runtime, the new name for TensorFlow Lite (Android: `com.google.ai.edge.litert:litert`), and `litert-torch`, the renamed `ai-edge-torch`, is its PyTorch converter: a PyTorch model converted unmodified with `litert_torch.convert` matched the original to 4e-7 on a Galaxy S26 ([measured](https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/results/android/tinyhybridnet-three-runtimes.md), LiteRT 2.2.0, Android 16, 2026-09-05).

# FLUX.2-klein-4B — LiteRT (on-device text-to-image and image editing)

Black Forest Labs **[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**
(Apache-2.0) converted to **LiteRT `CompiledModel`** int8 graphs, generating **and
editing** images fully on a phone GPU. The upstream model card says klein "runs on consumer GPUs, with
as little as 13 GB VRAM". These graphs run it on a Pixel 8a's Mali-G715, which has no
dedicated VRAM at all.

![generated on a Pixel 8a Mali GPU](pixel8a_generated.png)

Prompt: *"a red apple on a wooden table, studio lighting"*. 4 steps, 256×256, generated
end-to-end on a Pixel 8a Mali GPU in 306 s. Matches the fp32 `diffusers` pipeline at
**PSNR 36.8 dB / corr 0.9987**.

![edited on a Pixel 8a Mali GPU](pixel8a_edited.png)

Editing: source, the fp32 `diffusers` edit, and the same edit on a Pixel 8a. Prompt:
*"turn the apple into a green apple"*. 4 steps, 256×256, **PSNR 44.3 dB / SSIM 0.9998**
against the fp32 pipeline, in 328–369 s.

klein is **step-wise distilled**, so the sampling loop is unusually plain: 4 steps, no
classifier-free guidance (one DiT pass per step, not two), no sign flip — just a
flow-matching Euler update `latents += dsigma[step] * noise_pred`.

## What's here

The pipeline (Qwen3-4B text encoder → rectified-flow DiT → VAE) is exported as
**INTEGER-int8** LiteRT graphs — int×int compute, the path the GPU delegate actually
runs; weight-only-FLOAT quantization hangs the GPU compile. The 4B DiT and the 4B
encoder each exceed both LiteRT's 2 GB flatbuffer load limit and a phone's GPU budget,
so they are **split into chunks that are resident one at a time**: peak footprint is a
single ~912 MB graph rather than the 6.2 GB total.

| Graph | Role | int8 size | I/O (256 px) |
|-------|------|-----------|--------------|
| `ke_enc0` / `ke_enc1` / `ke_enc2` | Qwen3-4B layers 1-9 / 10-18 / 19-27 | 912 MB each | `[1,512,2560]` → `[1,512,2560]` |
| `kc_prep` | image + context embedders, 3 modulation FCs | 166 MB | `img[1,256,128]`, `txt[1,512,7680]`, `temb[1,3072]` → hidden + 3 modulations |
| `kc_double0` / `kc_double1` | 3 + 2 double-stream blocks | 739 / 492 MB | `img[1,256,3072]`, `txt[1,512,3072]` → same |
| `kc_single0..3` | 5 single-stream blocks each (20 total) | 615 MB each | `joint[1,768,3072]` → `[1,768,3072]` |
| `kc_final` | adaLN-continuous norm + projection | 19 MB | `[1,768,3072]` → `[1,256,128]` |
| `kv_vae` | VAE decoder | 50 MB | `latent[1,32,32,32]` → `img[1,3,256,256]` |
| `kce_prep` / `kce_double*` / `kce_single*` / `kce_final` | the same eight graphs re-exported for editing | 166 / 739 / 492 / 615×4 / 19 MB | image tokens `[1,512,128]`, joint `[1,1024,3072]` |
| `kv_vae_enc` | VAE encoder (editing only) | 35 MB | `img[1,3,256,256]` → `latent[1,32,32,32]` |

The text encoder is included because klein conditions on Qwen3-4B **hidden states from
layers 9 / 18 / 27**, interleaved to 7680 channels — not on a pooled embedding, so there
is no smaller drop-in replacement.

Tensors are raw `float32`, little-endian, row-major. Tokenization, `embed_tokens`, the
causal+padding mask, both rotary tables, the scheduler and the two tail permutations run
on the host.

## Typing the prompt

The prompt is not baked in — only two of the staged tensors depend on the words
(`inputs_embeds` and `enc_mask`), so the app encodes a typed prompt itself. Stage the
`tokenizer/` folder here (`qwen_vocab.txt`, `qwen_merges.txt`, `qwen_special.txt` and the
778 MB `qwen_embed_fp16.bin`) and the app shows editable prompt fields.

It carries a faithful `Qwen2Tokenizer` port, fixture-tested byte-for-byte against Python
(`tokenizer_fixture.txt`), and looks token rows up in the memory-mapped fp16 embedding
table — a `GATHER` over 151936 rows is not a GPU op, and the gathered row is the graph's
input anyway. The fp16 table is within 3e-8 of the fp32 weights because the checkpoint is
bf16. Tokenizing and embedding a prompt takes about 1 s; a typed "a blue ceramic teapot on
a marble counter, morning light" generates the teapot, not the baked apple.


## Image editing

`Flux2KleinPipeline.__call__` takes `image` as its **first** argument: klein is natively
an editing model and text-to-image is the `image=None` case. Editing VAE-encodes the
reference and appends its latent tokens to the noise tokens before every step,

```python
latent_model_input = torch.cat([latents, image_latents], dim=1)
latent_image_ids   = torch.cat([latent_ids, image_latent_ids], dim=1)
```

with the reference separated from the noise on the rotary `T` axis (noise `T = 0`, the
i-th reference `T = 10 + 10·i`). Afterwards only the noise half is kept:
`noise_pred[:, :latents.size(1)]`.

At 256×256 that grows the image sequence 256 → 512 tokens, so the joint sequence goes
768 → 1024. **The weights do not change**, so `kce_*` are the same tensors re-exported at
the longer shape and every chunk is byte-identical in size to its `kc_*` twin. Measured
on a Pixel 8a: peak RSS **+2 %**, per-step GPU time **1.6×**, shader compile **1.0–1.2×**
— and since compilation dominates, editing costs about **+7 %** end-to-end.

The VAE encoder needed one rewrite: `DiagonalGaussianDistribution.mode()` chunks the
64-channel moments in half, which lowers to the banned `SPLIT`. The mode *is* the mean,
so slicing the first 32 channels directly is bit-exact and GPU-clean.

**Caveats.** 256×256, one reference image, and the prompt is baked into the staged host
tensors. More references or a different resolution means a re-export, not a redesign.
Report SSIM next to PSNR: the residual error is *sparse*. On a moonlit-snow edit the
worst 0.5 % of pixels carry 76 % of the squared error (specular speckles flipping between
near-white and dark blue), which drags PSNR to 27.9 dB while SSIM stays at 0.989 and the
image is perceptually identical to the fp32 reference.

## Two things the graphs assume

Both come from the GPU delegate (ML Drift), and neither is visible to a desktop op check.

1. **The attention mask is pre-expanded across heads**: pass `[1, 32, 512, 512]`, not
   `[1, 1, 512, 512]`. A broadcast `ADD` whose left operand is a `BATCH_MATMUL` result is
   silently miscomputed — the probabilities still sum to 1 and still honour the causal and
   padding masks, but the logits are wrong.
2. **Compute must be FP32**: `GpuOptions(precision = FP32)`. The modulated (adaLN) blocks
   overflow fp16 and return NaN.

Also: create one `Environment` and share it across every `CompiledModel` (a null
environment leaks the OpenCL context), and close every `TensorBuffer` after each run.

## Usage — Python (reproduces the exact device loop)

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel


def run(path, *inputs):
    """Runs one chunk, then releases it — sequential residency, as on device."""
    model = CompiledModel.from_file(path)
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    input_details = model.get_input_tensor_details(key)
    output_details = model.get_output_tensor_details(key)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for name, buffer, value in zip(signatures[key]["inputs"], input_buffers, inputs):
        buffer.write(np.ascontiguousarray(value, np.dtype(input_details[name]["dtype"])))
    model.run_by_index(0, input_buffers, output_buffers)
    outputs = []
    for name, buffer in zip(signatures[key]["outputs"], output_buffers):
        detail = output_details[name]
        flat = buffer.read(int(np.prod(detail["shape"])), np.dtype(detail["dtype"]))
        outputs.append(flat.reshape(detail["shape"]).copy())
    return outputs


# Host prep (tokenizer, embed_tokens, mask, rotary tables, sigmas) omitted — see below.
hidden, taps = inputs_embeds, []
for i in range(3):
    hidden = run(f"ke_enc{i}.tflite", hidden, mask, enc_cos, enc_sin)[0]
    taps.append(hidden)
prompt_embeds = np.stack(taps, 1).transpose(0, 2, 1, 3).reshape(1, 512, 7680)

for step in range(4):
    image, text, mod_img, mod_txt, mod_single = run(
        "kc_prep.tflite", latents, prompt_embeds, temb[step:step + 1])
    for i in range(2):
        image, text = run(f"kc_double{i}.tflite", image, text, cos, sin, mod_img, mod_txt)
    joint = np.concatenate([text, image], axis=1)
    for i in range(4):
        joint = run(f"kc_single{i}.tflite", joint, cos, sin, mod_single)[0]
    latents = latents + dsigma[step] * run("kc_final.tflite", joint, temb[step:step + 1])[0]

latent = unpatchify(unpack(latents) * bn_std + bn_mean)   # two pure permutations
image = run("kv_vae.tflite", latent)[0]                   # [1,3,256,256] in [-1,1]
```

## Usage — Kotlin (Android, LiteRT GPU)

```kotlin
val environment = Environment.create()                    // create once, share

fun gpu(name: String, inputs: List<FloatArray>): List<FloatArray> {
    val options = CompiledModel.Options(Accelerator.GPU)
    options.gpuOptions = CompiledModel.GpuOptions(
        precision = CompiledModel.GpuOptions.Precision.FP32)
    val model = CompiledModel.create(File(dir, name).absolutePath, options, environment)
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()
    inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
    model.run(inputBuffers, outputBuffers)
    val outputs = outputBuffers.map { it.readFloat() }
    inputBuffers.forEach { it.close() }
    outputBuffers.forEach { it.close() }
    model.close()                                          // one graph resident at a time
    return outputs
}

var hidden = inputsEmbeds
val taps = (0 until 3).map { gpu("ke_enc$it.tflite", listOf(hidden, mask, encCos, encSin))[0]
    .also { output -> hidden = output } }
// interleave the three taps -> [1, 512, 7680], then the 4-step DiT loop, then kv_vae
```

## Conversion

Quantization is `litert_torch` `full_dynamic_recipe(weight_dtype=INT8, granularity=CHANNELWISE)`.
The conversion scripts (`build_klein_enc.py`, `chunked_export_klein.py`,
`vae_deploy_klein.py`) and the host-prep / verification reference
(`gen_prep_klein.py`, `gen_verify_klein.py`) ship alongside the LiteRT sample app for
this model. Three rewrites are required for a GPU-clean graph, all exact:

- **RoPE without `GATHER_ND`** — bake the even/odd de-interleave into the rows of
  `to_q` / `to_k` and the fused `to_qkv_mlp_proj`, turning it into a contiguous
  half-split rotation. `q · k` is invariant to a permutation applied to both.
- **GQA `repeat_kv` as a `CONCATENATION`** — the stock `expand` is rank-5 *and* lowers
  to `BROADCAST_TO`, which the GPU delegate rejects outright.
- **Safe RMSNorm / LayerNorm** (max-normalized) and `ManualGroupNormND` in the VAE.

Note that the **desktop int8 path is a pessimistic proxy**: the same graphs score
36.4 dB through the host CPU int8 kernels and 44.1 dB on the device. Weights are never
redistributed here — the graphs are produced from the original Apache-2.0 checkpoint
with those scripts.

## Performance

Measured on a **Pixel 8a** (Tensor G3, Android 16) with the standard TFLite [`benchmark_model`](https://ai.google.dev/edge/litert/models/measurement) tool — 5 warm-up runs then 20 timed runs, reported as the tool's mean.

This pipeline ships 21 graphs; the 6 below are the representative ones — the shards that dominate a step plus the small head and tail graphs. The repeated shards (`kc_single1..3` and friends) are the same shape as the one measured here.

| Graph | Graph on GPU | GPU (OpenCL) | CPU (XNNPACK, 4 threads) |
|---|---|---|---|
| `kc_final.tflite` | 29 / 36 | 58 ms | 7 ms |
| `kc_prep.tflite` | 10 / 10 | 72 ms | 44 ms |
| `kv_vae.tflite` | 642 / 642 | **did not run** | 1815 ms |
| `ke_enc0.tflite` | 855 / 855 | **did not run** | 2663 ms |
| `kc_double0.tflite` | 706 / 706 | 813 ms | 2272 ms |
| `kc_single0.tflite` | 544 / 544 | 1860 ms | 3815 ms |

**These are the classic TFLite OpenCL delegate, not LiteRT's own accelerator.** The Kotlin usage above runs through LiteRT `CompiledModel`, which is a different GPU implementation; any figure quoted elsewhere on this card came from that path and is not comparable to this table. Read this table as a reproducible floor anyone can re-measure with a public tool.

Two graphs do not run on this delegate at all. `kv_vae` fails with `Failed to create Image2D from Buffer (clCreateImage): Invalid image size`, and `ke_enc0` with an OpenCL program build error; both then fall back to OpenGL, which rejects them with `Batch size mismatch, expected 1 but got 32`.

The small head and tail graphs are faster on the CPU here — `kc_final.tflite` (7 ms on CPU against 58 ms on GPU), `kc_prep.tflite` (44 ms on CPU against 72 ms on GPU) — so a host loop that sends every graph to the GPU is leaving time on the table.

## Snapdragon NPU (Hexagon)

This repo publishes 21 graphs and the sweep measured each one separately. On all 18 where both accelerators ran, the GPU is faster. The NPU loads faster on every one of them. Per-file rows are below.

- `ke_enc0.tflite` — the GPU runs it at 339.8 ms. The NPU does not — the graph compiles and then fails to run (`LiteRtException: Failed to invoke the compiled model`).
- `ke_enc1.tflite` — the GPU runs it at 340.0 ms. The NPU does not — the run reported no result line.
- `ke_enc2.tflite` — the GPU runs it at 339.2 ms. The NPU does not — the run reported no result line.

| file | backend | compiled | inference (median / min) | load |
|---|---|---|---:|---:|
| `kc_double0.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 492.5 ms / 451.5 ms | 1511 ms |
| `kc_double0.tflite` | GPU (Adreno) | — | 155.4 ms / 150.5 ms | 8426 ms |
| `kc_double1.tflite` | NPU (Hexagon v81) | on-device JIT | 300.4 ms / 294.3 ms | 1283 ms |
| `kc_double1.tflite` | GPU (Adreno) | — | 106.9 ms / 102.4 ms | 3341 ms |
| `kc_final.tflite` | NPU (Hexagon v81) | on-device JIT | 4.92 ms / 4.76 ms | 136 ms |
| `kc_final.tflite` | GPU (Adreno) | — | 1.49 ms / 1.31 ms | 603 ms |
| `kc_prep.tflite` | NPU (Hexagon v81) | on-device JIT | 20.67 ms / 20.21 ms | 375 ms |
| `kc_prep.tflite` | GPU (Adreno) | — | 12.64 ms / 10.10 ms | 2530 ms |
| `kc_single0.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 791.9 ms / 775.2 ms | 1147 ms |
| `kc_single0.tflite` | GPU (Adreno) | — | 255.3 ms / 249.9 ms | 4211 ms |
| `kc_single1.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 795.9 ms / 776.7 ms | 1390 ms |
| `kc_single1.tflite` | GPU (Adreno) | — | 250.0 ms / 243.8 ms | 4835 ms |
| `kc_single2.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 793.6 ms / 771.3 ms | 1163 ms |
| `kc_single2.tflite` | GPU (Adreno) | — | 311.2 ms / 245.9 ms | 4981 ms |
| `kc_single3.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 903.2 ms / 779.5 ms | 1328 ms |
| `kc_single3.tflite` | GPU (Adreno) | — | 250.9 ms / 245.1 ms | 6705 ms |
| `kce_double0.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 675.7 ms / 670.0 ms | 1507 ms |
| `kce_double0.tflite` | GPU (Adreno) | — | 222.8 ms / 214.4 ms | 4851 ms |
| `kce_double1.tflite` | NPU (Hexagon v81) | on-device JIT | 450.5 ms / 397.4 ms | 1465 ms |
| `kce_double1.tflite` | GPU (Adreno) | — | 151.0 ms / 144.6 ms | 3364 ms |
| `kce_final.tflite` | NPU (Hexagon v81) | on-device JIT | 9.96 ms / 9.85 ms | 137 ms |
| `kce_final.tflite` | GPU (Adreno) | — | 2.94 ms / 2.42 ms | 597 ms |
| `kce_prep.tflite` | NPU (Hexagon v81) | on-device JIT | 22.41 ms / 21.07 ms | 380 ms |
| `kce_prep.tflite` | GPU (Adreno) | — | 16.27 ms / 10.64 ms | 2511 ms |
| `kce_single0.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 1627.6 ms / 1535.0 ms | 1379 ms |
| `kce_single0.tflite` | GPU (Adreno) | — | 419.8 ms / 404.0 ms | 4494 ms |
| `kce_single1.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 1614.4 ms / 1597.7 ms | 1421 ms |
| `kce_single1.tflite` | GPU (Adreno) | — | 421.0 ms / 407.7 ms | 4409 ms |
| `kce_single2.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 1606.3 ms / 1583.5 ms | 1348 ms |
| `kce_single2.tflite` | GPU (Adreno) | — | 413.8 ms / 399.5 ms | 4291 ms |
| `kce_single3.tflite` | NPU (Hexagon v81) | AOT (SM8850) | 1638.3 ms / 1548.8 ms | 1302 ms |
| `kce_single3.tflite` | GPU (Adreno) | — | 433.9 ms / 411.9 ms | 4393 ms |
| `ke_enc0.tflite` | GPU (Adreno) | — | 339.8 ms / 328.1 ms | 5798 ms |
| `ke_enc1.tflite` | GPU (Adreno) | — | 340.0 ms / 328.7 ms | 5724 ms |
| `ke_enc2.tflite` | GPU (Adreno) | — | 339.2 ms / 332.7 ms | 5716 ms |
| `kv_vae.tflite` | NPU (Hexagon v81) | on-device JIT | 2263.7 ms / 2010.9 ms | 492 ms |
| `kv_vae.tflite` | GPU (Adreno) | — | 230.4 ms / 225.5 ms | 2520 ms |
| `kv_vae_enc.tflite` | NPU (Hexagon v81) | on-device JIT | 959.4 ms / 901.0 ms | 309 ms |
| `kv_vae_enc.tflite` | GPU (Adreno) | — | 117.4 ms / 116.3 ms | 1824 ms |

Measured on a **Samsung Galaxy S26** (Snapdragon 8 Elite Gen 5 / SM8850, Hexagon v81, Android 16) with LiteRT `CompiledModel` 2.2.0, one accelerator per process, 5 warm-up runs then N=50 timed runs, median reported. Every run held thermal status `NONE` throughout. Headroom 0.58–0.83, where 1.0 is the throttling threshold.

**The NPU rows marked *on-device JIT* ran the published file unchanged.** LiteRT compiled it for the Hexagon on the device at first load. Those first compiles took 664 ms to 107 s here. The `load` column above is the cached load every later run pays. Recipe and the runtime libraries it needs: [NPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md).

The NPU rows marked *AOT* ran an artifact compiled ahead of time for SM8850 (ai-edge-litert 2.2.0 + QAIRT 2.47.0), not the published file. That artifact is not distributed here; the compile is one command in the [NPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md).

GPU wiring: [GPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md).

## License

Apache-2.0, inherited from `black-forest-labs/FLUX.2-klein-4B` (weights and text encoder).
