---
license: apache-2.0
base_model: prism-ml/bonsai-image-ternary-4B-unpacked
pipeline_tag: text-to-image
tags:
- litert
- tflite
- on-device
- ternary
- flux2
base_model_relation: quantized
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

LiteRT is Google's on-device runtime, the new name for TensorFlow Lite (Android: `com.google.ai.edge.litert:litert`), and `litert-torch`, the renamed `ai-edge-torch`, is its PyTorch converter: a PyTorch model converted unmodified with `litert_torch.convert` matched the original to 4e-7 on a Galaxy S26 ([measured](https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/results/android/tinyhybridnet-three-runtimes.md), LiteRT 2.2.0, Android 16, 2026-09-05).

# Bonsai Image 4B — LiteRT (.tflite)

[PrismML Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked) — a ternary-weight diffusion transformer built on the FLUX.2-klein-4B architecture — converted to LiteRT for on-device text-to-image. The whole pipeline runs in three fixed-shape `.tflite` graphs; the host does only tokenization, the FlowMatch-Euler loop, and latent unpatchify (`generate.py` in this repo, ~150 lines, no torch and no diffusers).

| graph | file | recipe | size |
|---|---|---|---|
| DiT (Flux2Transformer2DModel, 3.88 B) | `dit_int4b32.tflite` | int4 block-32 | 2.11 GiB |
| DiT — GPU-shaped export (Apple GPU / Metal) | `dit_gpu_int4b32.tflite` | int4 block-32, same weights | 2.11 GiB |
| DiT — 256×256 export (16×16 patch grid, 256 image tokens) | `dit_256_int4b32.tflite` | int4 block-32, same weights | 2.11 GiB |
| text encoder (Qwen3-4B, top 9 layers pruned) | `textenc_int4.tflite` | int4 block-128 DRQ | 1.68 GiB |
| text encoder — higher-fidelity variant | `textenc_int8_weightonly.tflite` | int8 weight-only | 2.91 GiB |
| VAE decoder (AutoencoderKLFlux2) | `vae_dec_fp32.tflite` | fp32 | 0.19 GiB |
| VAE decoder — 256×256 export (32×32 latent) | `vae_dec_256_fp32.tflite` | fp32 | 0.19 GiB |

Smallest working set: **3.97 GiB** (int4 text encoder). Output is 512×512 with the default pair of graphs, or 256×256 with the `dit_256` / `vae_dec_256` pair (`generate.py --size 256`; `variants.256` in `pipeline_meta.json`) — the text encoder, tokenizer and latent constants are shared by both sizes; 256 prompt tokens and 4 sampling steps by default (the model is step-distilled; more steps also work).

## The ternary weights survive int4 exactly

The vendor ships the DiT with ternary weights (every block linear uses values {−scale, 0, +scale} per 128-group). In the int4 block-32 container those land as exactly **{−7, 0, +7}** — zero rounding decisions, verified in the converted artifact. The int4 file is a lossless container for this model's weights, at 2.11 GiB versus 14.4 GiB fp32.

## Quality

Converting without quantizing anything is essentially exact (76.8 dB vs the PyTorch pipeline). With quantization, low PSNR against the PyTorch sample is **sampler drift, not degradation**: a small weight perturbation amplified through 4 sampling steps lands on a different draw from the same distribution. Across a 6-prompt stress set (legible text, a face, macro texture, a smooth gradient, a cluttered scene), reference-free sharpness stays flat across every recipe (Laplacian variance 1086 → 1069 at the lowest-PSNR variant), text stays legible, and faces stay correct in all columns. Pick a text encoder by size, not by fear: int4 if you want the smallest set, int8 weight-only if you want to track the reference sample more closely.

![quality grid](quality_grid.png)

## Run it

```bash
pip install ai-edge-litert numpy pillow transformers jinja2 huggingface_hub
python generate.py --model-dir . --prompt "a red fox sitting in fresh snow at sunrise" --seed 42 --out fox.png
python generate.py --model-dir . --size 256 --prompt "a red fox sitting in fresh snow at sunrise" --seed 42 --out fox_256.png
```

The same sample is maintained in Google's [litert-samples](https://github.com/google-ai-edge/litert-samples/tree/main/models/bonsai/bonsai_image_4b) repository (`models/bonsai/bonsai_image_4b`).

![example](fox.png)

## Performance

All rows are for one 512×512 image at the default 4 sampling steps with a 256-token prompt. The pipeline is three graphs — text encoder once, DiT once per step, VAE decoder once — so the DiT column dominates.

| Device | Backend | Text encoder | DiT (per step) | VAE | Total per image | Peak memory |
|---|---|---|---|---|---|---|
| Apple-silicon Mac | CPU, 8 threads | 2.6 s | 3.9 s | 1.3 s | ~19 s | — |
| Apple-silicon Mac | GPU (Metal, fp32) — DiT only | on CPU | **0.74 s** | on CPU | **~6 s** steady state | ~22 GB |
| iPhone 17 Pro | CPU/XNNPACK, 6 threads | 1.8 s | 13 s | 3.1 s | ~64 s | ~2.9 GiB |
| Pixel 8a (8 GB) | CPU | — | — | — | ~7–8 min | — |

Conditions worth knowing before you read those numbers:

- The Metal row uses `dit_gpu_int4b32.tflite` and needs fp32 GPU precision (`gpu_options` TOML `precision = 2`) — the default fp16 overflows this DiT's activation range. It also pays a one-time ~40 s Metal compile per launch, which the steady-state figure excludes, and its fp32-resident weights are what put memory near 22 GB, so a 32 GB+ Mac is the practical target. The CPU-shaped `dit_int4b32.tflite` does not run on the Metal accelerator at all (rank-5 rotary tensors).
- The iPhone DiT loads and prepares in ~4.5 s before the first step. Attach the XNNPACK delegate explicitly, with your thread count, when using the C API — without it the runtime falls back to reference kernels, which are orders of magnitude slower on this model.
- Device output is bit-exact against the Mac run: every DiT step matches to the last float, and the final PNG scores 51.2 dB PSNR.
- The full-app measurements land in the same place: ~62 s per image on an iPhone 17 Pro. Treat 8 GB of RAM as the floor and 12 GB+ as the practical target on Android.


**On-device (iPhone 17 Pro, CPU via XNNPACK, 6 threads):** the 2.11 GiB DiT loads and prepares in ~4.5 s; 13 s per DiT step, text encoder 1.8 s, VAE 3.1 s — **~64 s per 512×512 image** at 4 steps, ~2.9 GiB peak memory. Device output is bit-exact against the Mac run (every DiT step matches to the last float; 51.2 dB PSNR on the final PNG). One integration note: attach the XNNPACK delegate explicitly (with your thread count) when using the C API — without it the runtime falls back to reference kernels, which are orders of magnitude slower on this model.

### 256×256

The 256×256 pair is the same weights traced at 256 image tokens (16×16 patch grid) and a 32×32 latent, so it changes the time per image, not the memory: the DiT file is the same 2.11 GiB, and its peak memory on a Galaxy S26 is the same 4.1 GB at both sizes. Per-graph times on the S26 (CPU with XNNPACK, 6 threads, one call per graph with LiteRT's `benchmark_model`, the binary behind `litert benchmark --android --cpu`, 3 runs):

| Graph, one call | 512×512 | 256×256 |
|---|---|---|
| text encoder (shared) | 1.8 s | 1.8 s |
| DiT, per step | 14.6 s | 4.3 s |
| VAE decoder | 4.9 s | 1.1 s |
| sum for one image, 4 steps | ~65 s | ~20 s |

The sum is the sum of per-graph benchmarks, not an app run. On an Apple-silicon Mac (CPU, 8 threads, same session) the DiT step goes 5.5 s → 2.6 s and the VAE decode 1.0 s → 0.24 s. Fidelity at 256: with the same prompt embeds and the same initial noise, the all-tflite image scores 39.2 dB PSNR against the torch pipeline at 256×256 (Laplacian variance 873 vs 865, the same sharpness). The six prompts of the quality grid at both sizes, 256 shown at 2×:

![512 vs 256](grid_512_vs_256.png)

### Measured in an app on three Galaxy phones

[@mekabe](https://huggingface.co/mekabe) timed both sizes in their own Android app and posted the results in [discussion #2](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/discussions/2). Seconds per stage at 2 sampling steps (the rows above use 4), with the app's own language model released from memory before the pipeline starts; the total is both loads, the text encoder, two DiT steps and the VAE decoder.

| Device | Size | Text encoder | Text encoder load | DiT load | DiT, per step | VAE decoder | Total, 2 steps |
|---|---|---|---|---|---|---|---|
| Galaxy S26 Ultra | 256×256 | 2.90 | 1.60 | 2.80 | 9.55 | 2.80 | 29.20 |
| Galaxy S22 Ultra | 256×256 | 8.30 | 3.20 | 6.70 | 34.15 | 16.60 | 103.10 |
| Galaxy Z Flip 6 | 256×256 | 5.00 | 2.10 | 4.30 | 17.85 | 9.10 | 56.20 |
| Galaxy S26 Ultra | 512×512 | 2.80 | 1.70 | 2.70 | 25.80 | 5.80 | 64.60 |
| Galaxy S22 Ultra | 512×512 | 7.10 | 2.90 | 6.60 | 64.25 | 55.20 | 200.30 |
| Galaxy Z Flip 6 | 512×512 | 5.10 | 2.20 | 4.80 | 50.70 | 46.10 | 159.60 |

## Snapdragon NPU (Hexagon)

- `dit_gpu_int4b32.tflite` — neither accelerator produced a usable row on the S26. NPU: the ahead-of-time compile for SM8850 failed on the host. GPU: the benchmark process was killed, most likely out of memory.
- `dit_int4b32.tflite` — neither accelerator produced a usable row on the S26. NPU: the ahead-of-time compile for SM8850 failed on the host. GPU: the benchmark process was killed, most likely out of memory.
- `textenc_int4.tflite` — neither accelerator produced a usable row on the S26. NPU: the ahead-of-time compile for SM8850 failed on the host. GPU: the benchmark process was killed, most likely out of memory.
- `textenc_int8_weightonly.tflite` — neither accelerator produced a usable row on the S26. NPU: the ahead-of-time compile for SM8850 failed on the host. GPU: the benchmark process was killed, most likely out of memory.
- `vae_dec_fp32.tflite` — neither accelerator produced a usable row on the S26. NPU: the graph compiles and then fails to run (`LiteRtException: Failed to invoke the compiled model`). GPU: `LiteRtException: Failed to compile model`.

Target: **Samsung Galaxy S26** (Snapdragon 8 Elite Gen 5 / SM8850, Hexagon v81, Android 16).

GPU wiring: [GPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md). NPU recipe: [NPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md).

## Run it on-device (Android / iOS)

Sample app (Android): [`android/`](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/tree/main/android) — type a prompt, tap Generate; Galaxy S26, CPU (LiteRT `CompiledModel`, XNNPACK, 6 threads), 256×256 in 23.5 s (4 steps, the first run on a cool phone), measured by the app's device check on 2026-10-09. Its README has the five lines that run it and the failure paths it was checked on.

Complete single-screen apps for both platforms — including Swift/Kotlin ports of the Qwen3 tokenizer (token-exact against the Python tokenizer), the FlowMatch-Euler loop, and the latent unpatchify — are at [hf-to-litertlm/bonsai_image_work/device](https://github.com/john-rocky/hf-to-litertlm/tree/main/bonsai_image_work/device) (`BonsaiApp` for iOS, `BonsaiAppAndroid` for Android). Measured: iPhone 17 Pro ~62 s per image; Pixel 8a (8 GB RAM) completes at ~7–8 min per image, so treat 8 GB as the floor and 12 GB+ as the practical target. The iOS app runs the 512×512 pair; the Android app under `android/` runs both sizes (256×256 in 23.5 s and 512×512 in 66.2 s on a Galaxy S26, CPU).

## Run it on the Apple GPU (macOS)

`dit_gpu_int4b32.tflite` is the same DiT re-exported GPU-clean — rope tables precomputed with the pipeline's constant position ids and the interleaved rotation rewritten rank-4, same weights, verified numerically identical at export. On an Apple-Silicon Mac the LiteRT Metal accelerator runs it at ~0.74 s/DiT-step, ~6 s per 512×512 image steady-state (one-time ~40 s Metal compile per launch). Force fp32 GPU precision (`gpu_options` TOML `precision = 2`) — default fp16 overflows this DiT's activation range. A complete macOS SwiftUI app (DiT on Metal, text encoder + VAE on CPU) is `BonsaiAppMac` in the same device directory linked above; the fp32-resident GPU weights put steady-state memory around 22 GB, so a 32 GB+ Mac is recommended. The CPU-shaped `dit_int4b32.tflite` does not run on the Metal accelerator (rank-5 rotary tensors), so pick the file that matches your target.

Load the graphs through the [LiteRT](https://github.com/google-ai-edge/litert) CompiledModel API, on the CPU with XNNPACK, as the Python host loop here and the iOS and macOS apps do; the litert-samples page above carries the three rules for a port: map inputs by argument position (`args_<n>`), never by shape; load one graph at a time, freeing it before the next; keep the graphs on XNNPACK. [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and the Gallery app load `.litertlm` bundles, not these graphs. The graph contract:

1. text encoder: `args_0` ids (1, 256) int32, `args_1` mask (1, 256) int32 → `output_0` embeds (1, 256, 7680) float32
2. DiT, once per step: `args_0` latents (1, 1024, 128), `args_1` embeds, `args_2` sigma (1,), `args_3` img_ids (1024, 4), `args_4` txt_ids (256, 4), all float32 → velocity (1, 1024, 128); `lat += (sigma[k+1] - sigma[k]) * velocity`
3. VAE decoder: `args_0` z (1, 32, 64, 64) float32 → image (1, 3, 512, 512) float32, `clamp(y / 2 + 0.5)`

At 256×256 the DiT takes latents (1, 256, 128) and `img_ids` (256, 4), and the VAE decoder takes z (1, 32, 32, 32) → image (1, 3, 256, 256). `img_ids` and `txt_ids` then share a shape, which is why inputs are mapped by argument position. On the host, the position grid is 16×16, the unpatchify target is (1, 32, 32, 32), and the sigma schedule's `mu` is a function of the image-token count (`flowmatch_sigmas` in `generate.py`).

## Conversion

Converted with [litert-torch] via the recipe published at [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) (export scripts, quantization, and the two conversion gotchas: Flux2's float64 RoPE frequency table must be forced to float32 to legalize `tfl.pow`, and blockwise zero-scales in all-zero blocks need patching for XNNPACK). The text encoder is exported as a prompt embedder — the pipeline reads hidden states from layers (9, 18, 27) only, so the top 9 of 36 layers and the LM head are pruned for free.

## License

Apache-2.0, following the upstream [prism-ml/bonsai-image-ternary-4B-unpacked](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked) release. PrismML's NOTICE, carried here as `NOTICE.md`, credits FLUX.2 [klein] 4B (Black Forest Labs) and Qwen3-4B (Alibaba Cloud), both Apache-2.0, and asks for a credit such as "Created using Bonsai Image by Prism ML." when you deploy the model publicly.
