# Audio LoRA Converter

This document describes the PEFT Audio LoRA conversion path for LiteRT-LM.

LiteRT-LM can represent a LoRA sidecar as a small TFLite flatbuffer. It
contains tensors whose names match LoRA input tensors in the target graph, plus
a `lora_rank` metadata entry. This converter compiles PEFT audio LoRA tensors
into that sidecar format.

The runtime and graph must still provide an audio LoRA binding contract. In
practice, this means the target audio graph must expose official LoRA input
names, and the runtime path must attach the generated sidecar through the
audio-scoped session slot, not the language-model LoRA slot.

## Input

```text
adapter/
  adapter_config.json
  adapter_model.safetensors

audio_encoder_lora_ready.tflite
```

The converter recognizes common PEFT attention keys:

```text
...layers.<n>.self_attn.q_proj.lora_A.weight
...layers.<n>.self_attn.q_proj.lora_B.weight
...layers.<n>.self_attn.k_proj.lora_A.weight
...
```

They are mapped to LiteRT-LM LoRA tensor names:

```text
lora_audio_attn_q_a_weight_<n>
lora_audio_attn_q_b_weight_<n>
lora_audio_attn_k_a_weight_<n>
...
lora_audio_ff1_l1_a_weight_<n>
lora_audio_lconv_end_b_weight_<n>
```

## Command

If the base model is a `.litertlm` bundle, first dump its sections and use the
audio encoder TFLite section as the target graph:

```bash
litert-lm-peek --litertlm_file=/path/to/model.litertlm \
  --dump_files_dir=/tmp/litertlm_sections
```

The dumped TFLite file whose model type is `tf_lite_audio_encoder_hw` is the
`--target_tflite` argument below.

```bash
bazel run //python/litert_lm_builder:audio_lora_converter_cli -- \
  --peft_lora_dir=/path/to/adapter \
  --target_tflite=/path/to/audio_encoder_lora_ready.tflite \
  --output_lora_tflite=/path/to/audio_lora.tflite \
  --report_json=/path/to/audio_lora_report.json
```

For CI or release builds, keep the strict defaults:

```bash
litert-lm-audio-lora-converter \
  --peft_lora_dir=/path/to/adapter \
  --target_tflite=/path/to/audio_encoder_lora_ready.tflite \
  --output_lora_tflite=/path/to/audio_lora.tflite
```

The CLI is strict by default. It fails on missing target inputs, unmatched PEFT
keys, PEFT tensors absent from the target graph, and unconsumed target inputs.
The `--allow_*` flags are intended only for graph bring-up and debugging.

`--target_tflite` is optional but recommended for production conversion.
When provided, the converter extracts LoRA tensor names, shapes, dtypes, and
byte sizes from the target graph inputs. The target graph must expose LiteRT-LM
LoRA input names accepted by `runtime/util/lora_data.cc`, such as
`lora_audio_attn_q_a_weight_0`. If the target graph exposes no official LoRA
inputs, conversion stops before writing an artifact. Passing an existing LoRA
sidecar as `--target_tflite` also fails because sidecars contain payload tensors,
not graph inputs.

For release conversion, the unconsumed-target-inputs gate is important: it
rejects graphs that list LoRA tensors as signature inputs but never feed those
tensors into an operator. Such a graph can appear LoRA-ready while silently
dropping part of the adapter.

The PEFT tensor orientation is chosen by exact target-shape matching: original
orientation first, transpose second. If neither matches, conversion stops.

## Current Scope

This first converter supports the audio tower LoRA tensors exposed by the
LiteRT-LM LoRA naming contract:

```text
audio_tower q_proj / k_proj / v_proj / o_proj / post -> lora_audio_attn_{q,k,v,o}_{a,b}_weight_<n>
feed_forward1.ffw_layer_1.linear -> lora_audio_ff1_l1_{a,b}_weight_<n>
feed_forward1.ffw_layer_2.linear -> lora_audio_ff1_l2_{a,b}_weight_<n>
feed_forward2.ffw_layer_1.linear -> lora_audio_ff2_l1_{a,b}_weight_<n>
feed_forward2.ffw_layer_2.linear -> lora_audio_ff2_l2_{a,b}_weight_<n>
lconv1d.linear_start.linear -> lora_audio_lconv_start_{a,b}_weight_<n>
lconv1d.linear_end.linear -> lora_audio_lconv_end_{a,b}_weight_<n>
audio_tower.output_proj -> lora_audio_output_proj_{a,b}_weight_0
```

The converter is audio-scoped. It does not compile language-model LoRA keys
into this audio sidecar. Mixed PEFT adapters should use `--peft_key_regex` to
select the audio tensors intended for the audio sidecar; otherwise strict mode
reports the language-model keys as unmatched.

The target graph still controls what is actually bound. If the PEFT adapter has
more tensors than the target graph exposes, those keys are reported as
`unused_peft_keys`. Strict CLI conversion fails on these by default because all
recognized PEFT tensors should be represented in the target graph for a release
artifact.

This converter does not rewrite the model graph. It only compiles PEFT tensors
into the LiteRT-LM LoRA sidecar format expected by a graph that is already
LoRA-ready. A target graph that is made LoRA-ready by a separate exporter or
graph transformation must pass zero-sidecar identity on the real runtime before
CER or downstream task metrics are meaningful.

## PEFT Scaling

PEFT applies LoRA as:

```text
y = base(x) + (alpha / rank) * B(A(x))
```

The LiteRT-LM runtime sidecar format does not carry a separate `alpha` value.
The converter therefore folds `alpha / rank` into each `lora_B` tensor. `rank`
and `alpha` are read from `adapter_config.json`, or may be overridden with
`--lora_rank` and `--lora_alpha`.

## Output

```text
audio_lora.tflite
audio_lora_report.json
```

The generated TFLite file stores LoRA payloads in external buffer ranges inside
the same file, using the TFLite Buffer `offset` and `size` fields. This is the
format read by `runtime/util/lora_data.cc`. Because LiteRT-LM currently reads
the LoRA rank from the `lora_rank` metadata entry's `buffer` field, the
converter also pads the buffer table so that this metadata buffer index is valid
for standard TFLite flatbuffer verification.

The converter uses the FlatBuffers Python runtime plus a small private TFLite
reader/writer for only the schema fields needed here. This avoids adding a PyPI
`tflite` dependency while keeping the generated artifact compatible with
LiteRT-LM's C++ `LoraData` reader.

The file tail is padded to a mmap-safe boundary. LiteRT-LM reads each payload by
mapping an aligned region around the requested `offset` and `size`, so the
physical file must extend far enough for that aligned mapping even when the last
payload itself is small.

At runtime, an audio LoRA-ready path is expected to attach the generated file
through the audio-scoped session slot:

```cpp
SessionConfig session_config;
session_config.SetAudioScopedLoraFile(audio_lora_file);
```

The audio executor should then load the sidecar through its `LoraManager`, match
sidecar tensors by input name, and zero-fill graph LoRA inputs that are absent
from the sidecar. This converter only creates the sidecar; it does not rewrite
the graph or prove runtime binding.

## Validation Gates

Use the converter in this order:

1. Zero sidecar: no output change.
2. Random sidecar: output changes and binding is observable.
3. Target contract: the graph exposes official LoRA input names.
4. Consumption contract: every exposed LoRA input is consumed by at least one op.
5. PEFT sidecar: all expected tensors are present and byte sizes match.
6. Device smoke: scoped audio LoRA loads on device.
7. Task metric: run CER or downstream quality only after the binding gates pass.

## Target Graph Requirements

The converter is intentionally strict because a target graph can expose LoRA
inputs without using them. That state is worse than a hard error: the sidecar
appears valid, but part of the adapter is silently ignored.

For Gemma audio encoder graphs, the graph rewriter must wire every LoRA input
into the actual audio branch:

```text
source -> BMM(lora_A) -> BMM(lora_B) -> optional view chain -> ADD at float landing
```

Some modules are direct:

```text
q_proj / o_proj / FFN / lconv_end:
  projection output and float landing have the same element count
```

Other modules require cloning the same view chain used by the base graph before
the LoRA delta can be added:

```text
k_proj / v_proj:
  projection output -> reshape/pad/slice/concat/transpose -> dequant landing

lconv_start:
  projection output -> multiple slice branches -> multiple dequant landings
```

A graph that simply lists `k_proj`, `v_proj`, or `lconv_start` LoRA tensors as
signature inputs but skips those cloned view-chain branches must fail the
unconsumed-target-inputs gate.

Runtime binding must also distinguish language LoRA from audio LoRA. Audio
sidecars should be attached through the audio-scoped session slot, then loaded
into the audio executor's `LoraManager` for the `serving_default` signature.
Reusing the language-scoped LoRA slot for audio makes the runtime try to load
audio tensors into the language executor and can produce misleading regressions.

## Runtime Gate

Host TFLite algebra checks are necessary but not sufficient. A release path
should also run a device or runtime-equivalent zero-sidecar gate:

```text
base .litertlm + no audio LoRA sidecar
LoRA-ready .litertlm + zero audio LoRA sidecar
```

The outputs should match within the model family's normal numerical tolerance.
Only after this gate passes should the generated PEFT sidecar be evaluated on a
task metric such as CER.
