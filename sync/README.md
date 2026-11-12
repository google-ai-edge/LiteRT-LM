# LiteRT-LM ModelSync

`ModelSync` is a high-performance C++ library and command-line utility for
**section-level differential synchronization** of LiteRT-LM (`.litertlm`)
container files.

Instead of re-downloading multi-gigabyte models when small components change
(such as prompt metadata, chat templates, or vision adapters), `ModelSync` reads
only the remote FlatBuffer header (~8 KB) via HTTP Range requests, identifies
which section SHA-256 hashes differ, and fetches only the modified byte ranges,
splicing them with existing local weights on disk.

---

## Building and Running the CLI

You can run the CLI directly using Bazel:

```bash
bazel run //sync:model_sync_cli -- --help
```

Or build the binary and run it directly:

```bash
bazel build //sync:model_sync_cli

# Binary location:
./bazel-bin/sync/model_sync_cli --help
```

---

## CLI Flags Reference

<!-- disableFinding(LINE_OVER_80) -->

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--remote` | `string` | `""` *(required)* | Path or URL of the upstream remote `.litertlm` file (supports byte-range fetching). |
| `--local` | `string` | `""` | Path to the existing local `.litertlm` file on disk to compare against and reuse unchanged sections from. |
| `--output` | `string` | `""` | Path where the synced model will be written. If omitted, updates `--local` **in-place**. |
| `--diff` | `bool` | `false` | **Dry-run inspection mode:** compares local and remote models and prints the section diff plan without downloading any payloads. |
| `--clean` | `bool` | `false` | Deletes the local model upfront to reclaim disk space before downloading, reducing peak disk overhead from 2× to 1× on storage-constrained devices. |
| `--threshold` | `double` | `0.80` | Change ratio threshold (0.0 to 1.0). If more than this fraction of bytes changed, automatically falls back to a direct full sequential download. |
| `--check_disk` | `bool` | `true` | Validates available disk space before starting the download to prevent mid-transfer out-of-disk failures. |

<!-- enableFinding(LINE_OVER_80) -->

---

## Examples for Each Flag

### 1. `--remote` (Upstream Source)

Specifies the remote model file or HTTP/HTTPS URL to synchronize against.
*(Required)*

```bash
# Sync against a remote Hugging Face model URL:
bazel run //sync:model_sync_cli -- --diff \
  --local=/tmp/local_model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

---

### 2. `--local` (Local Model Cache)
Specifies the existing `.litertlm` model file on disk. Unchanged sections (like
heavy weights) are reused directly from this file:

```bash
# Compare and reuse sections from local_model.litertlm:
bazel run //sync:model_sync_cli -- \
  --local=/data/models/local_model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

---

### 3. `--output` (Target Output Destination)
By default, omitting `--output` performs an **in-place update** on `--local`.
Providing `--output` preserves the existing local file and writes the new model
to a separate path:

```bash
# Write the spliced model to a new file without touching the old one:
bazel run //sync:model_sync_cli -- \
  --local=/data/models/v1.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm \
  --output=/data/models/v2_synced.litertlm
```

---

### 4. `--diff` (Dry-Run Section Comparison)
Compares local and remote models and prints the diff plan without downloading
any payloads:

```bash
# Check what sections changed without performing any downloads:
bazel run //sync:model_sync_cli -- \
  --diff \
  --local=/data/models/local_model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

**Example output:**

```text
=== LiteRT-LM Sync Plan ===
Status: UPDATE REQUIRED
Mode:   DIFFERENTIAL SLICE UPDATE
Total Remote Size: 2583082648 bytes (2463.42 MB)
Bytes to Reuse:    2578236383 bytes (2458.80 MB)
Bytes to Download: 4730240 bytes (4.51 MB)

Section Comparison:
  [0] data_type=LlmMetadataProto, model_type="", size=12190 bytes (REUSE LOCAL)
  [1] data_type=SP_Tokenizer, model_type="", size=4689013 bytes (REUSE LOCAL)
  ...
  [8] data_type=TFLiteModel, model_type="tf_lite_vision_adapter", size=4727112 bytes
      Action: DOWNLOAD DIFF (Local: dd590b96..., Remote: ce882453...)
      Range:  [1715716096, 1720443207]
```

---

### 5. `--clean` (Constrained Storage Mode)
Under standard differential sync, a temporary file is assembled before replacing
the original, which briefly requires disk space for both files (~2× model size).
Passing `--clean` deletes the local copy upfront when disk space is constrained
or when a full download is required, reducing peak disk overhead from 2× to 1×:

```bash
# Cleanly remove existing model before download on low-storage devices:
bazel run //sync:model_sync_cli -- \
  --clean \
  --local=/data/models/model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

---

### 6. `--threshold` (Differential vs. Full Fallback Threshold)
Defines the fraction of changed bytes (between `0.0` and `1.0`) at which
differential slicing is abandoned in favor of a direct sequential full download.
The default is `0.80` (80%):

```bash
# Fall back to full download if more than 50% of the model changed:
bazel run //sync:model_sync_cli -- \
  --threshold=0.50 \
  --local=/data/models/model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

---

### 7. `--check_disk` (Pre-Flight Disk Space Validation)
By default (`--check_disk=true`), the tool queries available filesystem capacity
and halts before downloading if disk space is insufficient. You can disable this
check in specialized environments (e.g. containers, RAM disks, or custom virtual
filesystems):

```bash
# Bypass pre-flight disk space validation:
bazel run //sync:model_sync_cli -- \
  --check_disk=false \
  --local=/data/models/model.litertlm \
  --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
```

---

## C++ Library API (`ModelSync`)

Applications can integrate `ModelSync` directly:

```cpp
#include "sync/model_sync.h"

using ::litert::lm::sync::CreateHttpRangeFetcher;
using ::litert::lm::sync::ModelSync;
using ::litert::lm::sync::SyncOptions;

// 1. Create an HTTP RangeFetcher (or provide a custom RangeFetcher callback)
auto fetcher = CreateHttpRangeFetcher(
    "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/"
    "resolve/main/gemma-4-E2B-it.litertlm");

// 2. Plan sync: inspect remote header and determine byte ranges
SyncOptions options;
options.verify_checksums = true;
options.full_download_threshold = 0.80;

auto plan_or = ModelSync::PlanSync("/path/to/local.litertlm", fetcher, options);
if (!plan_or.ok()) {
  // Handle planning error
}

const auto& plan = *plan_or;
if (plan.is_up_to_date) {
  // Model is already latest version
}

// 3. Execute differential update
absl::Status status = ModelSync::Update(
    "/path/to/local.litertlm",
    "/path/to/output.litertlm",
    fetcher,
    options);
```
