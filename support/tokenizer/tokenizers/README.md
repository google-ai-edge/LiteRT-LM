# HuggingFace `tokenizers` — C++ port

This directory contains a line-by-line C++ port of the *inference* path of the
[HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) Rust
crate, version **0.23.2** (crate `tokenizers`, Apache License 2.0, Copyright
2020 The HuggingFace Team; see [LICENSE](LICENSE)). It backs
`litert::support::HuggingFaceTokenizer`
(`support/tokenizer/huggingface_tokenizer.cc`) and replaces the previous Rust
implementation reached through `tokenizers-cpp`, so that LiteRT-LM no longer
needs a Rust toolchain to load `tokenizer.json` files.

The port keeps the crate's module layout, type and function names (Google C++
spelling), control flow, comments and error messages, so that it can be diffed
against the Rust sources when the crate is updated. The entry point is
`tokenizers::Tokenizer` in [`tokenizer/tokenizer.h`](tokenizer/tokenizer.h); the
umbrella header [`tokenizers.h`](tokenizers.h) mirrors the crate's `lib.rs`
re-exports.

## Layout

| Directory         | Rust module           | Contents                                                      |
| ----------------- | --------------------- | ------------------------------------------------------------- |
| `tokenizer/`      | `src/tokenizer/`      | `Normalizer`/`PreTokenizer`/`Model`/`PostProcessor`/`Decoder` |
:                   :                       : interfaces (`api.h`), `Encoding`, `NormalizedString`,         :
:                   :                       : `PreTokenizedString`, `Pattern`, `AddedVocabulary`,           :
:                   :                       : padding/truncation, `Tokenizer` + JSON (de)serialization      :
| `models/`         | `src/models/`         | `BPE`, `WordPiece`, `WordLevel`, `Unigram` and `ModelWrapper` |
| `normalizers/`    | `src/normalizers/`    | all normalizers, `NormalizerWrapper`, and the                 |
:                   :                       : `spm_precompiled` charsmap (crate `spm_precompiled` 0.1)      :
| `pre_tokenizers/` | `src/pre_tokenizers/` | all pre-tokenizers, `PreTokenizerWrapper`, Unicode scripts    |
:                   :                       : table                                                         :
| `processors/`     | `src/processors/`     | `BertProcessing`, `RobertaProcessing`, `TemplateProcessing`,  |
:                   :                       : `Sequence`, `PostProcessorWrapper`                            :
| `decoders/`       | `src/decoders/`       | all decoders and `DecoderWrapper`                             |
| `utils/`          | `src/utils/`          | regex wrapper (`SysRegex`), JSON/serde helpers, UTF-8         |
:                   :                       : helpers, `Cache`                                              :
| `unicode/`        | external crates       | Unicode normalization, general categories / `char`            |
:                   :                       : predicates, grapheme segmentation (see below)                 :
| `tests/`          | `tests/*.rs`          | the crate's integration tests                                 |
| `testdata/`       | `data/`               | the crate's test fixtures (`tokenizer.json`, vocab and merges |
:                   :                       : files)                                                        :

Every Rust file maps to one `.h`/`.cc` pair with the same base name (`mod.rs`
becomes `<dir>.{h,cc}`; `src/tokenizer/mod.rs` is split into `api.{h,cc}` for
the traits and `tokenizer.{h,cc}` for `Tokenizer`). Every `#[cfg(test)]` module
and every file under the crate's `tests/` directory is ported to a gtest file
next to its counterpart.

## Third-party dependencies replaced

| Rust crate                          | Replacement                                    |
| ----------------------------------- | ---------------------------------------------- |
| `onig` (`SysRegex`) and `regex`     | [Oniguruma](https://github.com/kkos/oniguruma) |
:                                     : 6.9.10 (`ONIG_ENCODING_UTF8`,                  :
:                                     : `ONIG_SYNTAX_DEFAULT`), wrapped in             :
:                                     : `utils/regex.h`. The Rust `regex`/`onig`       :
:                                     : syntaxes and Oniguruma's Ruby syntax agree on  :
:                                     : everything the shipped tokenizers use          :
:                                     : (`\p{...}` classes, lookaround, `(?i\:...)`,   :
:                                     : `\s` = `White_Space`).                         :
| `serde` / `serde_json`              | `nlohmann::ordered_json` through               |
:                                     : `utils/json.h`. Field order,                   :
:                                     : `#[serde(default)]`, `deny_unknown_fields`,    :
:                                     : untagged fallbacks and the serde error texts   :
:                                     : are reproduced by hand in each                 :
:                                     : `FromJson`/`ToJson`.                           :
| `unicode-normalization` (+          | `unicode/normalization.{h,cc}` with tables     |
: `unicode_normalization_alignments`) : generated from the crate's `tables.rs`         :
:                                     : (`unicode/gen_tables.py`), i.e. the same       :
:                                     : Unicode version as the Rust build.             :
| `unicode_categories` and the `char` | `unicode/categories.{h,cc}` (tables generated  |
: predicates of the Rust standard     : from the Rust crates;                          :
: library                             : `ToLowercase`/`ToUppercase` implement the full :
:                                     : multi-code-point mappings).                    :
| `unicode-segmentation` (extended    | `unicode/graphemes.{h,cc}`.                    |
: grapheme clusters)                  :                                                :
| `spm_precompiled`                   | `normalizers/spm_precompiled.{h,cc}`.          |
| `daachorse`                         | `tokenizer/aho_corasick.{h,cc}`, a small       |
: (`DoubleArrayAhoCorasick`,          : matcher with identical `leftmost_find_iter`    :
: leftmost-longest)                   : semantics.                                     :
| `ahash`/`HashMap`, `dary_heap`,     | `absl::flat_hash_map`, `std::priority_queue`,  |
: `rand`, `derive_builder`,           : `absl\:\:BitGen`, plain C++.                   :
: `thiserror`, ...                    :                                                :

## Deviations from the Rust crate

Deviations are marked with `// PORT NOTE:` comments at the point where they
occur. The systematic ones are:

*   **Not ported:** trainers (`*/trainer.rs`, `Trainer`, `train*`,
    `Model::get_trainer`), `Model::save`, `from_pretrained` (network access),
    `utils/parallelism.rs` (`encode_batch`/`decode_batch` are plain loops),
    progress bars, and the Python/Node bindings.
*   **Errors instead of panics:** `Result<T>` is `absl::StatusOr<T>`; `panic!`,
    `unwrap()` and `expect()` on user-reachable paths return `absl::Status`
    errors (no `CHECK`, no exceptions). Invalid UTF-8 input is rejected with
    `kInvalidArgument` at `Tokenizer::Encode` instead of panicking.
*   **JSON:** `nlohmann::json` cannot distinguish all serde cases (e.g. `-0` vs
    `0`); numeric type checks follow serde (`u32` rejects negative and
    fractional numbers, `f64` accepts integers). Serialization (`ToString`)
    emits the same fields in the same order as serde, including `null` for
    absent components.
*   **HashMap iteration order** is unspecified in both languages; code never
    depends on it, and vocabularies are emitted sorted by id like the Rust
    `Serialize` implementations do.
*   **Regex engine:** patterns are compiled by Oniguruma rather than the `onig`
    crate's bundled Oniguruma (same engine, version 6.9.10 vs. the crate's
    6.9.x); `fancy_regex` (only used by the crate's `unstable_wasm` feature) is
    not supported.
*   `Tokenizer::Save` writes with `std::ofstream`; `FromFile` reads with
    `std::ifstream`.

## Updating

When bumping the crate version, diff the Rust sources against the previous
version and apply the changes to the corresponding `.cc` files; port new tests
alongside. The Unicode tables are regenerated with `unicode/gen_tables.py` from
the Rust crates' `tables.rs` files.
