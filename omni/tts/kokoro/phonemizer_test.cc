// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "omni/tts/kokoro/phonemizer.h"

#include <unistd.h>

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

std::string GetTestEspeakDataDir() {
  std::string base_dir = ::testing::SrcDir();

  // Candidate paths across internal and external test environments:
  const std::vector<std::string> candidate_paths = {
      // 1. Direct workspace / relative path (short path avoiding N_PATH_HOME
      // limits):
      "third_party/espeak_ng/espeak-ng-data",
      "external/espeak_ng/espeak-ng-data",
      "external/org_espeak_ng/espeak-ng-data",
      "espeak_ng/espeak-ng-data",
      // 2. Internal test runfiles:
      absl::StrCat(base_dir, "/",
                   "espeak_ng/espeak-ng-data"),
      // 3. External repository runfiles:
      absl::StrCat(base_dir, "/espeak_ng/espeak-ng-data"),
      absl::StrCat(base_dir, "/org_espeak_ng/espeak-ng-data"),
      absl::StrCat(base_dir, "/litert_lm/third_party/espeak_ng/espeak-ng-data"),
  };

  for (const auto& path : candidate_paths) {
    if (!ResolveEspeakDataDir(path).empty()) {
      return path;
    }
  }
  return candidate_paths[0];
}

TEST(PhonemizerTest, EmptyPathFails) {
  auto phonemizer = KokoroPhonemizer::Create("");
  EXPECT_FALSE(phonemizer.ok());
}

TEST(PhonemizerTest, NonExistentPathFails) {
  auto phonemizer = KokoroPhonemizer::Create("/non_existent_directory");
  EXPECT_FALSE(phonemizer.ok());
}

TEST(PhonemizerTest, KokoroPhonemizerBasic) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  std::string text = "Hello world.";
  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  const std::vector<int> expected_tokens = {0,  50,  42, 54, 156, 31, 16,
                                            65, 156, 87, 54, 46,  4,  0};
  EXPECT_EQ(tokens, expected_tokens);
}

TEST(PhonemizerTest, CustomLexicon) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"customword", "kˈʌstəmˌwɜːd"},
      {"litert", "lˌItˌɑɹtˈi"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(GetTestEspeakDataDir(), custom_lexicon));

  ASSERT_OK_AND_ASSIGN(std::string custom_ipa,
                       phonemizer->WordToIpa("customword"));
  EXPECT_EQ(custom_ipa, "kˈʌstəmˌwɜːd");
  ASSERT_OK_AND_ASSIGN(std::string upper_custom_ipa,
                       phonemizer->WordToIpa("CUSTOMWORD"));
  EXPECT_EQ(upper_custom_ipa, "kˈʌstəmˌwɜːd");
  ASSERT_OK_AND_ASSIGN(std::string text_ipa,
                       phonemizer->TextToIpa("customword"));
  EXPECT_FALSE(text_ipa.empty());
}

TEST(PhonemizerTest, TextToIpa) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa("Hello world!"));
  EXPECT_FALSE(ipa.empty());
  EXPECT_NE(ipa.find('!'), std::string::npos);
}

TEST(PhonemizerTest, NormalizeMisakiPhonemes) {
  std::string normalized = NormalizeMisakiPhonemes("hɛloʊ");
  EXPECT_FALSE(normalized.empty());
  EXPECT_EQ(NormalizeMisakiPhonemes("aɪ"), "I");
  EXPECT_EQ(NormalizeMisakiPhonemes("aʊ"), "W");
  EXPECT_EQ(NormalizeMisakiPhonemes("eɪ"), "A");
  EXPECT_EQ(NormalizeMisakiPhonemes("ɔɪ"), "Y");
  EXPECT_EQ(NormalizeMisakiPhonemes("oʊ"), "O");
  EXPECT_EQ(NormalizeMisakiPhonemes("əʊ"), "Q");
  EXPECT_EQ(NormalizeMisakiPhonemes("tʃ"), "ʧ");
  EXPECT_EQ(NormalizeMisakiPhonemes("dʒ"), "ʤ");
}

TEST(PhonemizerTest, GetKokoroVocabMapNotEmpty) {
  const auto& vocab = GetKokoroVocabMap();
  EXPECT_FALSE(vocab.empty());
  EXPECT_TRUE(vocab.contains("a"));
  EXPECT_TRUE(vocab.contains(" "));
  EXPECT_TRUE(vocab.contains(";"));
  EXPECT_TRUE(vocab.contains("?"));
}

TEST(PhonemizerTest, NormalizeLanguageCode) {
  EXPECT_EQ(NormalizeLanguageCode("a"), "en-us");
  EXPECT_EQ(NormalizeLanguageCode("en-us"), "en-us");
  EXPECT_EQ(NormalizeLanguageCode("en-US"), "en-us");
  EXPECT_EQ(NormalizeLanguageCode("en_US"), "en-us");
  EXPECT_EQ(NormalizeLanguageCode("en-AU"), "en-us");
  EXPECT_EQ(NormalizeLanguageCode("b"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("en-gb"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("en-GB"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("en-UK"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("e"), "es");
  EXPECT_EQ(NormalizeLanguageCode("es"), "es");
  EXPECT_EQ(NormalizeLanguageCode("es-MX"), "es");
  EXPECT_EQ(NormalizeLanguageCode("es-419"), "es");
  EXPECT_EQ(NormalizeLanguageCode("f"), "fr-fr");
  EXPECT_EQ(NormalizeLanguageCode("fr-fr"), "fr-fr");
  EXPECT_EQ(NormalizeLanguageCode("fr-CA"), "fr-fr");
  EXPECT_EQ(NormalizeLanguageCode("h"), "hi");
  EXPECT_EQ(NormalizeLanguageCode("hi"), "hi");
  EXPECT_EQ(NormalizeLanguageCode("hi-IN"), "hi");
  EXPECT_EQ(NormalizeLanguageCode("i"), "it");
  EXPECT_EQ(NormalizeLanguageCode("it"), "it");
  EXPECT_EQ(NormalizeLanguageCode("it-IT"), "it");
  EXPECT_EQ(NormalizeLanguageCode("p"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("pt-br"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("pt-BR"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("pt-PT"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("j"), "ja");
  EXPECT_EQ(NormalizeLanguageCode("ja"), "ja");
  EXPECT_EQ(NormalizeLanguageCode("ja-JP"), "ja");
  EXPECT_EQ(NormalizeLanguageCode("z"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("zh"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("cmn"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("zh-CN"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("zh-Hant-TW"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("chinese"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("mandarin"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("de"), "de");
  EXPECT_EQ(NormalizeLanguageCode(""), "en-us");
}

TEST(PhonemizerTest, MultilingualPhonemization) {
  // American English ('a' / "en-us")
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer_us,
      KokoroPhonemizer::Create(GetTestEspeakDataDir(), "en-us"));
  EXPECT_EQ(phonemizer_us->language(), "en-us");
  ASSERT_OK_AND_ASSIGN(std::string us_ipa,
                       phonemizer_us->TextToIpa("Hello world"));
  EXPECT_FALSE(us_ipa.empty());

  // Multilingual with custom lexicon override (e.g. Spanish)
  absl::flat_hash_map<std::string, std::string> spanish_lexicon = {
      {"hola", "ˈola"},
      {"mundo", "mˈundo"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer_es,
      KokoroPhonemizer::Create(GetTestEspeakDataDir(), "es", spanish_lexicon));
  EXPECT_EQ(phonemizer_es->language(), "es");
  ASSERT_OK_AND_ASSIGN(std::string hola_ipa, phonemizer_es->WordToIpa("hola"));
  EXPECT_EQ(hola_ipa, "ˈola");
  ASSERT_OK_AND_ASSIGN(std::string mundo_ipa,
                       phonemizer_es->WordToIpa("mundo"));
  EXPECT_EQ(mundo_ipa, "mˈundo");
  ASSERT_OK_AND_ASSIGN(std::string es_ipa,
                       phonemizer_es->TextToIpa("hola mundo"));
  EXPECT_FALSE(es_ipa.empty());
}

TEST(PhonemizerTest, EspeakVoiceForLanguage) {
  // espeak-ng resolves a voice name against the voice file identifier, so every
  // Kokoro language maps to itself except French, whose file is `roa/fr`.
  EXPECT_EQ(EspeakVoiceForLanguage("fr-fr"), "fr");
  EXPECT_EQ(EspeakVoiceForLanguage("en-us"), "en-us");
  EXPECT_EQ(EspeakVoiceForLanguage("en-gb"), "en");
  EXPECT_EQ(EspeakVoiceForLanguage("es"), "es");
  EXPECT_EQ(EspeakVoiceForLanguage("hi"), "hi");
  EXPECT_EQ(EspeakVoiceForLanguage("it"), "it");
  EXPECT_EQ(EspeakVoiceForLanguage("pt-br"), "pt-br");
}

TEST(PhonemizerTest, BritishEnglishPhonemization) {
  // Regression test: "en-gb" voice file in espeak-ng is `gmw/en`, so it must
  // resolve to "en" via EspeakVoiceForLanguage.
  ASSERT_OK_AND_ASSIGN(auto phonemizer, KokoroPhonemizer::Create(
                                            GetTestEspeakDataDir(), "en-gb"));
  EXPECT_EQ(phonemizer->language(), "en-gb");

  ASSERT_OK_AND_ASSIGN(std::string ipa,
                       phonemizer->TextToIpa("Good afternoon!"));
  EXPECT_FALSE(ipa.empty());

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds("Good afternoon."));
  EXPECT_GT(tokens.size(), 2);
}

TEST(PhonemizerTest, DevanagariDandaPunctuation) {
  // Regression test: Devanagari danda '।' (U+0964) should be recognized as
  // sentence-ending punctuation and mapped to '.', rather than swallowed or
  // glued to the preceding word.
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir(), "hi"));
  EXPECT_EQ(phonemizer->language(), "hi");

  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa("नमस्ते।"));
  EXPECT_FALSE(ipa.empty());
  EXPECT_TRUE(absl::StrContains(ipa, "."));
}

TEST(PhonemizerTest, FrenchPhonemization) {
  // Regression test: "fr-fr" is not a name `espeak_SetVoiceByName` resolves, so
  // French used to fail on every word and produce no phonemes at all.
  ASSERT_OK_AND_ASSIGN(auto phonemizer, KokoroPhonemizer::Create(
                                            GetTestEspeakDataDir(), "fr-fr"));
  EXPECT_EQ(phonemizer->language(), "fr-fr");

  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa("Bonjour !"));
  EXPECT_FALSE(ipa.empty());
  EXPECT_TRUE(absl::StrContains(ipa, "ʒ"));

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds("Bonjour le monde."));
  EXPECT_GT(tokens.size(), 2);
}

TEST(PhonemizerTest, NonAsciiUtf8Tokenization) {
  absl::flat_hash_map<std::string, std::string> lexicon = {
      {"español", "esˈpaɲol"},
      {"café", "kæˈfeɪ"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(GetTestEspeakDataDir(), "en-us", lexicon));
  ASSERT_OK_AND_ASSIGN(std::string ipa,
                       phonemizer->TextToIpa("El español y café."));
  EXPECT_FALSE(ipa.empty());
  // Verify that multi-byte UTF-8 accented words like "español" and "café"
  // are tokenized as unified words and match custom lexicon entries without
  // being split into fragmented characters.
  EXPECT_TRUE(absl::StrContains(ipa, "esˈpaɲol"));
  EXPECT_TRUE(absl::StrContains(ipa, "kæˈfA"));
}

TEST(PhonemizerTest, ConcurrentInitialization) {
  std::vector<std::thread> threads;
  std::vector<absl::StatusOr<std::unique_ptr<KokoroPhonemizer>>> results(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([i, &results]() {
      results[i] = KokoroPhonemizer::Create(GetTestEspeakDataDir(), "en-us");
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK_AND_ASSIGN(auto result, std::move(results[i]));
    EXPECT_EQ(result->language(), "en-us");
  }
}

TEST(PhonemizerTest, ResolveEspeakDataDirValid) {
  std::string dir = ResolveEspeakDataDir(GetTestEspeakDataDir());
  EXPECT_FALSE(dir.empty());
}

TEST(PhonemizerTest, ResolveEspeakDataDirNotFound) {
  std::string dir = ResolveEspeakDataDir("/non_existent_explicit_dir");
  EXPECT_TRUE(dir.empty());
}

TEST(PhonemizerTest, SetLanguageDynamically) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, KokoroPhonemizer::Create(
                                            GetTestEspeakDataDir(), "en-us"));
  EXPECT_EQ(phonemizer->language(), "en-us");

  // Dynamically update to British English using 1-letter voice code.
  phonemizer->SetLanguage("b");
  EXPECT_EQ(phonemizer->language(), "en-gb");
  ASSERT_OK_AND_ASSIGN(std::string gb_ipa,
                       phonemizer->TextToIpa("Hello world"));
  EXPECT_FALSE(gb_ipa.empty());

  // Dynamically update using full name alias "english".
  phonemizer->SetLanguage("english");
  EXPECT_EQ(phonemizer->language(), "en-us");
  ASSERT_OK_AND_ASSIGN(std::string en_ipa,
                       phonemizer->TextToIpa("Hello world"));
  EXPECT_FALSE(en_ipa.empty());

  // Dynamically update to Spanish.
  phonemizer->SetLanguage("spanish");
  EXPECT_EQ(phonemizer->language(), "es");
}

TEST(PhonemizerTest, PhonemeTokensDiagnostic) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  struct TestCase {
    std::string prompt;
    std::vector<std::string> valid_ipa_candidates;
    std::vector<std::vector<int>> valid_token_candidates;
  };

  const std::vector<TestCase> test_cases = {
      {
          .prompt = "Hello world.",
          .valid_ipa_candidates = {"hᵊlˈO wˈɜld."},
          .valid_token_candidates = {{0, 50, 42, 54, 156, 31, 16, 65, 156, 87,
                                      54, 46, 4, 0}},
      },
      {
          .prompt = "Hello from LiteRT TTS Engine.",
          .valid_ipa_candidates = {"hᵊlˈO fɹʌm lˌItˌɑɹtˈi tˌitˌiˈɛs ˈɛnʤɪn."},
          .valid_token_candidates = {{0,   50,  42,  54, 156, 31, 16,  48,  123,
                                      138, 55,  16,  54, 157, 25, 62,  157, 69,
                                      123, 62,  156, 51, 16,  62, 157, 51,  62,
                                      157, 51,  156, 86, 61,  16, 156, 86,  56,
                                      82,  102, 56,  4,  0}},
      },
      {
          .prompt = "The quick brown fox jumps over the lazy dog.",
          .valid_ipa_candidates =
              {"ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɑɡ."},
          .valid_token_candidates = {{0,   81, 83,  16,  53, 65,  156, 102, 53,
                                      16,  44, 123, 156, 39, 56,  16,  48,  156,
                                      69,  53, 61,  16,  82, 156, 138, 55,  58,
                                      61,  16, 156, 31,  64, 83,  123, 16,  81,
                                      83,  16, 54,  156, 24, 68,  51,  16,  46,
                                      156, 69, 92,  4,   0}},
      },
      {
          .prompt = "Kokoro is an open-source text-to-speech model.",
          .valid_ipa_candidates =
              {"kˈOkəɹO ɪz æn ˈOpənsˈoɹs tˈɛksttəspˈiʧ mˈɑdᵊl.",
               "kˈOkəɹO ɪz æn ˈOpənsˈɔɹs tˈɛksttəspˈiʧ mˈɑdᵊl."},
          .valid_token_candidates =
              {{0,   53, 156, 31,  53, 83,  123, 31, 16, 102, 68, 16,
                72,  56, 16,  156, 31, 58,  83,  56, 61, 156, 57, 123,
                61,  16, 62,  156, 86, 53,  61,  62, 62, 83,  61, 58,
                156, 51, 133, 16,  55, 156, 69,  46, 42, 54,  4,  0},
               {0,   53, 156, 31,  53, 83,  123, 31, 16, 102, 68, 16,
                72,  56, 16,  156, 31, 58,  83,  56, 61, 156, 76, 123,
                61,  16, 62,  156, 86, 53,  61,  62, 62, 83,  61, 58,
                156, 51, 133, 16,  55, 156, 69,  46, 42, 54,  4,  0}},
      },
  };

  for (const auto& test_case : test_cases) {
    ASSERT_OK_AND_ASSIGN(std::string ipa,
                         phonemizer->TextToIpa(test_case.prompt));
    bool ipa_matched = false;
    for (const auto& candidate : test_case.valid_ipa_candidates) {
      if (ipa == candidate) {
        ipa_matched = true;
        break;
      }
    }
    EXPECT_TRUE(ipa_matched) << "Got unexpected IPA: " << ipa;

    ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                         phonemizer->TextToPhonemeIds(test_case.prompt));
    bool tokens_matched = false;
    for (const auto& candidate : test_case.valid_token_candidates) {
      if (tokens == candidate) {
        tokens_matched = true;
        break;
      }
    }
    EXPECT_TRUE(tokens_matched);
  }
}

TEST(PhonemizerTest, EmDashAndUnicodePunctuation) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  ASSERT_OK_AND_ASSIGN(std::string dash_ipa1,
                       phonemizer->TextToIpa("Obamacare — the"));
  EXPECT_TRUE(absl::StrContains(dash_ipa1, "—"));

  ASSERT_OK_AND_ASSIGN(std::string dash_ipa2,
                       phonemizer->TextToIpa("Obamacare—the"));
  EXPECT_TRUE(absl::StrContains(dash_ipa2, "—"));

  ASSERT_OK_AND_ASSIGN(std::vector<int> dash_tokens,
                       phonemizer->TextToPhonemeIds("Obamacare — the"));
  // Token ID 9 is '—' in Kokoro vocab.
  EXPECT_THAT(dash_tokens, ::testing::Contains(9));
}

TEST(PhonemizerTest, CurlyApostropheContraction) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  ASSERT_OK_AND_ASSIGN(std::string straight_ipa,
                       phonemizer->TextToIpa("don't"));
  ASSERT_OK_AND_ASSIGN(std::string curly_ipa, phonemizer->TextToIpa("don’t"));
  EXPECT_EQ(curly_ipa, straight_ipa);
}

TEST(PhonemizerTest, WhitespaceOnlyReturnsEmptyTokens) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir()));

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens1,
                       phonemizer->TextToPhonemeIds("   \n\n\t  "));
  EXPECT_TRUE(tokens1.empty());
}

TEST(PhonemizerTest, LanguageForVoiceName) {
  EXPECT_EQ(LanguageForVoiceName("ef_dora"), "es");
  EXPECT_EQ(LanguageForVoiceName("em_alex"), "es");
  EXPECT_EQ(LanguageForVoiceName("hf_alpha"), "hi");
  EXPECT_EQ(LanguageForVoiceName("hm_omega"), "hi");
  EXPECT_EQ(LanguageForVoiceName("af_heart"), "en-us");
  EXPECT_EQ(LanguageForVoiceName("am_adam"), "en-us");
  EXPECT_EQ(LanguageForVoiceName("bf_emma"), "en-gb");
  EXPECT_EQ(LanguageForVoiceName("bm_george"), "en-gb");
  EXPECT_EQ(LanguageForVoiceName("ff_siwis"), "fr-fr");
  EXPECT_EQ(LanguageForVoiceName("if_sara"), "it");
  EXPECT_EQ(LanguageForVoiceName("pf_dora"), "pt-br");
  EXPECT_EQ(LanguageForVoiceName("jf_alpha"), "ja");
  EXPECT_EQ(LanguageForVoiceName("zf_xiaobei"), "cmn");
  // Strips leading paths and file extensions.
  EXPECT_EQ(LanguageForVoiceName("voices/ef_dora.bin"), "es");
  EXPECT_EQ(LanguageForVoiceName("/tmp/hf_alpha.pt"), "hi");
  // Unknown or un-prefixed voices return empty string from
  // LanguageForVoiceName.
  EXPECT_EQ(LanguageForVoiceName("my_custom_voice"), "");
}

TEST(PhonemizerTest, ResolveAndValidateLanguage) {
  // Auto-detection when language is empty.
  ASSERT_OK_AND_ASSIGN(std::string es_lang,
                       ResolveAndValidateLanguage("ef_dora", ""));
  EXPECT_EQ(es_lang, "es");
  ASSERT_OK_AND_ASSIGN(std::string hi_lang,
                       ResolveAndValidateLanguage("hf_alpha", ""));
  EXPECT_EQ(hi_lang, "hi");
  ASSERT_OK_AND_ASSIGN(std::string en_lang,
                       ResolveAndValidateLanguage("af_heart", ""));
  EXPECT_EQ(en_lang, "en-us");
  // Un-prefixed voice with empty language defaults to en-us.
  ASSERT_OK_AND_ASSIGN(std::string custom_lang,
                       ResolveAndValidateLanguage("my_custom_voice", ""));
  EXPECT_EQ(custom_lang, "en-us");

  // Explicit matching language and voice.
  ASSERT_OK_AND_ASSIGN(std::string match_es,
                       ResolveAndValidateLanguage("ef_dora", "es"));
  EXPECT_EQ(match_es, "es");
  ASSERT_OK_AND_ASSIGN(std::string match_es_alias,
                       ResolveAndValidateLanguage("ef_dora", "spanish"));
  EXPECT_EQ(match_es_alias, "es");
  ASSERT_OK_AND_ASSIGN(std::string match_en,
                       ResolveAndValidateLanguage("af_heart", "en-us"));
  EXPECT_EQ(match_en, "en-us");

  // Mismatch between voice prefix and requested language returns
  // InvalidArgument.
  auto mismatch1 = ResolveAndValidateLanguage("ef_dora", "en-us");
  EXPECT_FALSE(mismatch1.ok());
  EXPECT_EQ(mismatch1.status().code(), absl::StatusCode::kInvalidArgument);

  auto mismatch2 = ResolveAndValidateLanguage("af_heart", "es");
  EXPECT_FALSE(mismatch2.ok());
  EXPECT_EQ(mismatch2.status().code(), absl::StatusCode::kInvalidArgument);

  auto mismatch3 = ResolveAndValidateLanguage("hf_alpha", "fr-fr");
  EXPECT_FALSE(mismatch3.ok());
  EXPECT_EQ(mismatch3.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PhonemizerTest, IsWordCodePoint) {
  // ASCII alphanumeric and intra-word punctuation.
  EXPECT_TRUE(IsWordCodePoint('a'));
  EXPECT_TRUE(IsWordCodePoint('Z'));
  EXPECT_TRUE(IsWordCodePoint('5'));
  EXPECT_TRUE(IsWordCodePoint('\''));
  EXPECT_TRUE(IsWordCodePoint('-'));

  // Standard whitespace and ASCII punctuation are NOT word characters.
  EXPECT_FALSE(IsWordCodePoint(' '));
  EXPECT_FALSE(IsWordCodePoint('.'));
  EXPECT_FALSE(IsWordCodePoint(','));
  EXPECT_FALSE(IsWordCodePoint('!'));
  EXPECT_FALSE(IsWordCodePoint('?'));

  // Latin-1 accented letters (Spanish, French, Portuguese, Italian).
  EXPECT_TRUE(IsWordCodePoint(U'ñ'));
  EXPECT_TRUE(IsWordCodePoint(U'á'));
  EXPECT_TRUE(IsWordCodePoint(U'é'));
  EXPECT_TRUE(IsWordCodePoint(U'í'));
  EXPECT_TRUE(IsWordCodePoint(U'ó'));
  EXPECT_TRUE(IsWordCodePoint(U'ú'));
  EXPECT_TRUE(IsWordCodePoint(U'ç'));
  EXPECT_TRUE(IsWordCodePoint(U'ü'));

  // Latin-1 punctuation marks are NOT word characters.
  EXPECT_FALSE(IsWordCodePoint(U'¿'));
  EXPECT_FALSE(IsWordCodePoint(U'¡'));
  EXPECT_FALSE(IsWordCodePoint(U'«'));
  EXPECT_FALSE(IsWordCodePoint(U'»'));

  // Devanagari script (Hindi).
  EXPECT_TRUE(IsWordCodePoint(U'न'));
  EXPECT_TRUE(IsWordCodePoint(U'म'));
  EXPECT_TRUE(IsWordCodePoint(U'स'));
  EXPECT_TRUE(IsWordCodePoint(U'्'));
  EXPECT_TRUE(IsWordCodePoint(U'त'));
  EXPECT_TRUE(IsWordCodePoint(U'े'));

  // CJK and Kana (Chinese / Japanese).
  EXPECT_TRUE(IsWordCodePoint(U'你'));
  EXPECT_TRUE(IsWordCodePoint(U'好'));
  EXPECT_TRUE(IsWordCodePoint(U'あ'));
  EXPECT_TRUE(IsWordCodePoint(U'ア'));
}

TEST(PhonemizerTest, NormalizeMisakiPhonemesFlavors) {
  // English US converts 'x' to 'k' and strips vowel length marker 'ː'.
  EXPECT_EQ(NormalizeMisakiPhonemes("x", MisakiFlavor::kEnglishUs), "k");
  EXPECT_EQ(NormalizeMisakiPhonemes("aː", MisakiFlavor::kEnglishUs), "a");

  // Espeak Generic preserves Spanish 'x' (jota) and length marker 'ː'.
  EXPECT_EQ(NormalizeMisakiPhonemes("x", MisakiFlavor::kEspeakGeneric), "x");
  EXPECT_EQ(NormalizeMisakiPhonemes("aː", MisakiFlavor::kEspeakGeneric), "aː");
  EXPECT_EQ(NormalizeMisakiPhonemes("r", MisakiFlavor::kEspeakGeneric), "r");
  EXPECT_EQ(NormalizeMisakiPhonemes("tʃ", MisakiFlavor::kEspeakGeneric), "ʧ");
  EXPECT_EQ(NormalizeMisakiPhonemes("dʒ", MisakiFlavor::kEspeakGeneric), "ʤ");
}

TEST(PhonemizerTest, SpanishPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir(), "es"));
  EXPECT_EQ(phonemizer->language(), "es");

  std::string text = "El zorro y el gato en el ojo.";
  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa(text));
  EXPECT_FALSE(ipa.empty());
  // Spanish jota 'x' should be preserved for "ojo" / "zorro".
  EXPECT_TRUE(absl::StrContains(ipa, "x"))
      << "Expected Spanish jota 'x' in IPA: " << ipa;

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  EXPECT_GE(tokens.size(), 2);
  EXPECT_EQ(tokens.front(), 0);  // BOS
  EXPECT_EQ(tokens.back(), 0);   // EOS

  const auto& vocab = GetKokoroVocabMap();
  int jota_token = vocab.at("x");
  EXPECT_THAT(tokens, ::testing::Contains(jota_token));
}

TEST(PhonemizerTest, HindiPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(GetTestEspeakDataDir(), "hi"));
  EXPECT_EQ(phonemizer->language(), "hi");

  std::string text = "नमस्ते भारत";
  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa(text));
  EXPECT_FALSE(ipa.empty()) << "Hindi IPA should not be empty.";

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  EXPECT_GE(tokens.size(), 2);
  EXPECT_EQ(tokens.front(), 0);  // BOS
  EXPECT_EQ(tokens.back(), 0);   // EOS
  // Verify that phoneme tokens were produced between BOS and EOS.
  EXPECT_GT(tokens.size(), 2)
      << "Expected non-empty phoneme sequence for Hindi text.";
}

}  // namespace
}  // namespace litert::omni::tts
