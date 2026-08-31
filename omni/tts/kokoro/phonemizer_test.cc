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
  EXPECT_EQ(NormalizeLanguageCode("b"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("en-gb"), "en-gb");
  EXPECT_EQ(NormalizeLanguageCode("e"), "es");
  EXPECT_EQ(NormalizeLanguageCode("es"), "es");
  EXPECT_EQ(NormalizeLanguageCode("f"), "fr-fr");
  EXPECT_EQ(NormalizeLanguageCode("fr-fr"), "fr-fr");
  EXPECT_EQ(NormalizeLanguageCode("h"), "hi");
  EXPECT_EQ(NormalizeLanguageCode("hi"), "hi");
  EXPECT_EQ(NormalizeLanguageCode("i"), "it");
  EXPECT_EQ(NormalizeLanguageCode("it"), "it");
  EXPECT_EQ(NormalizeLanguageCode("p"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("pt-br"), "pt-br");
  EXPECT_EQ(NormalizeLanguageCode("j"), "ja");
  EXPECT_EQ(NormalizeLanguageCode("ja"), "ja");
  EXPECT_EQ(NormalizeLanguageCode("z"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("zh"), "cmn");
  EXPECT_EQ(NormalizeLanguageCode("cmn"), "cmn");
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
  // The test data fixture contains only en-us data, so en-gb returns an error
  // without falling back silently to en-us.
  auto gb_status = phonemizer->TextToIpa("Hello world");
  EXPECT_FALSE(gb_status.ok());
  EXPECT_TRUE(
      absl::StrContains(gb_status.status().message(),
                        "Failed to set espeak voice for language: en-gb"));

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

}  // namespace
}  // namespace litert::omni::tts
