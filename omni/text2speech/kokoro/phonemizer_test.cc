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

#include "omni/text2speech/kokoro/phonemizer.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

std::string BuildMinimalEnglishBlobForPhonemizerTest() {
  using ::litert::omni::tts::kokoro::AppendUint32;
  using ::litert::omni::tts::kokoro::BuildCjkBlob;

  // US entries are copied verbatim from the bundled `en-lexicon` (misaki
  // us_gold / us_silver), sorted by key.
  const std::vector<std::pair<std::string, std::string>> us_entries = {
      {"an", "4æn"},
      {"brown", "4bɹˈWn"},
      {"dog", "4dˈɔɡ"},
      {"engine", "4ˈɛnʤən"},
      {"fox", "4fˈɑks"},
      {"from", "4fɹʌm"},
      {"hello", "4həlˈO"},
      {"is", "4ɪz"},
      {"jumps", "3ʤˈʌmps"},
      {"lazy", "4lˈAzi"},
      {"model", "4mˈɑdᵊl"},
      {"open", "4ˈOpᵊn"},
      {"over", "4ˈOvəɹ"},
      {"quick", "4kwˈɪk"},
      {"source", "4sˈɔɹs"},
      {"speech", "4spˈiʧ"},
      {"text", "4tˈɛkst"},
      {"the", "4ði"},
      {"to", "4tu"},
      {"water", "4wˈɔɾəɹ"},
      {"world", "4wˈɜɹld"},
  };
  const std::vector<std::pair<std::string, std::string>> gb_entries = {
      {"hello", "4həlˈQ"},
      {"water", "4wˈɔːtə"},
      {"world", "4wˈɜːld"},
  };

  std::string us_idx;
  std::string us_key;
  std::string us_vidx;
  std::string us_val;
  for (const auto& [k, v] : us_entries) {
    AppendUint32(static_cast<uint32_t>(us_key.size()), us_idx);
    us_key.append(k);
    us_key.push_back('\0');
    AppendUint32(static_cast<uint32_t>(us_val.size()), us_vidx);
    us_val.append(v);
    us_val.push_back('\0');
  }
  AppendUint32(static_cast<uint32_t>(us_key.size()), us_idx);
  AppendUint32(static_cast<uint32_t>(us_val.size()), us_vidx);

  std::string gb_idx;
  std::string gb_key;
  std::string gb_vidx;
  std::string gb_val;
  for (const auto& [k, v] : gb_entries) {
    AppendUint32(static_cast<uint32_t>(gb_key.size()), gb_idx);
    gb_key.append(k);
    gb_key.push_back('\0');
    AppendUint32(static_cast<uint32_t>(gb_val.size()), gb_vidx);
    gb_val.append(v);
    gb_val.push_back('\0');
  }
  AppendUint32(static_cast<uint32_t>(gb_key.size()), gb_idx);
  AppendUint32(static_cast<uint32_t>(gb_val.size()), gb_vidx);

  return BuildCjkBlob({
      {"us-idx", us_idx},
      {"us-key", us_key},
      {"us-vidx", us_vidx},
      {"us-val", us_val},
      {"gb-idx", gb_idx},
      {"gb-key", gb_key},
      {"gb-vidx", gb_vidx},
      {"gb-val", gb_val},
  });
}

absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> CreateTestPhonemizer(
    absl::string_view language = "en-us",
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {},
    absl::string_view text_norm_rules = "", absl::string_view cjk_lexicon = "",
    absl::string_view zh_t2s_dictionary = "") {
  if (cjk_lexicon.empty()) {
    const std::string norm_lang = NormalizeLanguageCode(language);
    if (norm_lang == "en-us" || norm_lang == "en-gb") {
      // `Create` keeps views into the blob, so it must outlive every
      // phonemizer returned here.
      static const std::string& en_blob =
          *new std::string(BuildMinimalEnglishBlobForPhonemizerTest());
      return KokoroPhonemizer::Create(/*data_dir=*/"", language, custom_lexicon,
                                      text_norm_rules, en_blob,
                                      zh_t2s_dictionary);
    }
  }
  return KokoroPhonemizer::Create(/*data_dir=*/"", language, custom_lexicon,
                                  text_norm_rules, cjk_lexicon,
                                  zh_t2s_dictionary);
}

// Reads the Mandarin text normalization rule table that ships with the model,
// so that the tests below exercise the data we actually serve.
std::string ReadMandarinRuleTable() {
  const std::string base_dir = ::testing::SrcDir();
  const std::vector<std::string> candidate_paths = {
      "omni/text2speech/data/zh_textnorm.txt",
      absl::StrCat(base_dir, "/",
                   "odml/litert_lm/omni/text2speech/data/zh_textnorm.txt"),
      absl::StrCat(base_dir,
                   "/litert_lm/omni/text2speech/"
                   "data/zh_textnorm.txt"),
  };
  for (const std::string& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) continue;
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  }
  return "";
}

TEST(PhonemizerTest, UnsupportedLanguageFailsWithPrecondition) {
  // An unsupported language reports a precondition error.
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(/*data_dir=*/"", "de"));
  EXPECT_THAT(phonemizer->WordToIpa("hallo"),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(phonemizer->TextToIpa("Hallo Welt."),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(PhonemizerTest, KokoroPhonemizerBasic) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());

  std::string text = "Hello world.";
  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  const std::vector<int> expected_tokens = {
      0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 4, 0};
  EXPECT_EQ(tokens, expected_tokens);
}

TEST(PhonemizerTest, CustomLexicon) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"customword", "kˈʌstəmˌwɜːd"},
      {"litert", "lˌItˌɑɹtˈi"},
  };
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("en-us", custom_lexicon));

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer_us, CreateTestPhonemizer("en-us"));
  EXPECT_EQ(phonemizer_us->Language(), "en-us");
  ASSERT_OK_AND_ASSIGN(std::string us_ipa,
                       phonemizer_us->TextToIpa("Hello world"));
  EXPECT_FALSE(us_ipa.empty());

  // Multilingual with custom lexicon override (e.g. Spanish)
  absl::flat_hash_map<std::string, std::string> spanish_lexicon = {
      {"hola", "ˈola"},
      {"mundo", "mˈundo"},
  };
  ASSERT_OK_AND_ASSIGN(auto phonemizer_es,
                       CreateTestPhonemizer("es", spanish_lexicon));
  EXPECT_EQ(phonemizer_es->Language(), "es");
  ASSERT_OK_AND_ASSIGN(std::string hola_ipa, phonemizer_es->WordToIpa("hola"));
  EXPECT_EQ(hola_ipa, "ˈola");
  ASSERT_OK_AND_ASSIGN(std::string mundo_ipa,
                       phonemizer_es->WordToIpa("mundo"));
  EXPECT_EQ(mundo_ipa, "mˈundo");
  ASSERT_OK_AND_ASSIGN(std::string es_ipa,
                       phonemizer_es->TextToIpa("hola mundo"));
  EXPECT_FALSE(es_ipa.empty());
}

TEST(PhonemizerTest, BritishEnglishPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("en-gb"));
  EXPECT_EQ(phonemizer->Language(), "en-gb");

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("hi"));
  EXPECT_EQ(phonemizer->Language(), "hi");

  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa("नमस्ते।"));
  EXPECT_FALSE(ipa.empty());
  EXPECT_TRUE(absl::StrContains(ipa, "."));
}

TEST(PhonemizerTest, FrenchPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("fr-fr"));
  EXPECT_EQ(phonemizer->Language(), "fr-fr");

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("en-us", lexicon));
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
    threads.emplace_back(
        [i, &results]() { results[i] = CreateTestPhonemizer("en-us"); });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK_AND_ASSIGN(auto result, std::move(results[i]));
    EXPECT_EQ(result->Language(), "en-us");
  }
}

TEST(PhonemizerTest, SetLanguageDynamically) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("en-us"));
  EXPECT_EQ(phonemizer->Language(), "en-us");

  // Dynamically update to British English using 1-letter voice code.
  phonemizer->SetLanguage("b");
  EXPECT_EQ(phonemizer->Language(), "en-gb");
  ASSERT_OK_AND_ASSIGN(std::string gb_ipa,
                       phonemizer->TextToIpa("Hello world"));
  EXPECT_FALSE(gb_ipa.empty());

  // An unavailable language returns an error without falling back silently.
  phonemizer->SetLanguage("unsupported-lang");
  auto invalid_status = phonemizer->TextToIpa("Goodbye");
  EXPECT_FALSE(invalid_status.ok());
  EXPECT_TRUE(
      absl::StrContains(invalid_status.status().message(), "unsupported-lang"))
      << invalid_status.status().message();

  // Dynamically update using full name alias "english".
  phonemizer->SetLanguage("english");
  EXPECT_EQ(phonemizer->Language(), "en-us");
  ASSERT_OK_AND_ASSIGN(std::string en_ipa,
                       phonemizer->TextToIpa("Hello world"));
  EXPECT_FALSE(en_ipa.empty());

  // Dynamically update to Spanish.
  phonemizer->SetLanguage("spanish");
  EXPECT_EQ(phonemizer->Language(), "es");
}

TEST(PhonemizerTest, EmDashAndUnicodePunctuation) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());

  ASSERT_OK_AND_ASSIGN(std::string straight_ipa,
                       phonemizer->TextToIpa("don't"));
  ASSERT_OK_AND_ASSIGN(std::string curly_ipa, phonemizer->TextToIpa("don’t"));
  EXPECT_EQ(curly_ipa, straight_ipa);
}

TEST(PhonemizerTest, WhitespaceOnlyReturnsEmptyTokens) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer());

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
  // U+30FC prolonged sound mark lengthens the preceding vowel, so it belongs
  // to the word.
  EXPECT_TRUE(IsWordCodePoint(U'ー'));
  // U+30FB katakana middle dot separates words and must not be glued to one.
  EXPECT_FALSE(IsWordCodePoint(U'・'));
  // CJK punctuation is never part of a word.
  EXPECT_FALSE(IsWordCodePoint(U'、'));
  EXPECT_FALSE(IsWordCodePoint(U'。'));
  EXPECT_FALSE(IsWordCodePoint(U'「'));
  EXPECT_FALSE(IsWordCodePoint(U'」'));
}

TEST(PhonemizerTest, NormalizeMisakiPhonemesFlavors) {
  // English US converts 'x' to 'k' and strips vowel length marker 'ː'.
  EXPECT_EQ(NormalizeMisakiPhonemes("x", MisakiFlavor::kEnglishUs), "k");
  EXPECT_EQ(NormalizeMisakiPhonemes("aː", MisakiFlavor::kEnglishUs), "a");

  // Generic flavor preserves Spanish 'x' (jota) and length marker 'ː'.
  EXPECT_EQ(NormalizeMisakiPhonemes("x", MisakiFlavor::kGeneric), "x");
  EXPECT_EQ(NormalizeMisakiPhonemes("aː", MisakiFlavor::kGeneric), "aː");
  EXPECT_EQ(NormalizeMisakiPhonemes("r", MisakiFlavor::kGeneric), "r");
  // Only tied phoneme pairs are merged; an untied pair spans two phonemes and
  // is left alone.
  EXPECT_EQ(NormalizeMisakiPhonemes("t͡ʃ", MisakiFlavor::kGeneric), "ʧ");
  EXPECT_EQ(NormalizeMisakiPhonemes("d͡ʒ", MisakiFlavor::kGeneric), "ʤ");
  EXPECT_EQ(NormalizeMisakiPhonemes("t^s", MisakiFlavor::kGeneric), "ʦ");
  EXPECT_EQ(NormalizeMisakiPhonemes("ts", MisakiFlavor::kGeneric), "ts");
  EXPECT_EQ(NormalizeMisakiPhonemes("kaʊzˈata", MisakiFlavor::kGeneric),
            "kaʊzˈata");
}

TEST(PhonemizerTest, SpanishPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("es"));
  EXPECT_EQ(phonemizer->Language(), "es");

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
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("hi"));
  EXPECT_EQ(phonemizer->Language(), "hi");

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

TEST(PhonemizerTest, ItalianPhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("it"));
  EXPECT_EQ(phonemizer->Language(), "it");

  std::string text = "Ciao, questo è un test in italiano.";
  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa(text));
  EXPECT_FALSE(ipa.empty()) << "Italian IPA should not be empty.";

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  EXPECT_GE(tokens.size(), 2);
  EXPECT_EQ(tokens.front(), 0);  // BOS
  EXPECT_EQ(tokens.back(), 0);   // EOS
  EXPECT_GT(tokens.size(), 2);
}

TEST(PhonemizerTest, PortuguesePhonemization) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("pt-br"));
  EXPECT_EQ(phonemizer->Language(), "pt-br");

  std::string text = "Olá, este é um teste em português.";
  ASSERT_OK_AND_ASSIGN(std::string ipa, phonemizer->TextToIpa(text));
  EXPECT_FALSE(ipa.empty()) << "Portuguese IPA should not be empty.";

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds(text));
  EXPECT_GE(tokens.size(), 2);
  EXPECT_EQ(tokens.front(), 0);  // BOS
  EXPECT_EQ(tokens.back(), 0);   // EOS
  EXPECT_GT(tokens.size(), 2);
}

void AppendDouble(double value, std::string& out) {
  char buf[sizeof(double)];
  std::memcpy(buf, &value, sizeof(double));
  out.append(buf, sizeof(double));
}

void PackSortedMap(std::vector<std::pair<std::string, std::string>> items,
                   std::string& k_idx, std::string& k_blob, std::string& v_idx,
                   std::string& v_blob) {
  using ::litert::omni::tts::kokoro::AppendUint32;
  std::sort(items.begin(), items.end());
  for (const auto& [k, v] : items) {
    AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
    k_blob.append(k);
    k_blob.push_back('\0');
    AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
    v_blob.append(v);
    v_blob.push_back('\0');
  }
  AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
  AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
}

std::string BuildMinimalChineseBlobForPhonemizerTest() {
  using ::litert::omni::tts::kokoro::AppendUint32;
  using ::litert::omni::tts::kokoro::BuildCjkBlob;

  std::vector<std::pair<std::string, uint32_t>> words = {
      {"你好", 2000},  {"世界", 1800},  {"我", 2500},      {"喜欢", 1500},
      {"编程", 1200},  {"他", 2000},    {"买", 1500},      {"了", 3000},
      {"手机", 1500},  {"技术", 1600},  {"二〇二六", 500}, {"年", 2000},
      {"百分之", 800}, {"摄氏度", 600}, {"点", 1000},      {"一四", 400},
  };
  std::sort(words.begin(), words.end());
  std::string w_idx;
  std::string w_key;
  std::string w_frq;
  for (const auto& [w, f] : words) {
    AppendUint32(static_cast<uint32_t>(w_key.size()), w_idx);
    w_key.append(w);
    w_key.push_back('\0');
    AppendUint32(f, w_frq);
  }
  AppendUint32(static_cast<uint32_t>(w_key.size()), w_idx);
  std::string w_tot;
  AppendDouble(std::log(30000.0), w_tot);

  std::string hmm_st;
  for (double p : {-0.26, -3.14e100, -3.14e100, -1.46}) {
    AppendDouble(p, hmm_st);
  }
  std::string hmm_tr(16 * sizeof(double), '\0');
  std::string hmm_en(4 * sizeof(uint32_t), '\0');

  std::string p_idx, p_key, p_vidx, p_val;
  PackSortedMap(
      {
          {"你好", "ni↗xau↓"},
          {"世界", "ʂɨ↘ʨje↘"},
          {"喜欢", "ɕi↓xwan"},
          {"编程", "pjɛ→nꭧʰə↗ŋ"},
          {"手机", "ʂou↓ʨi→"},
          {"技术", "ʨi↘ʂu↘"},
          {"百分之", "pai↓fə→nꭧɨ→"},
          {"摄氏度", "ʂɤ↘ʂɨ↘tu↘"},
      },
      p_idx, p_key, p_vidx, p_val);

  std::string c_idx, c_key, c_vidx, c_val;
  PackSortedMap(
      {
          {"你", "ni↓"},    {"好", "xau↓"},  {"世", "ʂɨ↘"},   {"界", "ʨje↘"},
          {"我", "wo↓"},    {"他", "tʰa→"},  {"买", "mai↓"},  {"了", "lə"},
          {"二", "ɚ↘"},     {"〇", "li↗ŋ"},  {"零", "li↗ŋ"},  {"三", "san→"},
          {"六", "ljou↘"},  {"九", "ʨjou↓"}, {"年", "njɛ↗n"}, {"月", "ɥe↘"},
          {"日", "ʐɨ↘"},    {"號", "xau↘"},  {"点", "tjɛ↓n"}, {"一", "i→"},
          {"四", "sɨ↘"},    {"五", "wu↓"},   {"十", "ʂɨ↗"},   {"百", "pai↓"},
          {"千", "ʨʰjɛ→n"}, {"万", "wan↘"},
      },
      c_idx, c_key, c_vidx, c_val);

  return BuildCjkBlob({
      {"w-idx", w_idx},
      {"w-key", w_key},
      {"w-frq", w_frq},
      {"w-tot", w_tot},
      {"hmm-st", hmm_st},
      {"hmm-tr", hmm_tr},
      {"hmm-en", hmm_en},
      {"hmm-ec", ""},
      {"hmm-ep", ""},
      {"p-idx", p_idx},
      {"p-key", p_key},
      {"p-vidx", p_vidx},
      {"p-val", p_val},
      {"c-idx", c_idx},
      {"c-key", c_key},
      {"c-vidx", c_vidx},
      {"c-val", c_val},
  });
}

TEST(PhonemizerTest, ChineseWithoutLexiconFailsWithPrecondition) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("cmn"));
  EXPECT_THAT(
      phonemizer->TextToIpa("你好，世界！"),
      ::testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      phonemizer->WordToIpa("你好"),
      ::testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(PhonemizerTest, ChineseWithBundledLexiconAndPunctuation) {
  const std::string zh_blob = BuildMinimalChineseBlobForPhonemizerTest();
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("cmn", {}, "", zh_blob));

  ASSERT_OK_AND_ASSIGN(std::string full_ipa,
                       phonemizer->TextToIpa("你好，世界！"));
  EXPECT_EQ(full_ipa, "ni↗xau↓, ʂɨ↘ʨje↘!");

  // CJK punctuation marks map onto Kokoro vocabulary tokens.
  ASSERT_OK_AND_ASSIGN(std::string pause_ipa,
                       phonemizer->TextToIpa("你好、世界；你好：世界「你好」"));
  EXPECT_THAT(pause_ipa, ::testing::HasSubstr(","));
  EXPECT_THAT(pause_ipa, ::testing::HasSubstr(";"));
  EXPECT_THAT(pause_ipa, ::testing::HasSubstr(":"));
  EXPECT_THAT(pause_ipa, ::testing::HasSubstr("“"));
  EXPECT_THAT(pause_ipa, ::testing::HasSubstr("”"));

  // Embedded English inside Chinese is phonemized with en-us.
  ASSERT_OK_AND_ASSIGN(std::string mixed_ipa,
                       phonemizer->TextToIpa("我喜欢 Python 编程"));
  EXPECT_THAT(mixed_ipa, ::testing::HasSubstr("pjɛ→nꭧʰə↗ŋ"));
  EXPECT_THAT(mixed_ipa, ::testing::Not(::testing::HasSubstr("(en)")));

  // Initializing with zh_blob under en-us and switching to cmn via SetLanguage
  // retains chinese_g2p_.
  ASSERT_OK_AND_ASSIGN(auto switched_phonemizer,
                       CreateTestPhonemizer("en-us", {}, "", zh_blob));
  switched_phonemizer->SetLanguage("cmn");
  EXPECT_THAT(switched_phonemizer->TextToIpa("你好，世界！"),
              ::testing::status::IsOkAndHolds("ni↗xau↓, ʂɨ↘ʨje↘!"));
}

TEST(PhonemizerTest, ChineseTextNormalizationMatchesTheWrittenOutForm) {
  const std::string rules = ReadMandarinRuleTable();
  ASSERT_FALSE(rules.empty()) << "could not locate zh_textnorm.txt";
  const std::string zh_blob = BuildMinimalChineseBlobForPhonemizerTest();
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("cmn", {}, rules, zh_blob));

  ASSERT_OK_AND_ASSIGN(std::string year, phonemizer->TextToIpa("2026年"));
  ASSERT_OK_AND_ASSIGN(std::string year_zero,
                       phonemizer->TextToIpa("二零二六年"));
  ASSERT_OK_AND_ASSIGN(std::string year_circle,
                       phonemizer->TextToIpa("二〇二六年"));
  EXPECT_EQ(year, year_zero);
  EXPECT_EQ(year, year_circle);
  EXPECT_THAT(year, ::testing::HasSubstr("li↗ŋ"));
  EXPECT_THAT(year, ::testing::Not(::testing::HasSubstr("〇")));

  ASSERT_OK_AND_ASSIGN(std::string full_date,
                       phonemizer->TextToIpa("2026年9月24號"));
  ASSERT_OK_AND_ASSIGN(std::string full_date_written,
                       phonemizer->TextToIpa("二零二六年九月二十四號"));
  EXPECT_EQ(full_date, full_date_written);

  ASSERT_OK_AND_ASSIGN(std::string decimal, phonemizer->TextToIpa("3.14"));
  ASSERT_OK_AND_ASSIGN(std::string decimal_written,
                       phonemizer->TextToIpa("三点一四"));
  EXPECT_EQ(decimal, decimal_written);

  ASSERT_OK_AND_ASSIGN(std::string percent, phonemizer->TextToIpa("50%"));
  ASSERT_OK_AND_ASSIGN(std::string percent_written,
                       phonemizer->TextToIpa("百分之五十"));
  EXPECT_EQ(percent, percent_written);

  ASSERT_OK_AND_ASSIGN(std::string celsius, phonemizer->TextToIpa("25℃"));
  ASSERT_OK_AND_ASSIGN(std::string celsius_written,
                       phonemizer->TextToIpa("二十五摄氏度"));
  EXPECT_EQ(celsius, celsius_written);
}

TEST(PhonemizerTest, TraditionalChineseIsConvertedBeforeG2p) {
  const std::string zh_blob = BuildMinimalChineseBlobForPhonemizerTest();
  // 妳 and 丗 are Traditional/variant forms that the Simplified lexicon lacks.
  constexpr absl::string_view kT2s = "妳\t你\n丗界\t世界\n";
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("cmn", {}, "", zh_blob, kT2s));
  ASSERT_OK_AND_ASSIGN(std::string simplified,
                       phonemizer->TextToIpa("你好，世界！"));
  EXPECT_THAT(phonemizer->TextToIpa("妳好，丗界！"),
              ::testing::status::IsOkAndHolds(simplified));
}

TEST(PhonemizerTest, MalformedTraditionalChineseTableIsAnError) {
  EXPECT_THAT(CreateTestPhonemizer("cmn", {}, "", "", "not-an-entry\n"),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PhonemizerTest, RuleTableStopsApplyingAfterSwitchingLanguage) {
  const std::string rules = ReadMandarinRuleTable();
  ASSERT_FALSE(rules.empty()) << "could not locate zh_textnorm.txt";
  const std::string zh_blob = BuildMinimalChineseBlobForPhonemizerTest();
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("cmn", {}, rules, zh_blob));
  ASSERT_OK_AND_ASSIGN(auto plain_english,
                       CreateTestPhonemizer("en-us", {}, "", zh_blob));

  phonemizer->SetLanguage("en-us");
  ASSERT_OK_AND_ASSIGN(std::string switched,
                       phonemizer->TextToIpa("50% in 2026"));
  ASSERT_OK_AND_ASSIGN(std::string expected,
                       plain_english->TextToIpa("50% in 2026"));
  EXPECT_EQ(switched, expected);
}

TEST(PhonemizerTest, MalformedRuleTableIsAnError) {
  EXPECT_THAT(CreateTestPhonemizer("cmn", {}, "not-a-record\n").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
}

std::string BuildMinimalJapaneseBlobForPhonemizerTest() {
  using ::litert::omni::tts::kokoro::BuildCjkBlob;
  using ::litert::omni::tts::kokoro::BuildMecabCharProperty;
  using ::litert::omni::tts::kokoro::BuildMecabConnectionMatrix;
  using ::litert::omni::tts::kokoro::BuildMecabDoubleArray;
  using ::litert::omni::tts::kokoro::BuildMecabFeatures;
  using ::litert::omni::tts::kokoro::BuildMecabToken;
  using ::litert::omni::tts::kokoro::PackMecabCharInfo;

  std::vector<uint32_t> char_records(0x3100,
                                     PackMecabCharInfo(1, 0, 0, true, false));
  char_records[' '] = PackMecabCharInfo(1 << 1, 1, 0, true, false);
  for (int cp = 0x3040; cp <= 0x309F; ++cp) {
    char_records[cp] = PackMecabCharInfo(1 << 6, 6, 2, true, false);
  }
  char_records[0x3002] = PackMecabCharInfo(1 << 3, 3, 0, true, false);

  const std::vector<std::string> categories = {
      "DEFAULT",  "SPACE",    "KANJI",        "SYMBOL", "NUMERIC", "ALPHA",
      "HIRAGANA", "KATAKANA", "KANJINUMERIC", "GREEK",  "CYRILLIC"};
  std::vector<uint32_t> sys_offsets;
  const std::string sys_features = BuildMecabFeatures(
      {
          "名詞,普通名詞,一般,*,*,キョー,キョウ,1",
          "助詞,係助詞,*,*,*,ワ,ハ,*",
          "名詞,普通名詞,一般,*,*,テンキ,テンキ,1",
          "助動詞,*,*,*,助動詞-デス,デス,デス,*",
          "補助記号,句点,*,*,*,*,,*",
          "形容詞,非自立可能,*,*,形容詞,イー,イイ,1",
      },
      sys_offsets);
  std::string sys_tokens;
  for (uint32_t off : sys_offsets) {
    sys_tokens += BuildMecabToken(1, 1, 1, 100, off);
  }
  const std::string sys_da = BuildMecabDoubleArray({
      {"今日", (0 << 8) | 1},
      {"は", (1 << 8) | 1},
      {"天気", (2 << 8) | 1},
      {"です", (3 << 8) | 1},
      {"。", (4 << 8) | 1},
      {"いい", (5 << 8) | 1},
  });

  std::vector<uint32_t> unk_offsets;
  std::vector<std::string> unk_records(categories.size(),
                                       "名詞,普通名詞,一般,*,*,*,*");
  const std::string unk_features = BuildMecabFeatures(unk_records, unk_offsets);
  std::string unk_tokens;
  std::vector<std::pair<std::string, int>> unk_entries;
  for (size_t i = 0; i < categories.size(); ++i) {
    unk_tokens += BuildMecabToken(1, 1, 1, 5000, unk_offsets[i]);
    unk_entries.push_back({categories[i], static_cast<int>((i << 8) | 1)});
  }

  using ::litert::omni::tts::kokoro::AppendUint32;
  std::string w_idx;
  std::string w_key;
  AppendUint32(0, w_idx);
  w_key.append("天気");
  w_key.push_back('\0');
  AppendUint32(static_cast<uint32_t>(w_key.size()), w_idx);

  return BuildCjkBlob({
      {"sys-da", sys_da},
      {"sys-tok", sys_tokens},
      {"sys-feat", sys_features},
      {"matrix", BuildMecabConnectionMatrix(2, 2, {0, 0, 0, 0})},
      {"char", BuildMecabCharProperty(categories, char_records, 0x3100)},
      {"unk-da", BuildMecabDoubleArray(unk_entries)},
      {"unk-tok", unk_tokens},
      {"unk-feat", unk_features},
      {"w-idx", w_idx},
      {"w-key", w_key},
  });
}

TEST(PhonemizerTest, JapaneseWithoutLexiconFailsWithPrecondition) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("ja"));
  EXPECT_THAT(phonemizer->TextToIpa("今日はいい天気です。"),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(phonemizer->WordToIpa("今日"),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(PhonemizerTest, JapaneseWithBundledLexiconUsesJapaneseG2p) {
  const std::string ja_blob = BuildMinimalJapaneseBlobForPhonemizerTest();
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       CreateTestPhonemizer("ja", {}, "", ja_blob));

  ASSERT_OK_AND_ASSIGN(std::string ipa,
                       phonemizer->TextToIpa("今日はいい天気です。"));
  EXPECT_EQ(ipa, "kʲoː βa iː teŋkʲi desɨ.");

  ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                       phonemizer->TextToPhonemeIds("今日はいい天気です。"));
  EXPECT_GT(tokens.size(), 4);
  EXPECT_EQ(tokens.front(), 0);
  EXPECT_EQ(tokens.back(), 0);

  // Initializing under en-us with a ja-lexicon blob (containing sys-da and
  // w-idx) and switching to ja via SetLanguage retains japanese_g2p_ without
  // attempting to parse ja-lexicon as a Chinese lexicon.
  ASSERT_OK_AND_ASSIGN(auto switched_phonemizer,
                       CreateTestPhonemizer("en-us", {}, "", ja_blob));
  switched_phonemizer->SetLanguage("ja");
  EXPECT_THAT(switched_phonemizer->TextToIpa("今日はいい天気です。"),
              ::absl_testing::IsOkAndHolds("kʲoː βa iː teŋkʲi desɨ."));
}

TEST(PhonemizerTest, EnglishWithBundledLexiconUsesEnglishG2p) {
  const std::string en_blob = BuildMinimalEnglishBlobForPhonemizerTest();
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"geminitts", "dʒˈɛmənˌaɪ tˌiːtˌiːˈɛs"},
  };
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(/*data_dir=*/"", "en-us",
                                                custom_lexicon, "", en_blob));

  EXPECT_THAT(phonemizer->TextToIpa("Hello world!"),
              ::absl_testing::IsOkAndHolds("həlˈO wˈɜɹld!"));
  EXPECT_THAT(phonemizer->WordToIpa("water"),
              ::absl_testing::IsOkAndHolds("wˈɔɾəɹ"));
  EXPECT_THAT(phonemizer->TextToIpa("GeminiTTS"),
              ::absl_testing::IsOkAndHolds("ʤˈɛmənˌI tˌitˌiˈɛs"));

  phonemizer->SetLanguage("en-gb");
  EXPECT_THAT(phonemizer->TextToIpa("Hello world!"),
              ::absl_testing::IsOkAndHolds("həlˈQ wˈɜːld!"));
  EXPECT_THAT(phonemizer->WordToIpa("water"),
              ::absl_testing::IsOkAndHolds("wˈɔːtə"));
}

TEST(PhonemizerTest, SpanishUsesSpanishG2p) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"kokoro", "kokˈoɾo"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(/*data_dir=*/"", "es", custom_lexicon));
  EXPECT_EQ(phonemizer->Language(), "es");
  EXPECT_THAT(phonemizer->TextToIpa("¡Buenos días! ¿Cómo estás?"),
              ::absl_testing::IsOkAndHolds("bwˈenos ðˈias! kˈomo estˈas?"));
  EXPECT_THAT(phonemizer->WordToIpa("kokoro"),
              ::absl_testing::IsOkAndHolds("kokˈoɾo"));
}

TEST(PhonemizerTest, LongTextIsNotTruncated) {
  ASSERT_OK_AND_ASSIGN(auto phonemizer,
                       KokoroPhonemizer::Create(/*data_dir=*/"", "es"));
  const std::string sentence = "buenos días a todos los amigos de la casa ";
  ASSERT_OK_AND_ASSIGN(const std::vector<int> one,
                       phonemizer->TextToPhonemeIds(sentence));
  std::string text;
  for (int i = 0; i < 20; ++i) text += sentence;
  ASSERT_OK_AND_ASSIGN(const std::vector<int> many,
                       phonemizer->TextToPhonemeIds(text));

  // Long chunks are sliced to the model capacity downstream; the phonemizer
  // must keep every token instead of dropping the tail.
  EXPECT_GT(many.size(), 256);
  EXPECT_GT(many.size(), 15 * (one.size() - 2));
  EXPECT_EQ(many.front(), 0);  // BOS
  EXPECT_EQ(many.back(), 0);   // EOS
}

TEST(PhonemizerTest, HindiUsesHindiG2p) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"कोकोरो", "koːkˈoːɾoː"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(/*data_dir=*/"", "hi", custom_lexicon));
  EXPECT_EQ(phonemizer->Language(), "hi");
  EXPECT_THAT(phonemizer->TextToIpa("नमस्ते भारत।"),
              ::absl_testing::IsOkAndHolds("nəmˈʌsteː bʰˈaːɾət."));
  EXPECT_THAT(phonemizer->WordToIpa("कोकोरो"),
              ::absl_testing::IsOkAndHolds("koːkˈoːɾoː"));
}

TEST(PhonemizerTest, PortugueseUsesPortugueseG2p) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"kokoro", "kokˈoɾʊ"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(/*data_dir=*/"", "pt-br", custom_lexicon));
  EXPECT_EQ(phonemizer->Language(), "pt-br");
  EXPECT_THAT(
      phonemizer->TextToIpa("O pão estava muito gostoso."),
      ::absl_testing::IsOkAndHolds("ʊ pˈɐ̃ʊ̃ estˌavæ mwˈiŋtʊ ɡˌostˈozʊ."));
  EXPECT_THAT(phonemizer->WordToIpa("kokoro"),
              ::absl_testing::IsOkAndHolds("kokˈoɾʊ"));
}

TEST(PhonemizerTest, ItalianUsesItalianG2p) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"kokoro", "kokˈoro"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(/*data_dir=*/"", "it", custom_lexicon));
  EXPECT_EQ(phonemizer->Language(), "it");
  EXPECT_THAT(phonemizer->TextToIpa("Il treno per Milano parte dal binario 4."),
              ::absl_testing::IsOkAndHolds(
                  "il trˈɛno per milˈano pˈarte dal binˈario kwˈatːro."));
  EXPECT_THAT(phonemizer->WordToIpa("kokoro"),
              ::absl_testing::IsOkAndHolds("kokˈoro"));
}

TEST(PhonemizerTest, FrenchUsesFrenchG2p) {
  absl::flat_hash_map<std::string, std::string> custom_lexicon = {
      {"kokoro", "kokoʁˈo"},
  };
  ASSERT_OK_AND_ASSIGN(
      auto phonemizer,
      KokoroPhonemizer::Create(/*data_dir=*/"", "fr-fr", custom_lexicon));
  EXPECT_EQ(phonemizer->Language(), "fr-fr");
  EXPECT_THAT(
      phonemizer->TextToIpa(
          "Les poules du couvent couvent tranquillement leurs œufs."),
      ::absl_testing::IsOkAndHolds("le pˈul dy kuvˈɑ̃ kˈuv tʁɑ̃kilmˈɑ̃ lœʁz ˈø."));
  EXPECT_THAT(phonemizer->WordToIpa("kokoro"),
              ::absl_testing::IsOkAndHolds("kokoʁˈo"));
}

// A prompt with its expected IPA and Kokoro token IDs.
struct PhonemeTokensCase {
  std::string prompt;
  std::string ipa;
  std::vector<int> token_ids;
};

TEST(PhonemizerTest, PhonemeTokens) {
  // Prompts phonemized by EnglishG2p from real `en-lexicon` entries.
  ASSERT_OK_AND_ASSIGN(auto phonemizer, CreateTestPhonemizer("en-us"));
  const std::vector<PhonemeTokensCase> test_cases = {
      {
          .prompt = "Hello world.",
          .ipa = "həlˈO wˈɜɹld.",
          .token_ids = {0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 4,
                        0},
      },
      {
          .prompt = "Hello from LiteRT TTS Engine.",
          .ipa = "həlˈO fɹʌm lˌItˌɑɹtˈi ts ˈɛnʤən.",
          .token_ids = {0,  50,  83,  54, 156, 31, 16,  48, 123, 138, 55, 16,
                        54, 157, 25,  62, 157, 69, 123, 62, 156, 51,  16, 62,
                        61, 16,  156, 86, 56,  82, 83,  56, 4,   0},
      },
      {
          .prompt = "The quick brown fox jumps over the lazy dog.",
          .ipa = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.",
          .token_ids = {0,  81,  83,  16,  53,  65,  156, 102, 53, 16,
                        44, 123, 156, 39,  56,  16,  48,  156, 69, 53,
                        61, 16,  82,  156, 138, 55,  58,  61,  16, 156,
                        31, 64,  83,  123, 16,  81,  83,  16,  54, 156,
                        24, 68,  51,  16,  46,  156, 76,  92,  4,  0},
      },
      {
          .prompt = "Kokoro is an open-source text-to-speech model.",
          .ipa = "kˈɑkɔɹɑ ɪz ɐn ˈOpᵊnsˌɔɹs tˈɛksttəspˈiʧ mˈɑdᵊl.",
          .token_ids = {0,   53, 156, 69,  53, 76,  123, 69, 16, 102, 68, 16,
                        70,  56, 16,  156, 31, 58,  42,  56, 61, 157, 76, 123,
                        61,  16, 62,  156, 86, 53,  61,  62, 62, 83,  61, 58,
                        156, 51, 133, 16,  55, 156, 69,  46, 42, 54,  4,  0},
      },
  };
  for (const PhonemeTokensCase& test_case : test_cases) {
    EXPECT_THAT(phonemizer->TextToIpa(test_case.prompt),
                ::absl_testing::IsOkAndHolds(test_case.ipa))
        << test_case.prompt;
    EXPECT_THAT(phonemizer->TextToPhonemeIds(test_case.prompt),
                ::absl_testing::IsOkAndHolds(
                    ::testing::ElementsAreArray(test_case.token_ids)))
        << test_case.prompt;
  }
}

}  // namespace
}  // namespace litert::omni::tts
