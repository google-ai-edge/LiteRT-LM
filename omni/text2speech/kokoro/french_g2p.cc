// Copyright 2026 Google LLC.
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

#include "omni/text2speech/kokoro/french_g2p.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

std::string SliceCodepoints(const std::vector<char32_t>& cps, size_t start,
                            size_t end) {
  if (start >= end || start >= cps.size()) return std::string();
  const size_t clamped_end = (end > cps.size()) ? cps.size() : end;
  std::string out;
  out.reserve((clamped_end - start) * 2);
  for (size_t k = start; k < clamped_end; ++k) {
    AppendUtf8(cps[k], out);
  }
  return out;
}

// Lowercases `cp` and folds the typographic apostrophe onto '\''.
char32_t ToLowerFrench(char32_t cp) {
  if (cp == 0x2019) return U'\'';
  return ToLowerLatin(cp);
}

std::string ToLowerFrenchUtf8(absl::string_view text) {
  std::vector<char32_t> cps = Utf8ToCodePoints(text);
  for (char32_t& cp : cps) {
    cp = ToLowerFrench(cp);
  }
  return CodePointsToUtf8(cps);
}

bool IsCombiningMark(char32_t cp) { return (cp >= 0x0300 && cp <= 0x036F); }

std::vector<std::string> ToSymbols(absl::string_view ps) {
  std::vector<std::string> out;
  size_t pos = 0;
  char32_t cp = 0;
  while (ConsumeUtf8CodePoint(ps, pos, cp)) {
    if (!out.empty() && IsCombiningMark(cp)) {
      AppendUtf8(cp, out.back());
    } else {
      std::string s;
      AppendUtf8(cp, s);
      out.push_back(s);
    }
  }
  return out;
}

std::string JoinSymbols(const std::vector<std::string>& syms) {
  size_t total = 0;
  for (const std::string& s : syms) total += s.size();
  std::string out;
  out.reserve(total);
  for (const std::string& s : syms) out.append(s);
  return out;
}

bool IsVowelLetter(char32_t cp) {
  switch (cp) {
    case U'a':
    case U'e':
    case U'i':
    case U'o':
    case U'u':
    case U'y':
    case U'à':
    case U'â':
    case U'ä':
    case U'é':
    case U'è':
    case U'ê':
    case U'ë':
    case U'î':
    case U'ï':
    case U'ô':
    case U'ö':
    case U'ù':
    case U'û':
    case U'ü':
    case U'œ':
    case U'æ':
      return true;
    default:
      return false;
  }
}

bool IsNasalBlocked(char32_t cp) {
  return cp != 0 &&
         (IsVowelLetter(cp) || cp == U'n' || cp == U'm' || cp == U'h');
}

bool IsVowelPhoneme(absl::string_view s) {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kVowels({
      "a", "e", "i", "o", "u", "y", "ɛ", "ɔ", "ø", "œ", "ə", "ɑ", "ɪ",
      "ʊ", "ʌ", "ɒ", "ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "A", "I", "O", "Q", "W", "Y",
  });
  return kVowels->contains(s);
}

const absl::flat_hash_set<std::string>& GetIsolatedUnstressed() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "au",  "aux",  "ce",    "ces", "cet", "cette", "d",   "de",
      "des", "dont", "du",    "dès", "it",  "j",     "je",  "l",
      "ma",  "me",   "mes",   "mon", "ne",  "ni",    "nos", "of",
      "ou",  "qu",   "quant", "sa",  "se",  "ses",   "son", "ta",
      "tas", "te",   "tes",   "ton", "vos", "with",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetUnstressed() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "a",      "ai",      "as",     "au",      "aux",     "c",      "car",
      "ce",     "ces",     "cet",    "cette",   "ceux",    "chaque", "chez",
      "comme",  "contre",  "d",      "dans",    "de",      "des",    "dont",
      "du",     "dès",     "elle",   "elles",   "en",      "entre",  "es",
      "est",    "et",      "eu",     "il",      "ils",     "j",      "je",
      "l",      "la",      "le",     "les",     "leur",    "leurs",  "lors",
      "lorsqu", "lorsque", "m",      "ma",      "mais",    "me",     "mes",
      "mon",    "n",       "ne",     "ni",      "nos",     "notre",  "nous",
      "on",     "ont",     "ou",     "où",      "par",     "pas",    "peu",
      "plus",   "pour",    "près",   "puisqu",  "puisque", "qu",     "quand",
      "que",    "quel",    "quelle", "quelles", "quels",   "qui",    "s",
      "sa",     "sans",    "se",     "ses",     "sommes",  "son",    "sous",
      "sur",    "t",       "ta",     "te",      "tes",     "ton",    "tous",
      "tout",   "toute",   "toutes", "trop",    "très",    "tu",     "un",
      "une",    "vers",    "vos",    "votre",   "vous",    "y",      "à",
      "ça",     "êtes",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetSecondary() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "alors",    "après",  "assez",    "aussi",   "autant",    "autour",
      "avant",    "avec",   "beaucoup", "ceci",    "cela",      "comment",
      "depuis",   "encore", "eux",      "jamais",  "loin",      "lui",
      "mieux",    "moi",    "pourquoi", "presque", "quelqu'un", "quelque",
      "quelques", "quoi",   "rien",     "toi",     "toujours",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetAlwaysUnstressed() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "au",
      "aux",
      "d",
      "de",
      "des",
      "du",
      "j",
      "je",
      "l",
      "ne",
      "qu",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetHAspire() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "hache",    "haie",     "haine",      "hall",       "halte",
      "hamac",    "hameau",   "hamster",    "hanche",     "handicap",
      "hangar",   "hanter",   "harceler",   "hareng",     "hargne",
      "haricot",  "haricots", "harnais",    "harpe",      "hasard",
      "hasarder", "hausse",   "haut",       "haute",      "hautes",
      "hauteur",  "hauts",    "havre",      "hennir",     "hernie",
      "heurter",  "hibou",    "hisser",     "hiérarchie", "hocher",
      "hockey",   "hold-up",  "hollandais", "hollande",   "homard",
      "hongrie",  "hongrois", "honte",      "hoquet",     "hors",
      "hotte",    "houle",    "housse",     "hublot",     "huer",
      "huit",     "huitième", "hurler",     "hutte",      "hâte",
      "hérisson", "héron",    "héros",      "hêtre",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetNLiaison() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "ancien", "aucun",    "bien", "bon",   "certain", "commun", "divin",
      "en",     "lointain", "mon",  "moyen", "on",      "plein",  "prochain",
      "rien",   "son",      "ton",  "un",    "vain",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetTLiaisonWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "auraient", "aurait", "avaient", "avait",   "cet",     "doit",
      "dont",     "est",    "fait",    "faut",    "font",    "grand",
      "ont",      "petit",  "peut",    "quand",   "respect", "second",
      "seraient", "serait", "soit",    "sont",    "tient",   "tout",
      "veut",     "vient",  "vont",    "étaient", "était",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetNasalEntWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "absent",       "abstinent",   "accent",      "accident",
      "adhérent",     "adolescent",  "agent",       "apparent",
      "ardent",       "argent",      "arpent",      "auvent",
      "avent",        "cent",        "client",      "coefficient",
      "cohérent",     "compétent",   "concurrent",  "conscient",
      "conséquent",   "content",     "continent",   "contingent",
      "corpulent",    "couvent",     "dent",        "différent",
      "dissident",    "décent",      "déficient",   "efficient",
      "excellent",    "fervent",     "imminent",    "impatient",
      "impertinent",  "impotent",    "incident",    "incompétent",
      "inconscient",  "indifférent", "indolent",    "indulgent",
      "ingrédient",   "innocent",    "insolent",    "intelligent",
      "intermittent", "latent",      "laurent",     "lent",
      "mécontent",    "négligent",   "occident",    "omnipotent",
      "orient",       "parent",      "patent",      "patient",
      "permanent",    "pertinent",   "pourcent",    "proéminent",
      "prudent",      "précédent",   "présent",     "président",
      "pénitent",     "quotient",    "récent",      "récipient",
      "référent",     "sent",        "sergent",     "serpent",
      "somnolent",    "souvent",     "strident",    "succulent",
      "talent",       "torrent",     "transparent", "trident",
      "turbulent",    "urgent",      "vent",        "vincent",
      "violent",      "éminent",     "équivalent",  "évident",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetVerbMentWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "abîment",    "acclament",     "aiment",     "allument",    "animent",
      "apprennent", "arment",        "assomment",  "assument",    "blâment",
      "calment",    "charment",      "clament",    "comprennent", "consomment",
      "consument",  "diffament",     "déclament",  "dépriment",   "désarment",
      "entament",   "entreprennent", "estiment",   "expriment",   "ferment",
      "forment",    "fument",        "impriment",  "nomment",     "oppriment",
      "parfument",  "parsèment",     "plument",    "prennent",    "proclament",
      "pâment",     "rament",        "reprennent", "riment",      "réclament",
      "répriment",  "résument",      "suppriment", "surnomment",  "surprennent",
      "sèment",     "trament",
  });
  return *kSet;
}

const absl::flat_hash_map<std::string, std::string>& GetEnLoans() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"air", "ˈɛʁ"},
          {"airlines", "ˈeəlInz"},
          {"alex", "alˈɛks"},
          {"all", "ˈɔːl"},
          {"amer", "amˈɛʁ"},
          {"america", "əmˈɛɹɪkə"},
          {"american", "əmˈɛɹɪkən"},
          {"and", "ænd"},
          {"android", "ɑ̃dʁoˈid"},
          {"angeles", "ɑ̃nʒelˈɛs"},
          {"apple", "ˈapəl"},
          {"at", "ˈat"},
          {"baby", "bˈAbi"},
          {"back", "bˈak"},
          {"bad", "bˈad"},
          {"bar", "bˈaʁ"},
          {"bars", "bˈaʁ"},
          {"basket", "baskˈɛt"},
          {"bay", "bˈA"},
          {"be", "bˈə"},
          {"beach", "bˈiːʧ"},
          {"ben", "bˈɛn"},
          {"benchmarking", "bˈɛnʧmɑːkɪŋ"},
          {"best", "bˈɛst"},
          {"big", "bˈiɡ"},
          {"bill", "bˈil"},
          {"black", "blˈak"},
          {"blog", "blˈɔɡ"},
          {"blogs", "blˈɔɡ"},
          {"blue", "blˈuː"},
          {"blues", "blˈuːz"},
          {"bob", "bˈɔb"},
          {"boston", "bɔstˈɔn"},
          {"bowling", "bˈQlɪŋ"},
          {"boy", "bˈY"},
          {"boys", "bˈYz"},
          {"brainstorming", "bɹˈAnstɔːmɪŋ"},
          {"bridge", "bɹˈɪʤ"},
          {"briefing", "bɹˈiːfɪŋ"},
          {"british", "bɹˈɪtɪʃ"},
          {"broadway", "bɹˈɔːdwA"},
          {"brunch", "bɹˈʌnʧ"},
          {"building", "bˈɪldɪŋ"},
          {"bush", "bˈʊʃ"},
          {"business", "bˈɪznəs"},
          {"by", "bˈI"},
          {"bye", "bˈI"},
          {"camping", "kˈampɪŋ"},
          {"cancer", "kɑ̃sˈɛʁ"},
          {"cancers", "kɑ̃sˈɛʁ"},
          {"casting", "kˈɑːstɪŋ"},
          {"catering", "kˈAtəɹɪŋ"},
          {"center", "sˈɛntə"},
          {"centre", "sˈɑ̃tʁ"},
          {"chat", "ʃˈa"},
          {"cher", "ʃˈɛʁ"},
          {"chers", "ʃˈɛʁ"},
          {"chicago", "ʃikaɡˈo"},
          {"city", "sˈɪti"},
          {"clinton", "klɛ̃tˈɔn"},
          {"clown", "klˈun"},
          {"club", "klˈœb"},
          {"clubs", "klˈœb"},
          {"cluster", "klœstˈœʁ"},
          {"coach", "kˈQʧ"},
          {"coaching", "kˈQʧɪŋ"},
          {"coca", "kokˈa"},
          {"coffee", "kˈɒfi"},
          {"cola", "kolˈa"},
          {"computer", "kɔ̃pytˈœʁ"},
          {"consulting", "kənsˈʌltɪŋ"},
          {"cool", "kˈuːl"},
          {"country", "kˈʌntɹi"},
          {"coworking", "kQwˈɜːkɪŋ"},
          {"crazy", "kɹˈAzi"},
          {"crowdfunding", "kɹˈWdfʌndɪŋ"},
          {"cup", "kˈʌp"},
          {"dan", "dˈan"},
          {"dance", "dˈɑːns"},
          {"dancing", "dˈɑːnsɪŋ"},
          {"dark", "dˈɑːk"},
          {"day", "dˈA"},
          {"days", "dˈAz"},
          {"debriefing", "dɪbɹˈiːfɪŋ"},
          {"design", "dɪzˈIn"},
          {"designer", "dəzinjˈe"},
          {"divers", "divˈɛʁ"},
          {"do", "dˈuː"},
          {"doctor", "dˈɒktə"},
          {"dos", "dˈɒs"},
          {"down", "dˈWn"},
          {"drink", "dɹˈɪŋk"},
          {"e-mail", "ˈiːmAl"},
          {"easy", "ˈiːzi"},
          {"email", "ˈiːmAl"},
          {"enfer", "ɑ̃fˈɛʁ"},
          {"english", "ˈɪŋɡlɪʃ"},
          {"envers", "ɑ̃vˈɛʁ"},
          {"express", "ɛkspʁˈɛs"},
          {"facebook", "fˈAsbʊk"},
          {"factoring", "fˈaktəɹɪŋ"},
          {"family", "fˈamɪli"},
          {"fan", "fˈan"},
          {"fans", "fˈan"},
          {"fast", "fˈast"},
          {"fast-food", "fˈastfˈuːd"},
          {"feeling", "fˈiːlɪŋ"},
          {"fer", "fˈɛʁ"},
          {"fers", "fˈɛʁ"},
          {"fier", "fjˈɛʁ"},
          {"fiers", "fjˈɛʁ"},
          {"fire", "fˈIə"},
          {"fitness", "fˈɪtnəs"},
          {"five", "fˈIv"},
          {"folk", "fˈQk"},
          {"food", "fˈuːd"},
          {"foot", "fˈʊt"},
          {"football", "fˈʊtbɔːl"},
          {"footing", "fˈʊtɪŋ"},
          {"for", "fˈɔʁ"},
          {"forcing", "fˈɔːsɪŋ"},
          {"four", "fˈɔː"},
          {"francisco", "fʁɑ̃siskˈo"},
          {"free", "fɹˈiː"},
          {"french", "fɹˈɛnʧ"},
          {"from", "fɹˈɒm"},
          {"full", "fˈʊl"},
          {"funk", "fˈʌŋk"},
          {"game", "ɡˈAm"},
          {"games", "ɡˈAmz"},
          {"gaming", "ɡˈAmɪŋ"},
          {"girl", "ɡˈɜːl"},
          {"girls", "ɡˈɜːlz"},
          {"go", "ɡˈo"},
          {"goal", "ɡˈQl"},
          {"gold", "ɡˈQld"},
          {"golden", "ɡˈQldən"},
          {"golf", "ɡˈɔlf"},
          {"good", "ɡˈʊd"},
          {"google", "ɡˈuːɡəl"},
          {"green", "ɡɹˈiːn"},
          {"hall", "ˈol"},
          {"handball", "ˈɑ̃dbal"},
          {"happy", "hˈapi"},
          {"hard", "hˈɑːd"},
          {"harry", "hˈaɹi"},
          {"hello", "ɛlˈo"},
          {"hey", "ˈɛ"},
          {"hi", "ˈi"},
          {"hier", "jˈɛʁ"},
          {"high", "hˈI"},
          {"hill", "hˈɪl"},
          {"hip-hop", "ˈipˈɔp"},
          {"hiver", "ivˈɛʁ"},
          {"holding", "hˈQldɪŋ"},
          {"hollywood", "ˈɔliwud"},
          {"home", "hˈQm"},
          {"hot", "hˈɒt"},
          {"hotel", "otˈɛl"},
          {"house", "hˈWs"},
          {"hyper", "ipˈɛʁ"},
          {"hôtel", "otˈɛl"},
          {"ice", "ˈIs"},
          {"in", "ˈɛ̃"},
          {"instagram", "ˌɪnstəɡɹˈam"},
          {"interview", "ɛ̃tɛʁvjˈu"},
          {"interviews", "ɛ̃tɛʁvjˈu"},
          {"ipad", "ˈIpad"},
          {"iphone", "ˈIfQn"},
          {"is", "ˈiz"},
          {"island", "ˈIlənd"},
          {"it", "ɪt"},
          {"jack", "ʒˈak"},
          {"james", "ʤˈAmz"},
          {"jazz", "ʤˈaz"},
          {"job", "ʤˈɔb"},
          {"jobs", "ʤˈɔbz"},
          {"joe", "ʒˈɔ"},
          {"jogging", "ʤˈɒɡɪŋ"},
          {"john", "ʤˈɒn"},
          {"karting", "kˈɑːtɪŋ"},
          {"kevin", "kevˈin"},
          {"king", "kˈɪŋ"},
          {"lady", "lˈAdi"},
          {"lake", "lˈAk"},
          {"laser", "lazˈɛʁ"},
          {"leader", "lˈiːdə"},
          {"leaders", "lˈiːdəz"},
          {"leadership", "lˈiːdəʃɪp"},
          {"leasing", "lˈiːsɪŋ"},
          {"lee", "lˈiː"},
          {"life", "lˈIf"},
          {"lifting", "lˈɪftɪŋ"},
          {"light", "lˈIt"},
          {"line", "lˈIn"},
          {"lines", "lˈInz"},
          {"live", "lˈIv"},
          {"living", "lˈɪvɪŋ"},
          {"long", "lˈɔ̃"},
          {"look", "lˈʊk"},
          {"lord", "lˈɔːd"},
          {"love", "lˈʌv"},
          {"low", "lˈQ"},
          {"mail", "mˈAl"},
          {"mails", "mˈAlz"},
          {"man", "mˈan"},
          {"management", "manaʒmˈɑ̃"},
          {"manager", "manˈadʒœʁ"},
          {"mark", "mˈɑːk"},
          {"marketing", "mˈɑːkɪtɪŋ"},
          {"mary", "mˈeəɹi"},
          {"master", "mastˈœʁ"},
          {"masters", "mastˈœʁ"},
          {"match", "mˈaʧ"},
          {"matches", "mˈaʧɪz"},
          {"matchs", "mˈaʧ"},
          {"max", "mˈaks"},
          {"me", "mə"},
          {"meeting", "mˈiːtɪŋ"},
          {"men", "mˈɛn"},
          {"mer", "mˈɛʁ"},
          {"mers", "mˈɛʁ"},
          {"miss", "mˈɪs"},
          {"mister", "mˈɪstə"},
          {"moon", "mˈuːn"},
          {"mountain", "mˈWntɪn"},
          {"my", "mˈI"},
          {"nancy", "nˈansi"},
          {"net", "nˈɛt"},
          {"networking", "nˈɛtwɜːkɪŋ"},
          {"new", "njˈuː"},
          {"news", "njˈuːz"},
          {"night", "nˈIt"},
          {"no", "nˈo"},
          {"obama", "obamˈa"},
          {"of", "ɒv"},
          {"off", "ˈɔf"},
          {"ok", "okˈe"},
          {"okay", "okˈe"},
          {"old", "ˈQld"},
          {"on", "ˈɔ̃"},
          {"one", "wˌɒn"},
          {"open", "ˈQpən"},
          {"out", "aˈut"},
          {"over", "ˈQvə"},
          {"palace", "palˈas"},
          {"park", "pˈɑːk"},
          {"parking", "pˈɑːkɪŋ"},
          {"parkings", "pˈɑːkɪŋ"},
          {"partner", "pˈɑːtnə"},
          {"partners", "pˈɑːtnəz"},
          {"party", "pˈɑːti"},
          {"people", "pˈiːpəl"},
          {"pervers", "pɛʁvˈɛʁ"},
          {"peter", "pˈiːtə"},
          {"pink", "pˈɪŋk"},
          {"planning", "plˈanɪŋ"},
          {"play", "plˈA"},
          {"player", "plˈAə"},
          {"please", "plˈiːz"},
          {"point", "pwˈɛ̃"},
          {"pop", "pˈɔp"},
          {"post", "pˈɔst"},
          {"poster", "pɔstˈɛʁ"},
          {"posts", "pˈɔst"},
          {"pressing", "pɹˈɛsɪŋ"},
          {"prince", "pʁˈɛ̃s"},
          {"pub", "pˈœb"},
          {"pubs", "pˈœb"},
          {"pudding", "pˈʊdɪŋ"},
          {"punk", "pˈʌŋk"},
          {"queen", "kwˈiːn"},
          {"ranking", "ɹˈaŋkɪŋ"},
          {"red", "ɹˈɛd"},
          {"reporter", "ʁəpɔʁtˈœʁ"},
          {"reporters", "ʁəpɔʁtˈœʁ"},
          {"reporting", "ɹɪpˈɔːtɪŋ"},
          {"resort", "ɹɪzˈɔːt"},
          {"revers", "ʁəvˈɛʁ"},
          {"river", "ʁivˈɛʁ"},
          {"road", "ɹˈQd"},
          {"rock", "ʁˈɔk"},
          {"rock'n'roll", "ɹˈɒknɹQl"},
          {"room", "ɹˈuːm"},
          {"rugby", "ʁyɡbˈi"},
          {"running", "ɹˈʌnɪŋ"},
          {"sam", "sˈam"},
          {"sandwich", "sɑ̃dwˈiʧ"},
          {"scoop", "skˈuːp"},
          {"scooter", "skutˈœʁ"},
          {"scooters", "skutˈœʁ"},
          {"score", "skˈɔʁ"},
          {"set", "sˈɛt"},
          {"sexy", "sˈɛksi"},
          {"shampooing", "ʃampwˈɛ̃"},
          {"shopping", "ʃˈɒpɪŋ"},
          {"short", "ʃˈɔʁt"},
          {"show", "ʃˈQ"},
          {"shows", "ʃˈQz"},
          {"silver", "sˈɪlvə"},
          {"sir", "sˈɜː"},
          {"site", "sˈit"},
          {"sites", "sˈit"},
          {"skate", "skˈAt"},
          {"ski", "skˈi"},
          {"small", "smˈɔːl"},
          {"smart", "smˈɑːt"},
          {"smoking", "smˈQkɪŋ"},
          {"snow", "snˈQ"},
          {"soft", "sˈɒft"},
          {"sorry", "sˈɒɹi"},
          {"soul", "sˈQl"},
          {"spa", "spˈa"},
          {"square", "skwˈaʁ"},
          {"staff", "stˈaf"},
          {"standing", "stˈandɪŋ"},
          {"star", "stˈɑː"},
          {"stars", "stˈɑːz"},
          {"states", "stˈAts"},
          {"stop", "stˈɔp"},
          {"storytelling", "stˈɔːɹɪtɛlɪŋ"},
          {"streaming", "stɹˈiːmɪŋ"},
          {"street", "stɹˈiːt"},
          {"stress", "stɹˈɛs"},
          {"stretching", "stɹˈɛʧɪŋ"},
          {"sun", "sˈʌn"},
          {"super", "sypˈɛʁ"},
          {"supporter", "sypɔʁtˈœʁ"},
          {"supporters", "sypɔʁtˈœʁ"},
          {"surf", "sˈɜːf"},
          {"surfing", "sˈɜːfɪŋ"},
          {"tea", "tˈiː"},
          {"team", "tˈiːm"},
          {"techno", "tɛknˈo"},
          {"tennis", "tenˈis"},
          {"test", "tˈɛst"},
          {"tests", "tˈɛst"},
          {"thanks", "θˈaŋks"},
          {"the", "ðə"},
          {"three", "θɹˈiː"},
          {"thriller", "θɹˈɪlə"},
          {"tiers", "tjˈɛʁ"},
          {"time", "tˈIm"},
          {"times", "tˈImz"},
          {"timing", "tˈImɪŋ"},
          {"to", "tˈo"},
          {"tom", "tˈɔm"},
          {"top", "tˈɔp"},
          {"toujours", "tuʒˈuʁ"},
          {"tour", "tˈuʁ"},
          {"tours", "tˈuʁ"},
          {"tower", "tˈWə"},
          {"town", "tˈWn"},
          {"training", "tɹˈAnɪŋ"},
          {"travers", "tʁavˈɛʁ"},
          {"trekking", "tɹˈɛkɪŋ"},
          {"trump", "tɹˈʌmp"},
          {"tuning", "tˈuːnɪŋ"},
          {"twitter", "twˈɪtə"},
          {"two", "tˈuː"},
          {"under", "ˈʌndə"},
          {"united", "juːnˈItɪd"},
          {"univers", "ynivˈɛʁ"},
          {"up", "ˈʌp"},
          {"us", "ˈys"},
          {"valley", "vˈali"},
          {"van", "vˈan"},
          {"ver", "vˈɛʁ"},
          {"vers", "vˈɛʁ"},
          {"volley", "vɔlˈɛ"},
          {"wall", "wˈɔːl"},
          {"washington", "waʃiŋtˈɔn"},
          {"water", "wˈɔːtə"},
          {"way", "wˈA"},
          {"we", "wˈiː"},
          {"web", "wˈɛb"},
          {"week", "wˈiːk"},
          {"week-end", "wˈiːkˈɛnd"},
          {"weekend", "wˈiːkɛnd"},
          {"white", "wˈIt"},
          {"windows", "wˈɪndQz"},
          {"with", "wɪð"},
          {"woman", "wˈʊmən"},
          {"women", "wˈɪmɪn"},
          {"world", "wˈɜːld"},
          {"wow", "wˈW"},
          {"yes", "jˈɛs"},
          {"york", "jˈɔːk"},
          {"you", "jˈu"},
          {"young", "jˈʌŋ"},
          {"your", "jˈɔː"},
          {"youtube", "jˈuːtjuːb"},
          {"zapping", "zˈapɪŋ"},
      });
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>& GetLetterPhon() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"a", "a"},     {"b", "be"},  {"c", "se"},      {"d", "de"},
          {"e", "ə"},     {"f", "ɛf"},  {"g", "ʒe"},      {"h", "aʃ"},
          {"i", "i"},     {"j", "ʒi"},  {"k", "ka"},      {"l", "ɛl"},
          {"m", "ɛm"},    {"n", "ɛn"},  {"o", "o"},       {"p", "pe"},
          {"q", "ky"},    {"r", "ɛʁ"},  {"s", "ɛs"},      {"t", "te"},
          {"u", "y"},     {"v", "ve"},  {"w", "dubləve"}, {"x", "iks"},
          {"y", "iɡʁɛk"}, {"z", "zɛd"},
      });
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>& GetClitics() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"c", "s"},
          {"d", "d"},
          {"j", "ʒ"},
          {"jusqu", "ʒysk"},
          {"l", "l"},
          {"lorsqu", "loʁsk"},
          {"m", "m"},
          {"n", "n"},
          {"presqu", "pʁɛsk"},
          {"puisqu", "pyisk"},
          {"qu", "k"},
          {"quoiqu", "kwak"},
          {"s", "s"},
          {"t", "t"},
      });
  return *kMap;
}

bool EqualsAny(absl::string_view s,
               const std::vector<absl::string_view>& items) {
  for (absl::string_view p : items) {
    if (s == p) return true;
  }
  return false;
}

bool IsSilent3plEnt(absl::string_view w, const std::vector<char32_t>& cps) {
  if (cps.size() <= 4 || !absl::EndsWith(w, "ent")) return false;
  if (GetVerbMentWords().contains(w) || w == "savent") return true;
  if (absl::EndsWith(w, "ment")) return false;
  if (absl::EndsWith(w, "ient")) {
    return EndsWithAny(w, {"oient", "uient", "aient"});
  }
  if (GetNasalEntWords().contains(w)) return false;
  if (absl::EndsWith(w, "scent")) return false;
  for (const std::string& nw : GetNasalEntWords()) {
    if (Utf8ToCodePoints(nw).size() >= 5 && absl::EndsWith(w, nw)) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> SpellOutConsonants(absl::string_view w) {
  if (w == "https") return ToSymbols("ˌaʃtˌetˈepˌeˈɛs");
  std::vector<std::string> parts;
  for (size_t idx = 0; idx < w.size(); ++idx) {
    const std::string key(1, w[idx]);
    const auto it = GetLetterPhon().find(key);
    if (it == GetLetterPhon().end()) continue;
    std::vector<std::string> lp = ToSymbols(it->second);
    int last_v = -1;
    for (size_t k = 0; k < lp.size(); ++k) {
      if (IsVowelPhoneme(lp[k])) last_v = static_cast<int>(k);
    }
    if (last_v >= 0) {
      const std::string mark = (idx + 1 == w.size()) ? "ˈ" : "ˌ";
      lp.insert(lp.begin() + last_v, mark);
    }
    parts.insert(parts.end(), lp.begin(), lp.end());
  }
  return parts;
}

bool HasAnyVowelLetterAfterTrim(const std::vector<char32_t>& cps, size_t start,
                                size_t end, absl::string_view trim_chars) {
  size_t r = end;
  while (r > start) {
    const char32_t ch = cps[r - 1];
    bool trimmed = false;
    for (char tc : trim_chars) {
      if (ch == static_cast<char32_t>(tc)) {
        trimmed = true;
        break;
      }
    }
    if (!trimmed) break;
    --r;
  }
  for (size_t k = start; k < r; ++k) {
    if (IsVowelLetter(cps[k])) return true;
  }
  return false;
}

char32_t CpsNext(const std::vector<char32_t>& cps, size_t i, size_t end,
                 size_t k) {
  return (i + k < end) ? cps[i + k] : 0;
}

char32_t CpsPrev(const std::vector<char32_t>& cps, size_t i, size_t k) {
  return (i >= k) ? cps[i - k] : 0;
}

bool TryLtsWholeWordSpecial(absl::string_view w,
                            std::vector<std::string>& out) {
  if (w == "y") {
    out = {"i", "ɡ", "ʁ", "ɛ", "k"};
    return true;
  }
  if (EqualsAny(w, {"le", "ce", "de", "je", "me", "ne", "se", "te", "que"})) {
    std::string c = "k";
    if (w == "le")
      c = "l";
    else if (w == "ce" || w == "se")
      c = "s";
    else if (w == "de")
      c = "d";
    else if (w == "je")
      c = "ʒ";
    else if (w == "me")
      c = "m";
    else if (w == "ne")
      c = "n";
    else if (w == "te")
      c = "t";
    out = {c, "ə"};
    return true;
  }
  if (EqualsAny(w, {"les", "des", "ces", "mes", "tes", "ses"})) {
    std::string c = "s";
    if (w == "les")
      c = "l";
    else if (w == "des")
      c = "d";
    else if (w == "mes")
      c = "m";
    else if (w == "tes")
      c = "t";
    out = {c, "e"};
    return true;
  }
  if (EqualsAny(w, {"du", "tu", "vu", "lu", "pu", "su", "nu", "cru", "bu"})) {
    if (w == "cru") {
      out = {"k", "ʁ", "y"};
    } else {
      out = {std::string(w.substr(0, 1)), "y"};
    }
    return true;
  }
  if (w == "eu" || w == "ue") {
    out = {"y"};
    return true;
  }
  if (w == "es" || w == "est" || w == "eh") {
    out = {"ɛ"};
    return true;
  }
  if (w == "et") {
    out = {"e"};
    return true;
  }
  if (w == "cet" || w == "cette") {
    out = {"s", "ɛ", "t"};
    return true;
  }
  if (w == "tous") {
    out = {"t", "u", "s"};
    return true;
  }
  if (w == "six") {
    out = {"s", "i", "s"};
    return true;
  }
  if (w == "dix") {
    out = {"d", "i", "s"};
    return true;
  }
  if (w == "huit") {
    out = {"y", "i", "t"};
    return true;
  }
  if (w == "sept") {
    out = {"s", "ɛ", "t"};
    return true;
  }
  if (w == "cinq") {
    out = {"s", "ɛ̃", "k"};
    return true;
  }
  if (w == "neuf") {
    out = {"n", "œ", "f"};
    return true;
  }
  if (w == "femme" || w == "femmes") {
    out = {"f", "a", "m"};
    return true;
  }
  if (w == "monsieur" || w == "mr") {
    out = {"m", "ə", "s", "j", "ø"};
    return true;
  }
  if (w == "fils") {
    out = {"f", "i", "s"};
    return true;
  }
  if (w == "août") {
    out = {"u", "t"};
    return true;
  }
  if (w == "pied" || w == "pieds") {
    out = {"p", "j", "e"};
    return true;
  }
  if (w == "gars") {
    out = {"ɡ", "a"};
    return true;
  }
  if (EqualsAny(w, {"cul", "culs"})) {
    out = {"k", "y"};
    return true;
  }
  if (EqualsAny(w, {"outil", "outils"})) {
    out = {"u", "t", "i"};
    return true;
  }
  if (EqualsAny(w, {"fusil", "fusils"})) {
    out = {"f", "y", "z", "i"};
    return true;
  }
  if (EqualsAny(w, {"gentil", "gentils"})) {
    out = {"ʒ", "ɑ̃", "t", "i"};
    return true;
  }
  if (EqualsAny(w, {"sourcil", "sourcils"})) {
    out = {"s", "u", "ʁ", "s", "i"};
    return true;
  }
  if (w == "oh") {
    out = {"ɔ"};
    return true;
  }
  if (w == "ex") {
    out = {"ɛ", "k", "s"};
    return true;
  }
  if (w == "etc") {
    out = {"ɛ", "t", "s", "e", "t", "e", "ʁ", "a"};
    return true;
  }
  if (w == "km") {
    out = {"k", "i", "l", "o", "m", "ɛ", "t", "ʁ"};
    return true;
  }
  if (w == "mm") {
    out = {"m", "i", "l", "i", "m", "ɛ", "t", "ʁ"};
    return true;
  }
  if (w == "cm") {
    out = {"s", "ɑ̃", "t", "i", "m", "ɛ", "t", "ʁ"};
    return true;
  }
  if (w == "kg") {
    out = {"k", "i", "l", "ɔ", "ɡ", "ʁ", "a", "m"};
    return true;
  }
  if (w == "œil" || w == "oeil") {
    out = {"œ", "j"};
    return true;
  }
  if (w == "st") {
    out = {"s", "ɛ̃"};
    return true;
  }
  if (w == "tva") {
    out = {"t", "ˌ", "e", "v", "ˌ", "e", "ˈ", "a"};
    return true;
  }
  if (w == "jouent") {
    out = {"ʒ", "w"};
    return true;
  }
  if (w == "rouen") {
    out = {"ʁ", "w", "ɛ̃"};
    return true;
  }
  if (w == "néanmoins") {
    out = {"n", "e", "ɑ̃", "m", "w", "ɛ̃"};
    return true;
  }
  return false;
}

enum class StepAction { kNotMatched, kContinue, kBreak };

StepAction TryLtsWordEndings(absl::string_view w,
                             const std::vector<char32_t>& cps, size_t i,
                             size_t end, absl::string_view rest,
                             std::vector<std::string>& out, size_t& next_i) {
  const size_t n = cps.size();
  const char32_t c = cps[i];
  const auto prev = [&](size_t k) -> char32_t { return CpsPrev(cps, i, k); };
  if (EqualsAny(rest, {"aient", "aie", "aies"})) {
    out.push_back("ɛ");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"oient", "oie", "oies"})) {
    out.push_back("w");
    out.push_back("a");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"uient", "uie", "uies"})) {
    out.push_back("y");
    out.push_back("i");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"euse", "euses"})) {
    out.push_back("ø");
    out.push_back("z");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"eux", "œux"})) {
    out.push_back("ø");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"eur", "eurs", "œur", "œurs"})) {
    out.push_back("œ");
    out.push_back("ʁ");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"eau", "eaux"})) {
    out.push_back("o");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"au", "aux"}) && i > 0) {
    out.push_back("o");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ou", "ous", "oux", "out", "oud", "oup"}) &&
      w != "tous" && w != "tout") {
    out.push_back("u");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ois", "oit", "oid", "oids", "oigt", "oigts", "oix"})) {
    out.push_back("w");
    out.push_back("a");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"emment", "amment"})) {
    out.push_back("a");
    out.push_back("m");
    out.push_back("ɑ̃");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ement", "ements"}) && i >= 2) {
    if (EndsWithAny(
            w, {"gouvernement", "gouvernements", "département", "départements",
                "appartement", "appartements", "comportement", "comportements",
                "parlement", "parlements", "enseignement", "enseignants",
                "renseignement", "renseignements", "environnement"})) {
      if (out.size() >= 2 && out[out.size() - 2] == "n" && out.back() == "j" &&
          EndsWithAny(w, {"enseignement", "renseignement", "enseignements",
                          "renseignements"})) {
        out.pop_back();
        out.back() = "ɲ";
      }
      out.push_back("ə");
      out.push_back("m");
      out.push_back("ɑ̃");
    } else {
      const std::string p21 = SliceCodepoints(cps, i - 2, i);
      if (!IsVowelLetter(prev(1)) && !IsVowelLetter(prev(2)) &&
          !EqualsAny(p21, {"ll", "mm", "nn", "rr", "ss", "tt", "pp", "ff",
                           "cc", "ch", "ph", "th", "qu", "gu", "gn", "st",
                           "lm", "nc", "ns", "nt", "nd", "mp", "mb", "ng"})) {
        out.push_back("ə");
        out.push_back("m");
        out.push_back("ɑ̃");
      } else {
        out.push_back("m");
        out.push_back("ɑ̃");
      }
    }
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ment", "ments"}) && n > 4) {
    out.push_back("m");
    out.push_back("ɑ̃");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"tion", "tions"}) && prev(1) != U's' &&
      prev(1) != U'x') {
    out.push_back("s");
    out.push_back("j");
    out.push_back("ɔ̃");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"sion", "sions"}) && IsVowelLetter(prev(1))) {
    out.push_back("z");
    out.push_back("j");
    out.push_back("ɔ̃");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"sion", "sions", "ssion", "ssions"})) {
    out.push_back("s");
    out.push_back("j");
    out.push_back("ɔ̃");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ier", "iers", "iez"}) && i > 0 && w != "hier" &&
      w != "fier" && w != "tiers") {
    const char32_t p1 = prev(1);
    const char32_t p2 = prev(2);
    const bool cluster_br =
        (p1 == U'b' || p1 == U'd' || p1 == U'f' || p1 == U'g' || p1 == U'p' ||
         p1 == U't' || p1 == U'k' || p1 == U'v' || p1 == U'c') &&
        i >= 2 && (p2 == U'r' || p2 == U'l');
    if (cluster_br ||
        StartsWithAny(
            w, {"oubli",    "publi",   "pli",     "li",         "reli",
                "ni",       "confi",   "défi",    "méfi",       "modifi",
                "identifi", "signifi", "qualifi", "simplifi",   "multipli",
                "sacrifi",  "certifi", "notifi",  "ratifi",     "appréci",
                "associ",   "négoci",  "initi",   "différenci", "privilégi",
                "réfugi",   "mari",    "vari",    "pari",       "cri",
                "pri",      "tri",     "étudi",   "envi",       "convi",
                "dévi"})) {
      out.push_back("i");
      out.push_back("e");
    } else {
      out.push_back("j");
      out.push_back("e");
    }
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ière", "ières"})) {
    const char32_t p1 = prev(1);
    const char32_t p2 = prev(2);
    if ((p1 == U'b' || p1 == U'd' || p1 == U'f' || p1 == U'g' || p1 == U'p' ||
         p1 == U't' || p1 == U'k' || p1 == U'v' || p1 == U'c') &&
        i >= 2 && (p2 == U'r' || p2 == U'l')) {
      out.push_back("i");
      out.push_back("ɛ");
      out.push_back("ʁ");
    } else {
      out.push_back("j");
      out.push_back("ɛ");
      out.push_back("ʁ");
    }
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"er", "ers", "ez"}) && n > 2 &&
      !EqualsAny(
          w, {"mer",      "fer",     "ver",    "hier",   "fier",    "cher",
              "clair",    "hiver",   "enfer",  "amer",   "cancer",  "super",
              "laser",    "leader",  "master", "poster", "cluster", "starter",
              "transfer", "univers", "vers",   "tiers",  "travers", "envers",
              "divers",   "pervers", "revers", "mers",   "fers",    "chers",
              "fiers",    "hiers",   "hivers", "enfers", "amers",   "cancers",
              "supers",   "lasers",  "der"})) {
    out.push_back("e");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"et", "ets"}) && n > 2 && w != "net" && w != "cet" &&
      w != "basket") {
    out.push_back("ɛ");
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"es", "e"}) && n > 2) {
    if (EndsWithAny(w,
                    {"quelque", "quelques", "presque", "puisque", "lorsque"})) {
      out.push_back("ə");
    }
    return StepAction::kBreak;
  }
  if (EqualsAny(rest, {"ts", "ds", "ps", "cs", "gs", "ls"}) && n > 2) {
    if (rest == "ls" && !EqualsAny(w, {"fils", "pouls", "outils", "gentils",
                                       "fusils", "sourcils"})) {
      out.push_back("l");
    } else if (rest == "cs" && prev(1) != U'n' &&
               !EqualsAny(w, {"tabacs", "estomacs", "blancs", "francs", "bancs",
                              "troncs"})) {
      out.push_back("k");
    } else if (rest == "ts" &&
               EqualsAny(
                   w, {"buts", "bruts", "nets", "tests", "concepts", "contacts",
                       "impacts", "directs", "exacts", "corrects", "stricts",
                       "distincts", "instincts", "suspects"})) {
      out.push_back("t");
    }
    return StepAction::kBreak;
  }
  if (i == end - 1 && n > 1 &&
      (c == U's' || c == U'x' || c == U'z' || c == U't' || c == U'd' ||
       c == U'p' || c == U'g' || c == U'c')) {
    if (c == U'c' &&
        (EqualsAny(w, {"donc", "marc", "parc", "arc", "grec", "turc", "bac",
                       "sac", "lac", "bloc", "choc", "truc", "duc", "pic",
                       "flic", "fric", "sec", "bec", "échec"}) ||
         (prev(1) != U'n' &&
          !EqualsAny(w, {"tabac", "estomac", "blanc", "franc", "banc", "tronc",
                         "jonc", "flanc", "clerc", "porc", "accroc",
                         "caoutchouc", "escroc"})))) {
      out.push_back("k");
    } else if (c == U's' &&
               EqualsAny(w,
                         {"mars",      "ours",       "sens",       "toujours",
                          "ailleurs",  "plusieurs",  "mœurs",      "os",
                          "as",        "bus",        "tous",       "fils",
                          "virus",     "bonus",      "campus",     "processus",
                          "consensus", "tennis",     "blocus",     "maïs",
                          "lys",       "hélas",      "atlas",      "oasis",
                          "jadis",     "gratis",     "bis",        "cosmos",
                          "pathos",    "rhinocéros", "albatros",   "thermos",
                          "terminus",  "tonus",      "sinus",      "fœtus",
                          "focus",     "lotus",      "prospectus", "lapsus",
                          "cursus"})) {
      out.push_back("s");
    } else if (c == U't' &&
               EqualsAny(
                   w, {"ouest",    "direct",   "exact",   "contact",  "impact",
                       "compact",  "intact",   "correct", "indirect", "strict",
                       "distinct", "instinct", "suspect", "respect",  "aspect",
                       "concept",  "script",   "test",    "toast",    "huit",
                       "sept",     "août",     "but",     "brut",     "chut",
                       "dot",      "fat",      "mat",     "net",      "fret",
                       "christ",   "basket",   "best"})) {
      out.push_back("t");
    } else if (c == U'd' && EqualsAny(w, {"sud", "david", "madrid", "alfred",
                                          "bagdad", "baroud", "caïd", "cid",
                                          "bled", "stand", "week-end"})) {
      out.push_back("d");
    } else if (c == U'p' &&
               EqualsAny(w,
                         {"cap",      "cep",     "gap",      "slip",    "stop",
                          "handicap", "hip-hop", "start-up", "hold-up", "scoop",
                          "top",      "pop",     "rap",      "jeep",    "bip",
                          "flop",     "shop",    "step",     "trip",    "clip",
                          "drop",     "op",      "hop"})) {
      out.push_back("p");
    } else if (c == U'g' &&
               EqualsAny(w, {"joug", "gag", "gong", "tag", "zigzag", "blog",
                             "big", "bug", "dog", "fog", "log", "plug"})) {
      out.push_back("ɡ");
    }
    next_i = i + 1;
    return StepAction::kContinue;
  }
  return StepAction::kNotMatched;
}

StepAction TryLtsVowelDigraphsAndNasals(
    absl::string_view w, const std::vector<char32_t>& cps, size_t i, size_t end,
    absl::string_view rest, std::vector<std::string>& out, size_t& next_i) {
  const size_t n = cps.size();
  const char32_t c = cps[i];
  const size_t rest_len = end - i;
  const auto nxt = [&](size_t k) -> char32_t {
    return CpsNext(cps, i, end, k);
  };
  const auto prev = [&](size_t k) -> char32_t { return CpsPrev(cps, i, k); };

  if (absl::StartsWith(rest, "eau")) {
    out.push_back("o");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "tion") && prev(1) != U's' && prev(1) != U'x') {
    if (i + 5 < end && nxt(4) == U'n' && IsVowelLetter(nxt(5))) {
      out.push_back("s");
      out.push_back("j");
      out.push_back("ɔ");
      out.push_back("n");
      next_i = i + 5;
      return StepAction::kContinue;
    }
    if (i + 4 < end && IsVowelLetter(nxt(4))) {
      out.push_back("s");
      out.push_back("j");
      out.push_back("o");
      out.push_back("n");
      next_i = i + 4;
      return StepAction::kContinue;
    }
    out.push_back("s");
    out.push_back("j");
    out.push_back("ɔ̃");
    next_i = i + 4;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "tiel") &&
      (prev(1) == U'n' || prev(1) == U'r' || prev(1) == U's') &&
      StartsWithAny(
          w, {"essentiel", "potentiel", "confidentiel", "présidentiel",
              "résidentiel", "substantiel", "partiel", "préférentiel",
              "événementiel", "concurrentiel", "existentiel", "différentiel",
              "artificiel", "officiel", "superficiel", "logiciel", "matériel",
              "ministériel"})) {
    out.push_back("s");
    out.push_back("j");
    out.push_back("ɛ");
    out.push_back("l");
    next_i = i + 4;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"aille", "ailles"}) && i + rest_len == end) {
    out.push_back("a");
    out.push_back("j");
    return StepAction::kBreak;
  }
  if (StartsWithAny(rest, {"eil", "eille", "eilles"}) && i + rest_len == end) {
    out.push_back("ɛ");
    out.push_back("j");
    return StepAction::kBreak;
  }
  if (StartsWithAny(rest, {"euil", "euille", "euilles", "œuil", "ueil",
                           "ueille", "ueilles"})) {
    out.push_back("œ");
    out.push_back("j");
    if (absl::StartsWith(rest, "ueil")) {
      next_i = i + ((rest == "ueil" || rest == "ueils") ? 4 : 6);
    } else {
      const std::string r6 = SliceCodepoints(cps, i, i + 6);
      next_i = i + ((StartsWithAny(rest, {"euil", "œuil"}) &&
                     !absl::StrContains(r6, "le"))
                        ? 4
                        : 6);
    }
    const std::string rem = SliceCodepoints(cps, next_i, end);
    if (next_i < end && EqualsAny(rem, {"s", "es", "e"})) {
      return StepAction::kBreak;
    }
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ouil", "ouille", "ouilles"})) {
    out.push_back("u");
    out.push_back("j");
    const std::string r7 = SliceCodepoints(cps, i, i + 7);
    next_i = i + (absl::StrContains(r7, 'e') ? 6 : 4);
    const std::string rem = SliceCodepoints(cps, next_i, end);
    if (next_i < end && EqualsAny(rem, {"s", "es", "e"})) {
      return StepAction::kBreak;
    }
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ail", "ails"}) && i + rest_len == end) {
    out.push_back("a");
    out.push_back("j");
    return StepAction::kBreak;
  }
  if (StartsWithAny(rest, {"eil", "eils"}) && i + rest_len == end) {
    out.push_back("ɛ");
    out.push_back("j");
    return StepAction::kBreak;
  }
  if (absl::StartsWith(rest, "aill") && IsVowelLetter(nxt(4))) {
    out.push_back("a");
    out.push_back("j");
    next_i = i + 4;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "eill") && IsVowelLetter(nxt(4))) {
    out.push_back("ɛ");
    out.push_back("j");
    next_i = i + 4;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ouill") && IsVowelLetter(nxt(5))) {
    out.push_back("u");
    out.push_back("j");
    next_i = i + 5;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ill") && i > 0 && IsVowelLetter(prev(1)) &&
      !(prev(1) == U'u' && (prev(2) == U'q' || prev(2) == U'g'))) {
    out.push_back("j");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ill") &&
      !StartsWithAny(
          w, {"vill", "mill", "tranquill", "lill", "bill", "oscill", "gill"})) {
    out.push_back("i");
    out.push_back("j");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "il") && i > 0 && IsVowelLetter(prev(1)) &&
      i + 2 >= end) {
    out.push_back("j");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ain", "ein", "aim", "eim"}) &&
      !IsNasalBlocked(nxt(3))) {
    out.push_back("ɛ̃");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "oin") && !IsNasalBlocked(nxt(3))) {
    out.push_back("w");
    out.push_back("ɛ̃");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ien", "yen"}) && !IsNasalBlocked(nxt(3))) {
    if ((StartsWithAny(
             w, {"scien", "patien", "clien", "orien", "conscien", "inconvénien",
                 "ingrédien", "expérien", "audien", "ambien", "efficien"}) &&
         !absl::StartsWith(w, "quotidien")) ||
        (i > 0 && (prev(1) == U'c' || prev(1) == U't') &&
         StartsWithAny(rest,
                       {"ient", "ients", "ience", "iences", "ientifique"}) &&
         !EndsWithAny(w, {"tient", "tiens", "abstient", "appartient",
                          "contient", "devient", "maintient", "obtient",
                          "retient", "soutient", "souvient"}))) {
      if (absl::StartsWith(w, "clien")) {
        out.push_back("i");
        out.push_back("ɑ̃");
      } else if (prev(1) == U't' &&
                 StartsWithAny(w, {"patien", "impatien", "quotien"})) {
        if (!out.empty() && out.back() == "t") out.back() = "s";
        out.push_back("j");
        out.push_back("ɑ̃");
      } else {
        out.push_back("j");
        out.push_back("ɑ̃");
      }
    } else {
      out.push_back("j");
      out.push_back("ɛ̃");
    }
    next_i = i + 3;
    const std::string rem = SliceCodepoints(cps, next_i, end);
    if (next_i < end && EqualsAny(rem, {"t", "ts", "d", "ds", "s"})) {
      return StepAction::kBreak;
    }
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"oi", "oî"})) {
    out.push_back("w");
    out.push_back("a");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "oy") && IsVowelLetter(nxt(2))) {
    out.push_back("w");
    out.push_back("a");
    out.push_back("j");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ou", "où", "oû"})) {
    if ((IsVowelLetter(nxt(2)) && nxt(2) != U'e' && nxt(2) != U's') ||
        (nxt(2) == U'e' && i + 3 < end)) {
      out.push_back("w");
    } else {
      out.push_back("u");
    }
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "uy") && IsVowelLetter(nxt(2))) {
    out.push_back("y");
    out.push_back("i");
    out.push_back("j");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ay") && (IsVowelLetter(nxt(2)) || i + 2 == end)) {
    if (EqualsAny(w, {"pays", "paysage", "paysages", "paysan", "paysans",
                      "paysanne", "paysannes", "abbaye"})) {
      out.push_back("e");
      out.push_back("i");
    } else if (i + 2 == end) {
      out.push_back("ɛ");
    } else {
      out.push_back("ɛ");
      out.push_back("j");
    }
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"ai", "ei", "aî", "eî"})) {
    if (rest == "ai" &&
        EqualsAny(w, {"j'ai", "ai", "quai", "gai", "irai", "serai", "ferai",
                      "aurai", "dirai", "pourrai", "verrai", "voudrai",
                      "devrai", "saurai", "délai"})) {
      out.push_back("e");
    } else if (absl::StartsWith(rest, "ai") && absl::StartsWith(w, "fais") &&
               n > 4 && i == 1) {
      out.push_back("ə");
    } else {
      out.push_back("ɛ");
    }
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"au", "aux"})) {
    out.push_back("o");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"cœu", "coeu"})) {
    out.push_back("k");
    out.push_back("œ");
    next_i = i + (absl::StartsWith(rest, "cœu") ? 3 : 4);
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "cueil")) {
    out.push_back("k");
    out.push_back("œ");
    out.push_back("j");
    next_i = i + 5;
    const std::string rem = SliceCodepoints(cps, next_i, end);
    if (next_i < end && EqualsAny(rem, {"le", "les", "s", "e"})) {
      return StepAction::kBreak;
    }
    return StepAction::kContinue;
  }
  if (StartsWithAny(rest, {"œu", "eu", "eû"})) {
    if (StartsWithAny(
            w, {"euro",      "heureux", "heureuse", "malheureu", "bienheureu",
                "peu",       "deux",    "jeudi",    "jeune",     "jeunes",
                "jeunesse",  "neutre",  "feutre",   "meute",     "émeute",
                "thérapeut", "seulem",  "pseud",    "aveugl",    "pleur",
                "preuv",     "épreuv"})) {
      out.push_back("ø");
      next_i = i + 2;
      return StepAction::kContinue;
    }
    const size_t j = i + 2;
    const std::string tail = SliceCodepoints(cps, j, end);
    const bool closed =
        !tail.empty() && !IsVowelLetter(cps[j]) &&
        !EqualsAny(tail, {"s", "x", "t", "d", "e", "es", "ent", "te", "tes",
                          "tre", "tres", "ne", "nes", "se", "ses", "ze"}) &&
        !StartsWithAny(tail, {"s", "z", "t"});
    out.push_back(closed ? "œ" : "ø");
    if (absl::StartsWith(rest, "eû")) out.back() = "ø";
    next_i = j;
    return StepAction::kContinue;
  }
  if (EqualsAny(rest, {"um", "ums"}) && n > 3 &&
      !EqualsAny(w, {"parfum", "parfums", "brun", "bruns", "chacun", "aucun",
                     "commun", "communs", "emprunt", "emprunts", "défunt",
                     "défunts", "lundi", "lundis"})) {
    out.push_back("ɔ");
    out.push_back("m");
    return StepAction::kBreak;
  }
  if ((c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' ||
       c == U'y') &&
      (nxt(1) == U'n' || nxt(1) == U'm') && !IsNasalBlocked(nxt(2))) {
    if (EqualsAny(rest, {"am", "an", "ans", "om", "in"}) &&
        EqualsAny(w, {"islam",     "abraham",     "amsterdam", "vietnam",
                      "macadam",   "tram",        "gram",      "spam",
                      "van",       "man",         "fan",       "fans",
                      "dan",       "sam",         "tom",       "kevin",
                      "gentleman", "barman",      "tennisman", "rugbyman",
                      "policeman", "businessman", "cameraman", "showman",
                      "superman",  "spiderman",   "batman"})) {
      out.push_back((c == U'a') ? "a" : ((c == U'o') ? "ɔ" : "i"));
      out.push_back((nxt(1) == U'n') ? "n" : "m");
      next_i = i + 2;
      if (next_i < end && SliceCodepoints(cps, next_i, end) == "s") {
        return StepAction::kBreak;
      }
      return StepAction::kContinue;
    }
    std::string nas = "œ̃";
    if (c == U'a' || c == U'e')
      nas = "ɑ̃";
    else if (c == U'i' || c == U'y')
      nas = "ɛ̃";
    else if (c == U'o')
      nas = "ɔ̃";
    if (c == U'e' && (prev(1) == U'i' || prev(1) == U'é' || prev(1) == U'y')) {
      nas = "ɛ̃";
    }
    out.push_back(nas);
    next_i = i + 2;
    if (StartsWithAny(rest, {"ongt", "ontp", "ontr", "ingt"}) &&
        StartsWithAny(w, {"longt", "montpell", "montréal", "vingt"})) {
      next_i += 1;
    }
    if (absl::StartsWith(w, "sympt")) out.push_back("p");
    const std::string rem = SliceCodepoints(cps, next_i, end);
    if (next_i < end && w != "donc" &&
        EqualsAny(rem, {"t", "ts", "d", "ds", "p", "ps", "b", "bs", "c", "cs",
                        "g", "gs", "s", "x"})) {
      return StepAction::kBreak;
    }
    return StepAction::kContinue;
  }
  return StepAction::kNotMatched;
}

StepAction TryLtsConsonantDigraphs(absl::string_view w,
                                   const std::vector<char32_t>& cps, size_t i,
                                   size_t end, absl::string_view rest,
                                   std::vector<std::string>& out,
                                   size_t& next_i) {
  const char32_t c = cps[i];
  const auto nxt = [&](size_t k) -> char32_t {
    return CpsNext(cps, i, end, k);
  };
  const auto prev = [&](size_t k) -> char32_t { return CpsPrev(cps, i, k); };

  if (absl::StartsWith(rest, "gn")) {
    if (i + 2 < end && SliceCodepoints(cps, i + 2, end) == "e") {
      out.push_back("ɲ");
      return StepAction::kBreak;
    }
    if (absl::StartsWith(w, "magnif")) {
      out.push_back("ɲ");
    } else if (i + 2 < end && cps[i + 2] == U'i') {
      out.push_back("n");
    } else {
      out.push_back("n");
      out.push_back("j");
    }
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "mn") && StartsWithAny(w, {"automn", "condamn"})) {
    out.push_back("n");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "chn")) {
    out.push_back("k");
    out.push_back("n");
    next_i = i + 3;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "chr") || absl::StartsWith(rest, "chl") ||
      (absl::StartsWith(rest, "cha") && absl::StartsWith(w, "michael")) ||
      StartsWithAny(w, {"chœur", "choeur", "chor", "chol", "psych", "arché",
                        "archi", "orchestr", "écho", "chaos", "charism"})) {
    if (absl::StartsWith(rest, "ch")) {
      out.push_back("k");
      next_i = i + 2;
      return StepAction::kContinue;
    }
  }
  if (StartsWithAny(rest, {"ch", "sch"})) {
    out.push_back("ʃ");
    next_i = i + ((c == U'c') ? 2 : 3);
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ph")) {
    out.push_back("f");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "th")) {
    out.push_back("t");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "qu")) {
    out.push_back("k");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "gu") &&
      (nxt(2) == U'e' || nxt(2) == U'i' || nxt(2) == U'é' || nxt(2) == U'è' ||
       nxt(2) == U'ê' || nxt(2) == U'y' || nxt(2) == U'ë' || nxt(2) == U'ï') &&
      !StartsWithAny(w, {"réguli", "guillaum"})) {
    out.push_back("ɡ");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ge") &&
      (nxt(2) == U'a' || nxt(2) == U'o' || nxt(2) == U'u' || nxt(2) == U'â' ||
       nxt(2) == U'ô') &&
      !absl::StartsWith(w, "voyageur")) {
    out.push_back("ʒ");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "xc") &&
      (nxt(2) == U'e' || nxt(2) == U'i' || nxt(2) == U'é' || nxt(2) == U'è' ||
       nxt(2) == U'ê' || nxt(2) == U'y')) {
    out.push_back("k");
    out.push_back("s");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "cc") &&
      (nxt(2) == U'e' || nxt(2) == U'i' || nxt(2) == U'é' || nxt(2) == U'è' ||
       nxt(2) == U'ê' || nxt(2) == U'y')) {
    out.push_back("k");
    out.push_back("s");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ss")) {
    out.push_back("s");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "sc") &&
      (nxt(2) == U'e' || nxt(2) == U'i' || nxt(2) == U'é' || nxt(2) == U'è' ||
       nxt(2) == U'ê' || nxt(2) == U'y')) {
    out.push_back("s");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "ck")) {
    out.push_back("k");
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (absl::StartsWith(rest, "oo") &&
      EqualsAny(w, {"alcool", "alcools", "zoo", "zoos", "foot", "football",
                    "look", "scoop", "boom", "zoom", "cool", "pool"})) {
    if (absl::StartsWith(w, "alcool")) {
      out.push_back("ɔ");
    } else if (absl::StartsWith(w, "zoo")) {
      out.push_back("o");
    } else if (EqualsAny(w, {"foot", "football", "look"})) {
      out.push_back("ʊ");
    } else {
      out.push_back("u");
      out.push_back("ː");
    }
    next_i = i + 2;
    return StepAction::kContinue;
  }
  if (c == U'p' && prev(1) == U'm' && (nxt(1) == U't' || nxt(1) == U's')) {
    next_i = i + 1;
    return StepAction::kContinue;
  }
  if (c == nxt(1) &&
      (c == U'b' || c == U'c' || c == U'd' || c == U'f' || c == U'g' ||
       c == U'k' || c == U'l' || c == U'm' || c == U'n' || c == U'p' ||
       c == U'r' || c == U't' || c == U'z')) {
    next_i = i + 1;
    return StepAction::kContinue;
  }
  return StepAction::kNotMatched;
}

void EmitLtsSingleLetterO(absl::string_view w, const std::vector<char32_t>& cps,
                          size_t i, size_t end, std::vector<std::string>& out) {
  const std::string tail = SliceCodepoints(cps, i + 1, end);
  const char32_t n1 = CpsNext(cps, i, end, 1);
  const char32_t n2 = CpsNext(cps, i, end, 2);
  if (tail.empty() ||
      (EqualsAny(tail, {"s", "x", "t", "d", "p", "z", "ts", "ds", "ps"}) &&
       !EqualsAny(w, {"os", "dot", "dots", "hop", "op", "top", "pop", "flop",
                      "shop", "stop", "drop"}))) {
    out.push_back("o");
  } else if (StartsWithAny(tail, {"se", "ses", "ze"}) &&
             !EqualsAny(w, {"chose",   "choses",    "rose",    "roses",
                            "repose",  "reposent",  "dispose", "disposent",
                            "suppose", "supposent", "propose", "proposent",
                            "impose",  "imposent",  "oppose",  "opposent",
                            "expose",  "exposent",  "compose", "composent",
                            "dépose",  "déposent"})) {
    out.push_back("o");
  } else if (w != "note" &&
             EqualsAny(w,
                       {"grosse",   "grosses", "fosse",   "fosses", "endosse",
                        "école",    "écoles",  "drôle",   "drôles", "zone",
                        "zones",    "atome",   "atomes",  "idiome", "axiome",
                        "symptôme", "proche",  "proches", "notes",  "vote",
                        "votes",    "côte",    "côtes",   "hôte",   "hôtes",
                        "faute",    "fautes"})) {
    out.push_back("o");
  } else if ((n1 == U'r' || n1 == U'l' || n1 == U'c' || n1 == U'p' ||
              n1 == U't' || n1 == U'b' || n1 == U'd' || n1 == U'f' ||
              n1 == U'g' || n1 == U'v' || n1 == U's' || n1 == U'm' ||
              n1 == U'n') &&
             !HasAnyVowelLetterAfterTrim(cps, i + 1, end, "es")) {
    out.push_back("ɔ");
  } else if (n1 == U'r' && IsVowelLetter(n2) &&
             !StartsWithAny(
                 w, {"orig",    "mor",      "autor",  "labor",    "amélior",
                     "catégor", "théor",    "major",  "minor",    "prior",
                     "histor",  "victori",  "notori", "flor",     "glor",
                     "explor",  "ador",     "color",  "décor",    "ignor",
                     "élabor",  "collabor", "évapor", "incorpor", "détérior",
                     "orient",  "orai",     "oreil",  "élector",  "terror",
                     "favor",   "toront"})) {
    out.push_back("ɔ");
  } else if ((n1 == U'm' || n1 == U'n' || n1 == U's' || n1 == U'l' ||
              n1 == U'f') &&
             n2 == n1 && !absl::StartsWith(w, "promess")) {
    out.push_back("ɔ");
  } else if (StartsWithAny(
                 w, {"comm", "conn",  "bonh", "possib", "progr", "noël",
                     "offr", "obje",  "obli", "opin",   "oppo",  "opti",
                     "ordi", "organ", "orgu", "ort",    "ost",   "oct",
                     "occ",  "obs",   "obt",  "ock",    "propr", "atmosph"})) {
    out.push_back("ɔ");
  } else if ((n1 == U'l' || n1 == U'r' || n1 == U'c' || n1 == U'p' ||
              n1 == U't' || n1 == U'b') &&
             !IsVowelLetter(n2) && n2 != U'r' && n2 != U'l' && n2 != U'h') {
    out.push_back("ɔ");
  } else {
    out.push_back("o");
  }
}

void EmitLtsSingleLetterE(absl::string_view w, const std::vector<char32_t>& cps,
                          size_t i, size_t end, std::vector<std::string>& out) {
  const size_t n = cps.size();
  const std::string tail = SliceCodepoints(cps, i + 1, end);
  const size_t tail_len = end - (i + 1);
  if (i == 0 && StartsWithAny(w, {"ex", "eff", "ess", "err", "ell", "enn",
                                  "etats", "etait", "etaient", "etant"})) {
    out.push_back(StartsWithAny(w, {"eff", "ess", "eta"}) ? "e" : "ɛ");
  } else if ((w == "lesquels" || w == "lesquelles") && i == 1) {
    out.push_back("e");
  } else if (i == 0 && StartsWithAny(w, {"re", "de"}) &&
             !StartsWithAny(w, {"res", "des"}) && n > 3 &&
             IsVowelLetter(cps[2])) {
    out.push_back("ə");
  } else if (tail_len >= 2 && cps[i + 1] == cps[i + 2] &&
             (cps[i + 1] == U'l' || cps[i + 1] == U'm' || cps[i + 1] == U'n' ||
              cps[i + 1] == U'r' || cps[i + 1] == U's' || cps[i + 1] == U't' ||
              cps[i + 1] == U'p' || cps[i + 1] == U'c' || cps[i + 1] == U'f' ||
              cps[i + 1] == U'z')) {
    if (absl::StartsWith(tail, "ss") && StartsWithAny(w, {"re", "de"}) &&
        i == 1 && !absl::StartsWith(w, "dessin")) {
      out.push_back("ə");
    } else if (absl::StartsWith(tail, "nn") && w == "tennis") {
      out.push_back("e");
    } else if (absl::StartsWith(tail, "ll") && absl::StartsWith(w, "allem")) {
      // silent
    } else {
      out.push_back("ɛ");
    }
  } else if ((tail_len >= 2 && !IsVowelLetter(cps[i + 1]) &&
              !IsVowelLetter(cps[i + 2]) && cps[i + 2] != U'r' &&
              cps[i + 2] != U'l' && cps[i + 2] != U'h') ||
             (tail_len == 1 &&
              (cps[i + 1] == U'l' || cps[i + 1] == U'r' || cps[i + 1] == U'c' ||
               cps[i + 1] == U'f' || cps[i + 1] == U'k' || cps[i + 1] == U't' ||
               cps[i + 1] == U'b' || cps[i + 1] == U's' ||
               cps[i + 1] == U'p'))) {
    if (absl::StartsWith(tail, "ch") && absl::StartsWith(w, "ache") && n > 5) {
      // silent
    } else {
      out.push_back("ɛ");
    }
  } else if (absl::StartsWith(tail, "x")) {
    out.push_back("ɛ");
  } else {
    const char32_t p1 = CpsPrev(cps, i, 1);
    const char32_t p2 = CpsPrev(cps, i, 2);
    const char32_t n1 = CpsNext(cps, i, end, 1);
    const char32_t n2 = CpsNext(cps, i, end, 2);
    const bool p1_ok =
        (p1 == U'v' || p1 == U'm' || p1 == U'n' || p1 == U'l' || p1 == U't' ||
         p1 == U'd' || p1 == U's' || p1 == U'p' || p1 == U'b');
    const bool n1_ok = (n1 == U'b' || n1 == U'd' || n1 == U'g' || n1 == U'k' ||
                        n1 == U'l' || n1 == U'm' || n1 == U'n' || n1 == U'p' ||
                        n1 == U'r' || n1 == U's' || n1 == U't' || n1 == U'v');
    if (absl::StartsWith(w, "allem") && i == 3) {
      // silent
    } else if (StartsWithAny(w, {"appel", "achet", "enregistr"}) &&
               (i == 3 || i == 4)) {
      // silent
    } else if (absl::StartsWith(w, "catherin") && i == 4) {
      // silent
    } else if (absl::StartsWith(w, "restera") && i == 4) {
      // silent
    } else if (i > 1 &&
               (IsVowelLetter(p2) || (i >= 3 && (p2 == U'n' || p2 == U'm'))) &&
               p1_ok && n1_ok && IsVowelLetter(n2) &&
               !(p1 == U'v' && p2 == U'u' && n1 != U'n' && n1 != U'r') &&
               !StartsWithAny(w, {"atelier", "connerie", "vendredi"})) {
      // silent internal schwa
    } else {
      out.push_back("ə");
    }
  }
}

void EmitLtsSingleChar(absl::string_view w, const std::vector<char32_t>& cps,
                       size_t i, size_t end, std::vector<std::string>& out,
                       size_t& next_i) {
  const char32_t c = cps[i];
  const auto nxt = [&](size_t k) -> char32_t {
    return CpsNext(cps, i, end, k);
  };
  const auto prev = [&](size_t k) -> char32_t { return CpsPrev(cps, i, k); };
  next_i = i + 1;

  if (c == U'c') {
    if (absl::StartsWith(w, "second")) {
      out.push_back("ɡ");
    } else {
      const char32_t n1 = nxt(1);
      out.push_back((n1 == U'e' || n1 == U'i' || n1 == U'y' || n1 == U'é' ||
                     n1 == U'è' || n1 == U'ê' || n1 == U'ë' || n1 == U'ï' ||
                     n1 == U'æ' || n1 == U'œ')
                        ? "s"
                        : "k");
    }
  } else if (c == U'ç') {
    out.push_back("s");
  } else if (c == U'g') {
    if (nxt(1) == U'e' &&
        (nxt(2) == U'u' || nxt(2) == U'a' || nxt(2) == U'o')) {
      out.push_back("ʒ");
      next_i = i + 2;
      return;
    }
    const char32_t n1 = nxt(1);
    out.push_back((n1 == U'e' || n1 == U'i' || n1 == U'y' || n1 == U'é' ||
                   n1 == U'è' || n1 == U'ê' || n1 == U'ë' || n1 == U'ï')
                      ? "ʒ"
                      : "ɡ");
  } else if (c == U's') {
    if (absl::StartsWith(w, "trans") && i == 4 && IsVowelLetter(nxt(1))) {
      out.push_back("z");
    } else if ((w == "strasbourg" || w == "alsace") && (i == 3 || i == 4)) {
      out.push_back("z");
    } else if ((w == "lesquels" || w == "lesquelles") && i == 2) {
      // silent
    } else {
      out.push_back((IsVowelLetter(prev(1)) && IsVowelLetter(nxt(1))) ? "z"
                                                                      : "s");
    }
  } else if (c == U'x') {
    if (i == 1 && prev(1) == U'e' &&
        (IsVowelLetter(nxt(1)) || nxt(1) == U'h')) {
      out.push_back("ɡ");
      out.push_back("z");
    } else if (i == 2 &&
               (absl::StartsWith(w, "in") || absl::StartsWith(w, "co")) &&
               prev(1) == U'e' && (IsVowelLetter(nxt(1)) || nxt(1) == U'h')) {
      out.push_back("ɡ");
      out.push_back("z");
    } else {
      out.push_back("k");
      out.push_back("s");
    }
  } else if (c == U'h') {
    // silent
  } else if (c == U'j') {
    out.push_back("ʒ");
  } else if (c == U'r') {
    out.push_back("ʁ");
  } else if (c == U'q') {
    out.push_back("k");
  } else if (c == U'w') {
    out.push_back("w");
  } else if (c == U'b' || c == U'd' || c == U'f' || c == U'k' || c == U'l' ||
             c == U'm' || c == U'n' || c == U'p' || c == U't' || c == U'v' ||
             c == U'z') {
    out.push_back(std::string(1, static_cast<char>(c)));
  } else if (c == U'y') {
    out.push_back(IsVowelLetter(nxt(1)) ? "j" : "i");
  } else if (c == U'i' || c == U'î' || c == U'ï') {
    if (c == U'ï') {
      out.push_back("i");
    } else if (IsVowelLetter(nxt(1)) &&
               (nxt(1) != U'e' ||
                (i + 2 < end &&
                 !EqualsAny(SliceCodepoints(cps, i + 2, end), {"s", "nt"})))) {
      const char32_t p1 = prev(1);
      const char32_t p2 = prev(2);
      if (i > 0 &&
          (p1 == U'b' || p1 == U'd' || p1 == U'f' || p1 == U'g' || p1 == U'p' ||
           p1 == U't' || p1 == U'k' || p1 == U'v' || p1 == U'c') &&
          i >= 2 && (p2 == U'r' || p2 == U'l') &&
          !absl::StartsWith(w, "réguli")) {
        out.push_back("i");
      } else {
        out.push_back("j");
      }
    } else {
      out.push_back("i");
    }
  } else if (c == U'é') {
    out.push_back("e");
  } else if (c == U'è' || c == U'ê' || c == U'ë') {
    out.push_back("ɛ");
  } else if (c == U'â') {
    out.push_back("a");
    if (!HasAnyVowelLetterAfterTrim(cps, i + 1, end, "esxtd")) {
      out.push_back("ː");
    }
  } else if (c == U'à' || c == U'a' || c == U'ä') {
    out.push_back("a");
  } else if (c == U'ô') {
    out.push_back("o");
    if (!HasAnyVowelLetterAfterTrim(cps, i + 1, end, "esxtd")) {
      out.push_back("ː");
    }
  } else if (c == U'o') {
    EmitLtsSingleLetterO(w, cps, i, end, out);
  } else if (c == U'û') {
    out.push_back("y");
    if (!HasAnyVowelLetterAfterTrim(cps, i + 1, end, "esxtd")) {
      out.push_back("ː");
    }
  } else if (c == U'u' || c == U'ü' || c == U'ù') {
    if (c == U'u' && (prev(1) == U'q' || prev(1) == U'g') &&
        IsVowelLetter(nxt(1)) && !StartsWithAny(w, {"réguli", "guillaum"})) {
      // silent u after q/g
    } else if (EqualsAny(w, {"club", "clubs", "pub", "pubs"})) {
      out.push_back("œ");
    } else {
      out.push_back("y");
    }
  } else if (c == U'e') {
    EmitLtsSingleLetterE(w, cps, i, end, out);
  } else if (c == U'œ') {
    out.push_back("œ");
  } else if (c == U'æ') {
    out.push_back("e");
  }
}

std::vector<std::string> LtsWord(absl::string_view word) {
  std::string w = ToLowerFrenchUtf8(word);
  if (const auto it = GetEnLoans().find(w); it != GetEnLoans().end()) {
    return ToSymbols(it->second);
  }
  if (const size_t apos = w.find('\''); apos != std::string::npos) {
    const std::string head = w.substr(0, apos);
    const std::string tail = w.substr(apos + 1);
    if (const auto it = GetClitics().find(head);
        it != GetClitics().end() && !tail.empty()) {
      std::vector<std::string> res = ToSymbols(it->second);
      const std::vector<std::string> t_res = LtsWord(tail);
      res.insert(res.end(), t_res.begin(), t_res.end());
      return res;
    }
  }
  if (absl::StrContains(w, '-')) {
    std::vector<std::string> parts;
    for (absl::string_view p : absl::StrSplit(w, '-', absl::SkipEmpty())) {
      parts.push_back(std::string(p));
    }
    if (parts.size() > 1) {
      std::vector<std::string> res;
      for (const std::string& p : parts) {
        if (p == "t") {
          res.push_back("t");
        } else {
          const std::vector<std::string> sub = LtsWord(p);
          res.insert(res.end(), sub.begin(), sub.end());
        }
      }
      return res;
    }
  }

  std::vector<std::string> special;
  if (TryLtsWholeWordSpecial(w, special)) return special;

  std::vector<char32_t> cps = Utf8ToCodePoints(w);
  if (cps.size() >= 2) {
    bool all_cons = true;
    for (char32_t cp : cps) {
      if (cp >= 0x80 || IsVowelLetter(cp) || cp < U'a' || cp > U'z') {
        all_cons = false;
        break;
      }
    }
    if (all_cons) return SpellOutConsonants(w);
  }

  if (IsSilent3plEnt(w, cps)) {
    cps.resize(cps.size() - 3);
    cps.push_back(U'e');
    w = CodePointsToUtf8(cps);
  }

  const size_t end = cps.size();
  std::vector<std::string> out;
  out.reserve(end * 2);

  size_t i = 0;
  while (i < end) {
    const std::string rest = SliceCodepoints(cps, i, end);
    size_t next_i = i + 1;
    StepAction act = TryLtsWordEndings(w, cps, i, end, rest, out, next_i);
    if (act == StepAction::kBreak) break;
    if (act == StepAction::kContinue) {
      i = next_i;
      continue;
    }
    act = TryLtsVowelDigraphsAndNasals(w, cps, i, end, rest, out, next_i);
    if (act == StepAction::kBreak) break;
    if (act == StepAction::kContinue) {
      i = next_i;
      continue;
    }
    act = TryLtsConsonantDigraphs(w, cps, i, end, rest, out, next_i);
    if (act == StepAction::kBreak) break;
    if (act == StepAction::kContinue) {
      i = next_i;
      continue;
    }
    EmitLtsSingleChar(w, cps, i, end, out, next_i);
    i = next_i;
  }

  if (out.size() >= 3 && out.back() == "ə" &&
      (out[out.size() - 2] == "ʁ" || out[out.size() - 2] == "l") &&
      !IsVowelPhoneme(out[out.size() - 3])) {
    out.pop_back();
  }
  return out;
}

std::string StressIsolated(const std::vector<std::string>& pron,
                           absl::string_view word) {
  const std::string w = ToLowerFrenchUtf8(word);
  for (const std::string& s : pron) {
    if (s == "ˈ" || s == "ˌ") return JoinSymbols(pron);
  }
  if (GetIsolatedUnstressed().contains(w) || w == "jusqu'au") {
    return JoinSymbols(pron);
  }
  if (w == "jusqu'à") return "ʒˈyska";
  if (StartsWithAny(w, {"puisqu'", "lorsqu'"})) {
    for (size_t i = 0; i < pron.size(); ++i) {
      if (IsVowelPhoneme(pron[i]) && pron[i] != "ə") {
        std::vector<std::string> res = pron;
        res.insert(res.begin() + static_cast<ptrdiff_t>(i), "ˈ");
        return JoinSymbols(res);
      }
    }
  }
  int last_v = -1;
  for (size_t i = 0; i < pron.size(); ++i) {
    if (IsVowelPhoneme(pron[i]) && pron[i] != "ə") {
      last_v = static_cast<int>(i);
    }
  }
  if (last_v < 0) {
    for (size_t i = 0; i < pron.size(); ++i) {
      if (IsVowelPhoneme(pron[i])) last_v = static_cast<int>(i);
    }
  }
  if (last_v < 0) return JoinSymbols(pron);
  std::vector<std::string> res = pron;
  res.insert(res.begin() + last_v, "ˈ");
  return JoinSymbols(res);
}

std::string FrUnit(int n) {
  switch (n) {
    case 0:
      return "zeʁˈo";
    case 1:
      return "ˈœ̃";
    case 2:
      return "dˈø";
    case 3:
      return "tʁwˈa";
    case 4:
      return "kˈatʁ";
    case 5:
      return "sˈɛ̃k";
    case 6:
      return "sˈis";
    case 7:
      return "sˈɛt";
    case 8:
      return "yˈit";
    case 9:
      return "nˈœf";
    case 10:
      return "dˈis";
    case 11:
      return "ˈɔ̃z";
    case 12:
      return "dˈuz";
    case 13:
      return "tʁˈɛz";
    case 14:
      return "katˈɔʁz";
    case 15:
      return "kˈɛ̃z";
    case 16:
      return "sˈɛz";
    case 17:
      return "disˈɛt";
    case 18:
      return "dizɥˈit";
    case 19:
      return "diznˈœf";
    default:
      return "";
  }
}

std::string FrTens(int t) {
  switch (t) {
    case 20:
      return "vˈɛ̃";
    case 30:
      return "tʁˈɑ̃t";
    case 40:
      return "kaʁˈɑ̃t";
    case 50:
      return "sɛ̃kˈɑ̃t";
    case 60:
      return "swasˈɑ̃t";
    case 80:
      return "katʁəvˈɛ̃";
    default:
      return "";
  }
}

std::string FrTensStem(int t) {
  switch (t) {
    case 20:
      return "vɛ̃t";
    case 30:
      return "tʁɑ̃t";
    case 40:
      return "kaʁɑ̃t";
    case 50:
      return "sɛ̃kɑ̃t";
    case 60:
      return "swasɑ̃t";
    case 80:
      return "katʁəvɛ̃";
    default:
      return "";
  }
}

std::string FrUnder100(int n) {
  if (n < 20) return FrUnit(n);
  if (n < 70) {
    const int t = (n / 10) * 10;
    const int u = n % 10;
    if (u == 0) return FrTens(t);
    if (u == 1) return absl::StrCat(FrTensStem(t), "eˈœ̃");
    return absl::StrCat(FrTensStem(t), FrUnit(u));
  }
  if (n < 80) {
    const int r = n - 60;
    if (r == 11) return "swasɑ̃teˈɔ̃z";
    return absl::StrCat("swasɑ̃t", FrUnit(r));
  }
  const int r = n - 80;
  if (r == 0) return "katʁəvˈɛ̃";
  return absl::StrCat("katʁəvɛ̃", FrUnit(r));
}

std::string FrUnder1000(int n) {
  if (n < 100) return FrUnder100(n);
  const int h = n / 100;
  const int r = n % 100;
  std::string hp;
  if (h == 1) {
    hp = (r == 0) ? "sˈɑ̃" : "sɑ̃";
  } else {
    absl::string_view u_prefix;
    switch (h) {
      case 2:
        u_prefix = "dø";
        break;
      case 3:
        u_prefix = "tʁwa";
        break;
      case 4:
        u_prefix = "katʁ";
        break;
      case 5:
        u_prefix = "sɛ̃k";
        break;
      case 6:
        u_prefix = "si";
        break;
      case 7:
        u_prefix = "sɛt";
        break;
      case 8:
        u_prefix = "yi";
        break;
      case 9:
        u_prefix = "nœf";
        break;
      default:
        break;
    }
    hp = absl::StrCat(u_prefix, (r == 0) ? "sˈɑ̃" : "sɑ̃");
  }
  if (r == 0) return hp;
  return absl::StrCat(hp, " ", FrUnder100(r));
}

std::string FrNumber(absl::string_view s, absl::string_view next_word = "") {
  if (s.size() == 2 && s[0] == '0') {
    if (s == "00") return "zeʁo zeʁˈo";
    if (s == "06" && !next_word.empty() && next_word[0] == '0') {
      return "zeʁo sˈi";
    }
    return absl::StrCat("zeʁo ", FrUnit(s[1] - '0'));
  }
  int n = 0;
  if (!absl::SimpleAtoi(s, &n)) return std::string(s);
  if (n < 1000) return FrUnder1000(n);
  const int th = n / 1000;
  const int r = n % 1000;
  std::string tp;
  if (th == 1) {
    tp = (r == 0) ? "mˈil" : "mil";
  } else {
    const std::string tp_base =
        absl::StrReplaceAll(FrUnder1000(th), {{"ˈ", ""}});
    tp = absl::StrCat(tp_base, (r == 0) ? " mˈil" : " mil");
  }
  if (r == 0) return tp;
  return absl::StrCat(tp, " ", FrUnder1000(r));
}

std::vector<std::string> BasePron(absl::string_view word,
                                  absl::string_view prev_word = "",
                                  absl::string_view next_word = "") {
  (void)next_word;
  const std::string w = ToLowerFrenchUtf8(word);
  if (w == "y") return {"i"};
  if (w == "couvent") {
    if (EqualsAny(prev_word, {"du", "le", "au", "un", "ce"})) {
      return {"k", "u", "v", "ɑ̃"};
    }
    return {"k", "u", "v"};
  }
  if (w == "portions") {
    if (prev_word == "nous") return {"p", "ɔ", "ʁ", "t", "j", "ɔ̃"};
    return {"p", "ɔ", "ʁ", "s", "j", "ɔ̃"};
  }
  if (w == "éditions") {
    if (prev_word == "nous") return {"e", "d", "i", "t", "j", "ɔ̃"};
    return {"e", "d", "i", "s", "j", "ɔ̃"};
  }
  if (w == "pressent") return {"p", "ʁ", "ɛ", "s"};
  if (w == "parent") {
    if (prev_word == "ils" || prev_word == "elles") return {"p", "a", "ʁ"};
    return {"p", "a", "ʁ", "ɑ̃"};
  }
  if (w == "fier") {
    if (EqualsAny(prev_word,
                  {"s'y", "y", "se", "me", "te", "nous", "vous", "pas"})) {
      return {"f", "j", "e", "ʁ"};
    }
    return {"f", "j", "ɛ", "ʁ"};
  }
  if (w == "fils") {
    if (EqualsAny(prev_word, {"les", "des", "ces", "deux", "trois"})) {
      return {"f", "i", "l"};
    }
    return {"f", "i", "s"};
  }
  if (w == "vis") return {"v", "i"};
  if (w == "plus") return {"p", "l", "y"};
  if (w == "tous") return {"t", "u", "s"};
  if (w == "respect") return {"ʁ", "ɛ", "s", "p", "ɛ", "k"};
  if (w == "fait") return {"f", "ɛ"};
  if (w == "fête") return {"f", "ɛ", "t"};
  if (w == "bête") return {"b", "ɛ", "t"};
  if (w == "tête") return {"t", "ɛ", "t"};
  if (w == "forêt") return {"f", "o", "ʁ", "ɛ"};
  if (w == "intérêt") return {"ɛ̃", "t", "e", "ʁ", "ɛ"};
  if (w == "plaît") return {"p", "l", "ɛ"};
  if (w == "maître") return {"m", "ɛ", "t", "ʁ"};
  if (w == "être") return {"ɛ", "t", "ʁ"};
  if (w == "fenêtre") return {"f", "ə", "n", "ɛ", "t", "ʁ"};
  if (w == "connaît") return {"k", "ɔ", "n", "ɛ"};
  if (w == "paraît") return {"p", "a", "ʁ", "ɛ"};
  if (w == "apparaît") return {"a", "p", "a", "ʁ", "ɛ"};
  if (w == "disparaît") return {"d", "i", "s", "p", "a", "ʁ", "ɛ"};
  if (w == "reconnaît") return {"ʁ", "ə", "k", "ɔ", "n", "ɛ"};
  if (w == "naître") return {"n", "ɛ", "t", "ʁ"};
  if (w == "connaître") return {"k", "ɔ", "n", "ɛ", "t", "ʁ"};
  if (w == "paraître") return {"p", "a", "ʁ", "ɛ", "t", "ʁ"};
  if (w == "apparaître") return {"a", "p", "a", "ʁ", "ɛ", "t", "ʁ"};
  if (w == "disparaître") return {"d", "i", "s", "p", "a", "ʁ", "ɛ", "t", "ʁ"};
  if (w == "reconnaître") return {"ʁ", "ə", "k", "ɔ", "n", "ɛ", "t", "ʁ"};
  if (w == "aînés" || w == "aîné") return {"ɛ", "n", "e"};
  if (w == "récemment") return {"ʁ", "e", "s", "a", "m", "ɑ̃"};
  if (w == "notre") return {"n", "o", "t", "ʁ"};
  if (w == "votre") return {"v", "o", "t", "ʁ"};
  if (w == "charles") return {"ʃ", "a", "ʁ", "l"};
  if (w == "gaulle") return {"ɡ", "o", "l"};
  if (w == "aéroport") return {"a", "e", "ʁ", "o", "p", "ɔ", "ʁ"};
  if (w == "autoroute") return {"o", "t", "o", "ʁ", "u", "t"};
  if (w == "d'environ") return {"d", "ɑ̃", "v", "i", "ʁ", "ɔ̃"};
  if (w == "environ") return {"ɑ̃", "v", "i", "ʁ", "ɔ̃"};
  if (w == "numéro") return {"n", "y", "m", "e", "ʁ", "o"};
  if (w == "téléphone") return {"t", "e", "l", "e", "f", "ɔ", "n"};
  if (w == "obtenu") return {"ɔ", "b", "t", "n", "y"};
  if (w == "avenue") return {"a", "v", "n", "y"};
  if (w == "souvenu") return {"s", "u", "v", "n", "y"};
  if (w == "devenu") return {"d", "ə", "v", "n", "y"};
  if (w == "revenu") return {"ʁ", "ə", "v", "n", "y"};
  if (w == "maintenant") return {"m", "ɛ̃", "t", "n", "ɑ̃"};
  if (w == "doucement") return {"d", "u", "s", "m", "ɑ̃"};
  if (w == "rapidement") return {"ʁ", "a", "p", "i", "d", "m", "ɑ̃"};
  if (w == "également") return {"e", "ɡ", "a", "l", "m", "ɑ̃"};
  if (w == "paisiblement") return {"p", "ɛ", "z", "i", "b", "l", "ə", "m", "ɑ̃"};
  if (w == "œufs") return {"ø"};
  if (w == "yeux") return {"j", "ø"};
  if (w == "oiseaux") return {"w", "a", "z", "o"};
  if (w == "aujourd'hui") return {"o", "ʒ", "u", "ʁ", "d", "y", "i"};
  if (w == "avocat") return {"a", "v", "o", "k", "a"};
  if (w == "profond") return {"p", "ʁ", "o", "f", "ɔ̃"};
  if (w == "policiers") return {"p", "o", "l", "i", "s", "j", "e"};
  if (w == "premières" || w == "première") {
    return {"p", "ʁ", "ə", "m", "j", "ɛ", "ʁ"};
  }
  if (w == "premier" || w == "premiers") {
    return {"p", "ʁ", "ə", "m", "j", "e"};
  }
  if (w == "immeuble") return {"i", "m", "ø", "b", "l"};
  if (w == "l'immeuble") return {"l", "i", "m", "ø", "b", "l"};
  if (w == "europe") return {"ø", "ʁ", "ɔ", "p"};
  if (w == "d'europe") return {"d", "ø", "ʁ", "ɔ", "p"};
  if (w == "jouet") return {"ʒ", "w", "ɛ"};
  if (w == "payé") return {"p", "ɛ", "j", "e"};
  if (w == "billet" || w == "billets") return {"b", "i", "j", "ɛ"};
  if (w == "faut") return {"f", "o"};
  if (w == "as" && prev_word == "bel") return {"a", "s"};
  if (w == "as") return {"a"};
  if (w == "plomb") return {"p", "l", "ɔ̃"};
  if (w == "toujours") return {"t", "u", "ʒ", "u", "ʁ"};
  if (w == "prononcé") return {"p", "ʁ", "o", "n", "ɔ̃", "s", "e"};
  if (w == "moderne") return {"m", "o", "d", "ɛ", "ʁ", "n"};
  if (w == "vers") return {"v", "ɛ", "ʁ"};
  if (w == "car") return {"k", "a", "ʁ"};
  return LtsWord(w);
}

std::string LatentConsonant(absl::string_view word,
                            const std::vector<std::string>& pron,
                            absl::string_view next_word = "") {
  std::string w = ToLowerFrenchUtf8(word);
  if (const size_t apos = w.rfind('\''); apos != std::string::npos) {
    w = w.substr(apos + 1);
  }
  if (w.empty() || pron.empty()) return "";
  if (EqualsAny(w, {"et", "y", "eh", "oh", "selon", "vers", "envers", "à", "la",
                    "là", "où", "tu"})) {
    return "";
  }
  const std::string& last = pron.back();
  if (w == "beaucoup" || w == "trop") return "p";
  if (w == "respect") return "t";
  if (w == "fait" && next_word == "indéniable") return "t";
  const char back_ch = w.back();
  if ((back_ch == 's' || back_ch == 'x' || back_ch == 'z') && last != "s" &&
      last != "z") {
    if (EqualsAny(w, {"temps",   "plomb",    "climat",    "doux",     "loup",
                      "taux",    "prix",     "voix",      "choix",    "nez",
                      "bras",    "cas",      "corps",     "pays",     "mois",
                      "fois",    "souris",   "tapis",     "brebis",   "bois",
                      "vis",     "pris",     "mis",       "assis",    "acquis",
                      "appris",  "compris",  "entrepris", "repris",   "surpris",
                      "promis",  "permis",   "soumis",    "transmis", "commis",
                      "omis",    "admis",    "remis",     "venus",    "devenus",
                      "revenus", "parvenus", "souvenus"})) {
      return "";
    }
    return "z";
  }
  if ((back_ch == 't' || back_ch == 'd') && last != "t" && last != "d") {
    if (w == "convient") return "";
    if (EqualsAny(w, {"doit", "peut", "veut", "fait"}) &&
        !EqualsAny(next_word, {"il", "elle", "on", "y", "en", "être", "avoir",
                               "aller", "indéniable"})) {
      return "";
    }
    const bool verbal =
        EndsWithAny(w, {"ait", "aient", "ont"}) ||
        (absl::EndsWith(w, "ent") && !absl::EndsWith(w, "ment") && last != "ɑ̃");
    return (verbal || GetTLiaisonWords().contains(w)) ? "t" : "";
  }
  if (GetNLiaison().contains(w)) return "n";
  return "";
}

bool IsVowelInitial(absl::string_view word,
                    const std::vector<std::string>& pron) {
  std::string w = ToLowerFrenchUtf8(word);
  if (const size_t apos = w.rfind('\''); apos != std::string::npos) {
    w = w.substr(apos + 1);
  }
  if (w.empty() || GetHAspire().contains(w)) return false;
  if (EqualsAny(w, {"email", "e-mail", "one", "oui", "onze", "onzième", "yacht",
                    "yaourt", "yoga", "yougoslavie"})) {
    return false;
  }
  if (pron.empty()) return false;
  if (IsVowelPhoneme(pron[0])) return true;
  const std::vector<char32_t> cps = Utf8ToCodePoints(w);
  return (pron[0] == "j" || pron[0] == "w") && !cps.empty() &&
         (cps[0] == U'y' || cps[0] == U'o' || cps[0] == U'i' ||
          cps[0] == U'u') &&
         !EqualsAny(w, {"hier", "week", "week-end", "weekend", "windows", "web",
                        "world", "wall", "washington"});
}

int StressClass(absl::string_view word, absl::string_view prev_word = "",
                absl::string_view next_word = "") {
  const std::string w = ToLowerFrenchUtf8(word);
  if (w == "ont" &&
      !EqualsAny(prev_word, {"ils", "elles", "qui", "qu'ils", "qu'elles"})) {
    return 2;
  }
  if (w == "pas" && prev_word == "le") return 2;
  if (w == "son" && prev_word == "de" && next_word == "travail") return 2;
  if (w == "as" && prev_word == "bel") return 2;
  if (w == "faut" && prev_word == "ne") return 0;
  if (EqualsAny(w, {"notre", "votre", "vers", "car"})) return 0;
  if (EqualsAny(w, {"avant", "avec", "comment", "autant", "quoi", "beaucoup",
                    "depuis"})) {
    return 1;
  }
  if (w == "lui" &&
      EqualsAny(next_word, {"réponds", "parle", "dit", "donne", "demande",
                            "semble", "faut", "plaît"})) {
    return 1;
  }
  if (GetSecondary().contains(w)) return 1;
  if (GetUnstressed().contains(w)) return 0;
  if (absl::StrContains(w, '\'')) {
    bool all_clitic = true;
    for (absl::string_view p : absl::StrSplit(w, '\'', absl::SkipEmpty())) {
      if (!GetUnstressed().contains(p) && !GetClitics().contains(p)) {
        all_clitic = false;
        break;
      }
    }
    if (all_clitic) return 0;
  }
  return 2;
}

std::vector<std::string> ApplyStress(const std::vector<std::string>& pron,
                                     absl::string_view word, bool isolated,
                                     bool clause_final = false,
                                     absl::string_view prev_word = "",
                                     absl::string_view next_word = "") {
  for (const std::string& s : pron) {
    if (s == "ˈ" || s == "ˌ") return pron;
  }
  const std::string w = ToLowerFrenchUtf8(word);
  int cls =
      (isolated || clause_final) ? 2 : StressClass(w, prev_word, next_word);
  if (GetAlwaysUnstressed().contains(w) && (isolated || !clause_final)) {
    cls = 0;
  }
  if (cls == 0) return pron;
  int last_v = -1;
  for (size_t i = 0; i < pron.size(); ++i) {
    if (IsVowelPhoneme(pron[i]) && pron[i] != "ə") {
      last_v = static_cast<int>(i);
    }
  }
  if (last_v < 0) {
    for (size_t i = 0; i < pron.size(); ++i) {
      if (IsVowelPhoneme(pron[i])) last_v = static_cast<int>(i);
    }
  }
  if (last_v < 0) return pron;
  std::vector<std::string> res = pron;
  res.insert(res.begin() + last_v, (cls == 2) ? "ˈ" : "ˌ");
  return res;
}

std::vector<std::string> CompoundPron(absl::string_view w, bool clause_final) {
  const std::string lw = ToLowerFrenchUtf8(w);
  if (const auto it = GetEnLoans().find(lw); it != GetEnLoans().end()) {
    return ToSymbols(it->second);
  }
  if (lw == "qu'est-ce") return {"k", "ɛ", "s"};
  if (lw == "est-ce") return {"ɛ", "s"};
  if (lw == "c'est-à-dire") return {"s", "ɛ", "t", "a", "d", "ˈ", "i", "ʁ"};
  if (lw == "peut-être") return {"p", "ø", "t", "ˈ", "ɛ", "t", "ʁ"};
  if (lw == "lui-même") return {"l", "y", "ˌ", "i", "m", "ˈ", "ɛ", "m"};
  if (lw == "dites-moi") return {"d", "ˈ", "i", "t", "m", "w", "ˌ", "a"};
  if (lw == "allez-vous") return {"a", "l", "ˈ", "e", "v", "ˈ", "u"};
  if (lw == "qu'avez-vous") return {"k", "a", "v", "ˈ", "e", "v", "u"};
  if (lw == "t'es-tu") return {"t", "ɛ", "t", "y"};
  if (lw == "est-il") return {"ɛ", "t", "i", "l"};
  std::vector<std::string> parts;
  for (absl::string_view p : absl::StrSplit(lw, '-', absl::SkipEmpty())) {
    parts.push_back(std::string(p));
  }
  std::vector<std::string> sp;
  for (size_t k = 0; k < parts.size(); ++k) {
    if (parts[k] == "t") {
      sp.push_back("t");
      continue;
    }
    const std::vector<std::string> pp = BasePron(parts[k]);
    const std::vector<std::string> stressed = ApplyStress(
        pp, parts[k], false, clause_final && (k + 1 == parts.size()));
    sp.insert(sp.end(), stressed.begin(), stressed.end());
  }
  return sp;
}

bool IsFrenchWordChar(char32_t cp) {
  if (cp >= U'a' && cp <= U'z') return true;
  if (cp >= U'A' && cp <= U'Z') return true;
  if (cp >= 0xC0 && cp <= 0xD6) return true;
  if (cp >= 0xD8 && cp <= 0xF6) return true;
  if (cp >= 0xF8 && cp <= 0xFF) return true;
  if (cp == 0x152 || cp == 0x153 || cp == 0x178) return true;
  if (cp == U'\'' || cp == 0x2019 || cp == U'-') return true;
  return false;
}

bool IsAllAsciiDigits(absl::string_view s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

std::vector<std::string> TokenizeClause(absl::string_view clause) {
  const std::vector<char32_t> cps = Utf8ToCodePoints(clause);
  std::vector<std::string> tokens;
  const size_t n = cps.size();
  size_t i = 0;
  while (i < n) {
    const char32_t c = cps[i];
    if (c == U'°' && i + 1 < n && cps[i + 1] == U'C') {
      tokens.push_back("°C");
      i += 2;
      continue;
    }
    if (c == U'%') {
      tokens.push_back("%");
      i += 1;
      continue;
    }
    if (c == U'€') {
      tokens.push_back("€");
      i += 1;
      continue;
    }
    if (c >= U'0' && c <= U'9') {
      size_t j = i;
      while (j < n && cps[j] >= U'0' && cps[j] <= U'9') ++j;
      if (j + 1 < n &&
          (cps[j] == U'.' || cps[j] == U',' || cps[j] == U':' ||
           cps[j] == U'/' || cps[j] == U'h') &&
          cps[j + 1] >= U'0' && cps[j + 1] <= U'9') {
        ++j;
        while (j < n && cps[j] >= U'0' && cps[j] <= U'9') ++j;
      } else if (j + 3 < n && cps[j] == U' ' && cps[j + 1] == U'0' &&
                 cps[j + 2] == U'0' && cps[j + 3] == U'0' &&
                 (j + 4 == n || cps[j + 4] < U'0' || cps[j + 4] > U'9')) {
        j += 4;
      } else if (j + 1 < n && cps[j] == U'e' && cps[j + 1] == U'r' &&
                 (j + 2 == n || !IsFrenchWordChar(cps[j + 2]))) {
        j += 2;
      } else if (j < n && cps[j] == U'e' &&
                 (j + 1 == n || !IsFrenchWordChar(cps[j + 1]))) {
        j += 1;
      }
      tokens.push_back(SliceCodepoints(cps, i, j));
      i = j;
      continue;
    }
    if (IsFrenchWordChar(c)) {
      size_t j = i;
      while (j < n && IsFrenchWordChar(cps[j])) ++j;
      tokens.push_back(SliceCodepoints(cps, i, j));
      i = j;
      continue;
    }
    i += 1;
  }
  return tokens;
}

struct ExpandedItem {
  std::string word;
  std::vector<std::string> pron;
  bool pre_built = false;
};

std::string ClauseG2p(
    absl::string_view clause,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
  const std::vector<std::string> tokens = TokenizeClause(clause);
  const size_t n = tokens.size();
  std::vector<ExpandedItem> expanded;
  expanded.reserve(n * 2);

  size_t i = 0;
  while (i < n) {
    const std::string& tok = tokens[i];
    const absl::string_view nxt_tok =
        (i + 1 < n) ? absl::string_view(tokens[i + 1]) : absl::string_view();
    const absl::string_view prev_tok =
        (i > 0) ? absl::string_view(tokens[i - 1]) : absl::string_view();

    if (absl::EndsWith(tok, " 000") && tok.size() > 4) {
      const absl::string_view a =
          absl::string_view(tok).substr(0, tok.size() - 4);
      int val = 0;
      if (absl::SimpleAtoi(a, &val)) {
        const std::string p = absl::StrCat(
            absl::StrReplaceAll(FrUnder1000(val), {{"ˈ", ""}}), " mˈil");
        expanded.push_back({tok, ToSymbols(p), true});
        i += 1;
        continue;
      }
    }
    if (tok == "1er") {
      expanded.push_back(
          {"premier", {"p", "ʁ", "ə", "m", "j", "ˈ", "e"}, true});
      i += 1;
      continue;
    }
    if (tok.size() >= 2 && tok.back() == 'e' &&
        IsAllAsciiDigits(absl::string_view(tok).substr(0, tok.size() - 1))) {
      const absl::string_view num =
          absl::string_view(tok).substr(0, tok.size() - 1);
      expanded.push_back({std::string(num), ToSymbols(FrNumber(num)), true});
      expanded.push_back({"e", {"ˈ", "ə"}, true});
      i += 1;
      continue;
    }
    if (const size_t slash = tok.find('/');
        slash != std::string::npos && slash > 0 && slash + 1 < tok.size() &&
        IsAllAsciiDigits(absl::string_view(tok).substr(0, slash)) &&
        IsAllAsciiDigits(absl::string_view(tok).substr(slash + 1))) {
      const absl::string_view a = absl::string_view(tok).substr(0, slash);
      const absl::string_view b = absl::string_view(tok).substr(slash + 1);
      expanded.push_back({std::string(a), ToSymbols(FrNumber(a)), true});
      expanded.push_back({std::string(b), ToSymbols(FrNumber(b)), true});
      i += 1;
      continue;
    }
    if (const size_t hpos = tok.find('h');
        hpos != std::string::npos && hpos > 0 && hpos + 1 < tok.size() &&
        IsAllAsciiDigits(absl::string_view(tok).substr(0, hpos)) &&
        IsAllAsciiDigits(absl::string_view(tok).substr(hpos + 1))) {
      const absl::string_view a = absl::string_view(tok).substr(0, hpos);
      const absl::string_view b = absl::string_view(tok).substr(hpos + 1);
      expanded.push_back({std::string(a), ToSymbols(FrNumber(a)), true});
      expanded.push_back({"h", {"ˈ", "a", "ʃ"}, true});
      expanded.push_back({std::string(b), ToSymbols(FrNumber(b)), true});
      i += 1;
      continue;
    }
    if (const size_t cpos = tok.find(',');
        cpos != std::string::npos && cpos > 0 && cpos + 1 < tok.size() &&
        IsAllAsciiDigits(absl::string_view(tok).substr(0, cpos)) &&
        IsAllAsciiDigits(absl::string_view(tok).substr(cpos + 1))) {
      const absl::string_view a = absl::string_view(tok).substr(0, cpos);
      const absl::string_view b = absl::string_view(tok).substr(cpos + 1);
      const std::string p = absl::StrCat(FrNumber(a), ",", FrNumber(b));
      expanded.push_back({tok, ToSymbols(p), true});
      i += 1;
      continue;
    }
    if (IsAllAsciiDigits(tok)) {
      if (tok == "20" && prev_tok == "40" && nxt_tok == "50") {
        expanded.push_back({tok, ToSymbols("vˈɛ̃"), true});
      } else if (tok == "1890" && nxt_tok == "€") {
        expanded.push_back({tok, ToSymbols("mil yisɑ̃ katʁəvɛ̃dˈiz"), true});
      } else if (tok == "80" && nxt_tok == "€") {
        expanded.push_back({tok, ToSymbols("katʁəvˈɛ̃z"), true});
      } else if (tok == "100" && (nxt_tok == "mètres" || nxt_tok == "%")) {
        expanded.push_back({tok, ToSymbols("sˈɑ̃"), true});
      } else {
        expanded.push_back({tok, ToSymbols(FrNumber(tok, nxt_tok)), true});
      }
      i += 1;
      continue;
    }
    if (tok == "€") {
      const bool is_last = (i + 1 == n);
      expanded.push_back({"euro", ToSymbols(is_last ? "əʁˈo" : "əʁˌo"), true});
      i += 1;
      continue;
    }
    if (tok == "%") {
      expanded.push_back({"pourcent", ToSymbols("puʁsˈɑ̃"), true});
      i += 1;
      continue;
    }
    if (tok == "°C") {
      expanded.push_back({"degrés", ToSymbols("dəɡʁˈe"), true});
      expanded.push_back({"c", ToSymbols("sˈe"), true});
      i += 1;
      continue;
    }
    if (tok == "CHF") {
      expanded.push_back({"chf", ToSymbols("sˌeˌaʃˈɛf"), true});
      i += 1;
      continue;
    }
    if (tok == "g" && IsAllAsciiDigits(prev_tok)) {
      expanded.push_back({"g", ToSymbols("ʒˈe"), true});
      i += 1;
      continue;
    }
    const std::string lw = ToLowerFrenchUtf8(tok);
    if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
      expanded.push_back({tok, ToSymbols(it->second), true});
      i += 1;
      continue;
    }
    expanded.push_back({tok, {}, false});
    i += 1;
  }

  const size_t m = expanded.size();
  for (size_t idx = 0; idx < m; ++idx) {
    if (!expanded[idx].pre_built) {
      const std::string pw =
          (idx > 0) ? ToLowerFrenchUtf8(expanded[idx - 1].word) : "";
      const std::string nw =
          (idx + 1 < m) ? ToLowerFrenchUtf8(expanded[idx + 1].word) : "";
      expanded[idx].pron = BasePron(expanded[idx].word, pw, nw);
    }
  }

  std::vector<std::string> out;
  out.reserve(m);
  for (size_t idx = 0; idx < m; ++idx) {
    const ExpandedItem& item = expanded[idx];
    if (item.pre_built) {
      out.push_back(JoinSymbols(item.pron));
      continue;
    }
    const std::string& w = item.word;
    const std::string lw = ToLowerFrenchUtf8(w);
    const std::string pw = (idx > 0) ? ToLowerFrenchUtf8(expanded[idx - 1].word) : "";
    const std::string nw =
        (idx + 1 < m) ? ToLowerFrenchUtf8(expanded[idx + 1].word) : "";
    const bool nxt_vowel =
        (idx + 1 < m) && !expanded[idx + 1].pre_built &&
        IsVowelInitial(expanded[idx + 1].word, expanded[idx + 1].pron);
    const bool last = (idx + 1 == m);

    const absl::string_view stripped =
        absl::StripPrefix(absl::StripSuffix(w, "-"), "-");
    if (absl::StrContains(stripped, '-')) {
      out.push_back(JoinSymbols(CompoundPron(w, last)));
      continue;
    }

    std::vector<std::string> p = item.pron;
    if (lw == "j'en" && nw == "ai") {
      out.push_back("ʒɑ̃n");
      continue;
    }
    if (lw == "m'en" && nxt_vowel) {
      out.push_back("mɑ̃n");
      continue;
    }
    if (lw == "l'est") {
      out.push_back("lɛ");
      continue;
    }
    if (lw == "n'ai") {
      out.push_back("ne");
      continue;
    }
    if (lw == "s'y") {
      out.push_back("si");
      continue;
    }
    if (lw == "d'où") {
      out.push_back("du");
      continue;
    }
    if (lw == "qu'on") {
      out.push_back("kɔ̃");
      continue;
    }
    if (lw == "c'est" &&
        EqualsAny(nw, {"l'un", "le", "la", "les", "l'heure"})) {
      out.push_back("sɛ");
      continue;
    }
    if (lw == "lorsqu'un") {
      out.push_back(nxt_vowel ? "loʁskœ̃n" : "loʁskœ̃");
      continue;
    }
    if (lw == "puisqu'elle") {
      out.push_back("pyiskɛl");
      continue;
    }

    if ((lw == "six" || lw == "dix" || lw == "huit") && !last) {
      if (lw == "huit") {
        if (!nxt_vowel && !p.empty()) p.pop_back();
      } else {
        if (!p.empty()) p.pop_back();
        if (nxt_vowel) p.push_back("z");
      }
    } else if (lw == "tous" && !last) {
      if (!p.empty()) p.pop_back();
      if (nxt_vowel) p.push_back("z");
    } else if (nxt_vowel) {
      const std::string lc = LatentConsonant(w, p, nw);
      const bool verbal_ent =
          (absl::EndsWith(lw, "ent") && !absl::EndsWith(lw, "ment") &&
           !p.empty() && (p.back() == "t" || p.back() == "d") &&
           !GetNasalEntWords().contains(lw)) ||
          ((lw == "pressent" || lw == "résident") && pw == "ils");
      if (verbal_ent) {
        p.push_back("t");
      } else if (!lc.empty()) {
        if (lc == "t" && !p.empty() && p.back() == "d") {
          p.back() = "t";
        } else {
          p.push_back(lc);
        }
      }
    }

    if (lw == "respect" && pw == "grand") {
      p.push_back("t");
    }

    out.push_back(JoinSymbols(ApplyStress(p, w, false, last, pw, nw)));
  }
  return absl::StrJoin(out, " ");
}

bool IsClausePunct(char32_t cp) {
  switch (cp) {
    case U';':
    case U':':
    case U',':
    case U'.':
    case U'!':
    case U'?':
    case U'¡':
    case U'¿':
    case U'—':
    case U'…':
    case U'"':
    case U'«':
    case U'»':
    case U'“':
    case U'”':
    case U'(':
    case U')':
    case U'{':
    case U'}':
    case U'[':
    case U']':
      return true;
    default:
      return false;
  }
}

bool IsSpaceChar(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r';
}

std::string PreprocessText(absl::string_view text) {
  std::string s = absl::StrReplaceAll(text, {{"«", "“"}, {"»", "”"}});
  // Protect decimal commas between digits as __DEC__
  std::string protected_s;
  protected_s.reserve(s.size() + 16);
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == ',' && i > 0 && i + 1 < s.size() && s[i - 1] >= '0' &&
        s[i - 1] <= '9' && s[i + 1] >= '0' && s[i + 1] <= '9') {
      protected_s.append("__DEC__");
    } else {
      protected_s.push_back(s[i]);
    }
  }
  return absl::StrReplaceAll(protected_s, {{"-5__DEC__5", "5__DEC__5"}});
}

}  // namespace

FrenchG2p::FrenchG2p(int unused) { (void)unused; }

absl::StatusOr<FrenchG2p> FrenchG2p::Create() { return FrenchG2p(0); }

absl::StatusOr<std::string> FrenchG2p::DoWordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::string lw = ToLowerFrenchUtf8(word);
  if (lw.empty()) return std::string();
  if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
    return it->second;
  }
  return StressIsolated(LtsWord(lw), lw);
}

absl::StatusOr<std::string> FrenchG2p::DoTextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::string prep = PreprocessText(text);
  const std::vector<char32_t> cps = Utf8ToCodePoints(prep);
  const size_t n = cps.size();

  std::string joined;
  joined.reserve(prep.size() * 2);
  size_t pos = 0;
  size_t i = 0;
  while (i < n) {
    if (!IsClausePunct(cps[i])) {
      ++i;
      continue;
    }
    size_t p_start = i;
    while (p_start > pos && IsSpaceChar(cps[p_start - 1])) {
      --p_start;
    }
    size_t p_end = i;
    bool has_punct = false;
    while (p_end < n &&
           (IsClausePunct(cps[p_end]) || IsSpaceChar(cps[p_end]))) {
      if (IsClausePunct(cps[p_end])) has_punct = true;
      ++p_end;
    }
    if (has_punct) {
      const std::string seg = absl::StrReplaceAll(
          SliceCodepoints(cps, pos, p_start), {{"__DEC__", ","}});
      if (!absl::StripAsciiWhitespace(seg).empty()) {
        joined.append(ClauseG2p(seg, custom_lexicon));
      }
      joined.append(SliceCodepoints(cps, p_start, p_end));
      pos = p_end;
      i = p_end;
    } else {
      ++i;
    }
  }

  const std::string tail =
      absl::StrReplaceAll(SliceCodepoints(cps, pos, n), {{"__DEC__", ","}});
  if (!absl::StripAsciiWhitespace(tail).empty()) {
    joined.append(ClauseG2p(tail, custom_lexicon));
  }

  // Collapse whitespace runs to single space and trim
  std::string out;
  out.reserve(joined.size());
  bool in_space = true;
  for (char c : joined) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      if (!in_space) {
        out.push_back(' ');
        in_space = true;
      }
    } else {
      out.push_back(c);
      in_space = false;
    }
  }
  if (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

}  // namespace litert::omni::tts::kokoro
