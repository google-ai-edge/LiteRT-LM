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

#include "omni/text2speech/kokoro/italian_g2p.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

bool IsVowel(char32_t cp) {
  switch (cp) {
    case U'a':
    case U'e':
    case U'i':
    case U'o':
    case U'u':
    case U'à':
    case U'è':
    case U'é':
    case U'ì':
    case U'í':
    case U'ò':
    case U'ó':
    case U'ù':
    case U'ú':
      return true;
    default:
      return false;
  }
}

bool IsAccVowel(char32_t cp) {
  switch (cp) {
    case U'à':
    case U'è':
    case U'é':
    case U'ì':
    case U'í':
    case U'ò':
    case U'ó':
    case U'ù':
    case U'ú':
      return true;
    default:
      return false;
  }
}

bool IsFrontVowel(char32_t cp) {
  switch (cp) {
    case U'e':
    case U'i':
    case U'é':
    case U'è':
    case U'ì':
    case U'í':
      return true;
    default:
      return false;
  }
}

absl::string_view BaseVowel(char32_t cp) {
  switch (cp) {
    case U'a':
    case U'à':
      return "a";
    case U'e':
    case U'é':
      return "e";
    case U'è':
      return "ɛ";
    case U'i':
    case U'ì':
    case U'í':
      return "i";
    case U'o':
    case U'ó':
      return "o";
    case U'ò':
      return "ɔ";
    case U'u':
    case U'ù':
    case U'ú':
      return "u";
    default:
      return "";
  }
}

bool IsWordChar(char32_t cp) {
  if ((cp >= U'a' && cp <= U'z') || (cp >= U'A' && cp <= U'Z') ||
      (cp >= U'0' && cp <= U'9') || cp == U'_') {
    return true;
  }
  return IsVowel(ToLowerLatin(cp));
}

bool IsAlphaWord(absl::string_view tok) {
  if (tok.empty()) return false;
  size_t pos = 0;
  char32_t cp = 0;
  if (!ConsumeUtf8CodePoint(tok, pos, cp)) return false;
  if (cp >= U'0' && cp <= U'9') return false;
  return IsWordChar(cp);
}

bool IsAllDigits(absl::string_view s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

const absl::flat_hash_set<std::string>& GetAtoneMono() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "a",   "ai",  "al",  "che", "chi", "ci",  "col", "con",  "da",  "dai",
      "dal", "del", "di",  "e",   "ed",  "fra", "gli", "ha",   "hai", "hanno",
      "ho",  "i",   "il",  "in",  "la",  "le",  "lo",  "ma",   "mi",  "ne",
      "nei", "nel", "non", "o",   "od",  "per", "più", "quel", "se",  "si",
      "su",  "sul", "ti",  "tra", "tu",  "un",  "vi",  "è",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetAtonePoly() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "abbiamo", "agli",   "alla",   "alle",   "allo",   "anche",  "aveva",
      "come",    "contro", "dagli",  "dalla",  "dalle",  "dallo",  "degli",
      "della",   "delle",  "dello",  "dove",   "era",    "lei",    "loro",
      "lui",     "mentre", "mia",    "mie",    "miei",   "mio",    "nessuno",
      "noi",     "ogni",   "oltre",  "perché", "pure",   "quando", "quella",
      "quelle",  "quelli", "quello", "questa", "queste", "questi", "questo",
      "sei",     "senza",  "sia",    "siamo",  "sono",   "sua",    "sue",
      "sugli",   "sui",    "sulla",  "sulle",  "sullo",  "suo",    "suoi",
      "tua",     "tue",    "tuo",    "tuoi",   "una",    "uno",    "voi",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetHiatusFirst() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "aeree", "aree", "baule",   "bauli", "bue",  "dia",   "diano", "dii",
      "dio",   "due",  "fruscio", "idee",  "io",   "linee", "lui",   "mia",
      "mie",   "mio",  "paura",   "paure", "pia",  "pie",   "pii",   "pio",
      "scia",  "scie", "sia",     "siamo", "spia", "spie",  "stia",  "stiano",
      "sua",   "sue",  "suo",     "tua",   "tue",  "tuo",   "via",   "vie",
      "zia",   "zie",  "zii",     "zio",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetSdrucciolaExempt() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "abbandono",   "accadere",    "amici",       "amico",       "antica",
      "antiche",     "antichi",     "antico",      "appartenere", "astenersi",
      "atmosfera",   "atmosfere",   "avere",       "averi",       "bambina",
      "bambine",     "bambini",     "bambino",     "bandiera",    "bandiere",
      "barbiere",    "barbieri",    "barriera",    "barriere",    "bicchiere",
      "bicchieri",   "bufera",      "buona",       "buone",       "buoni",
      "buono",       "cadere",      "cameriera",   "cameriere",   "cammini",
      "cammino",     "campagnola",  "cantiere",    "cantieri",    "carabiniere",
      "carabinieri", "carriera",    "carriere",    "carriola",    "casseruola",
      "cavaliere",   "cavalieri",   "cera",        "chimera",     "cittadina",
      "cittadine",   "cittadini",   "cittadino",   "collina",     "colline",
      "coloni",      "colono",      "confine",     "confini",     "contadina",
      "contadine",   "contadini",   "contadino",   "contenere",   "corriere",
      "corrieri",    "cucina",      "cucine",      "decadere",    "destini",
      "destino",     "dissuadere",  "dolere",      "doni",        "dono",
      "dottrina",    "dottrine",    "dovere",      "doveri",      "fagioli",
      "fagiolo",     "farina",      "farine",      "fatica",      "fatiche",
      "figliola",    "figliolo",    "formica",     "frontiera",   "frontiere",
      "galera",      "giacere",     "giardini",    "giardino",    "godere",
      "ingegnere",   "ingegneri",   "intera",      "intere",      "interi",
      "intero",      "leggera",     "leggere",     "leggeri",     "leggero",
      "lenzuolo",    "maniera",     "maniere",     "mantenere",   "mattina",
      "mattine",     "mattini",     "mattino",     "mestiere",    "mestieri",
      "mollica",     "nemici",      "nemico",      "nera",        "nere",
      "neri",        "nero",        "nona",        "none",        "noni",
      "nono",        "ortica",      "ottenere",    "pantera",     "parere",
      "pareri",      "patroni",     "patrono",     "pensieri",    "pensiero",
      "pera",        "perdono",     "pere",        "persuadere",  "piacere",
      "piaceri",     "portiere",    "portieri",    "possedere",   "potere",
      "poteri",      "preghiera",   "preghiere",   "prevedere",   "prigioniera",
      "prigioniere", "prigionieri", "prigioniero", "primavera",   "primavere",
      "provvedere",  "pudico",      "quartiere",   "quartieri",   "regina",
      "regine",      "ricadere",    "rimanere",    "rivedere",    "rovina",
      "rovine",      "rubrica",     "sapere",      "saperi",      "scadere",
      "scogliera",   "scogliere",   "sedere",      "sentieri",    "sentiero",
      "sera",        "sere",        "severa",      "severe",      "severi",
      "severo",      "sincera",     "sincere",     "sinceri",     "sincero",
      "solere",      "sono",        "sostenere",   "spagnola",    "spagnole",
      "spagnoli",    "spagnolo",    "stadera",     "straniera",   "straniere",
      "stranieri",   "straniero",   "suoni",       "suono",       "tacere",
      "tastiera",    "tastiere",    "temere",      "tenere",      "toni",
      "tono",        "trattenere",  "troni",       "trono",       "tuoni",
      "tuono",       "usignolo",    "valere",      "vedere",      "vera",
      "vere",        "veri",        "vero",        "vescica",     "vicina",
      "vicine",      "vicini",      "vicino",      "volere",      "zeri",
      "zero",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetApocopeFinalWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "amor",   "andar",   "ardor",   "aver",      "calor",  "candor", "color",
      "dar",    "dolor",   "dottor",  "error",     "far",    "favor",  "fervor",
      "furor",  "hotel",   "maggior", "miglior",   "minor",  "odor",   "onor",
      "orror",  "peggior", "poter",   "professor", "rancor", "rigor",  "rumor",
      "saper",  "sapor",   "signor",  "splendor",  "star",   "stupor", "tenor",
      "terror", "timor",   "umor",    "valer",     "veder",  "vigor",  "voler",
  });
  return *kSet;
}

const absl::flat_hash_map<std::string, std::string>& GetWordOverrides() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"account", "akːˈawnt"},
          {"all", "ˈal"},
          {"apprezzato", "apːrˌeʦːˈato"},
          {"avesse", "avˈesse"},
          {"avessero", "avˈessero"},
          {"avessi", "avˈessɪ"},
          {"avessimo", "avˈessimo"},
          {"by", "bˈI"},
          {"colui", "kolˈuːi"},
          {"computer", "kompjˈuter"},
          {"facebook", "fˈAsbʊk"},
          {"gay", "ɡˈA"},
          {"ii", "jˈi"},
          {"john", "ʤˈɔn"},
          {"leader", "lˈider"},
          {"new", "njˈuː"},
          {"news", "njˈuːz"},
          {"online", "ˌon lˈIn"},
          {"paura", "paˈʊra"},
          {"riunione", "riʊnjˈone"},
          {"siano", "sˈiːano"},
          {"software", "sˈɔft̪wer"},
          {"sì", "sˈiː"},
          {"tour", "tˈuːr"},
          {"you", "jˈoʊ"},
          {"youtube", "jʊː tjˈuːb"},
      });
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>& GetLetterNames() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"a", "ˈaː"},       {"b", "bˈi"},        {"c", "ʧˈi"},
          {"cm", "ʧˌiˈɛmme"}, {"d", "d̪ˈiː"},       {"f", "ˈɛffe"},
          {"g", "ʤˈi"},       {"h", "ˈakːa"},      {"j", "ˈi lˈuŋɡa"},
          {"k", "kˈapːa"},    {"kg", "kˌapːaʤˈi"}, {"km", "kˌapːaˈɛmme"},
          {"l", "ˈɛlle"},     {"m", "ˈɛmme"},      {"n", "ˈɛnne"},
          {"p", "pˈi"},       {"pp", "pˌipˈi"},    {"q", "kˈu"},
          {"r", "ˈɛrɾe"},     {"s", "ˈɛSe"},       {"t", "tˈi"},
          {"u", "ˈuː"},       {"v", "vˈu"},        {"w", "dˌopːja vˈu"},
          {"x", "ˈiks"},      {"y", "ˈipsilon"},   {"z", "ʣˈɛta"},
      });
  return *kMap;
}

const absl::flat_hash_set<std::string>& GetSdrWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "abbiano",       "abita",         "abiti",       "abito",
      "acido",         "adige",         "aiutano",     "alberi",
      "amano",         "amazon",        "ambiti",      "ambito",
      "amsterdam",     "analisi",       "andarsene",   "angela",
      "angeles",       "angeli",        "angelo",      "angoli",
      "angolo",        "anima",         "anime",       "animo",
      "anonima",       "anonimo",       "anticipo",    "aquila",
      "araba",         "arabi",         "arabo",       "arbitro",
      "arcivescovo",   "articolo",      "aspettano",   "assumere",
      "attimo",        "auguro",        "aumentano",   "autobus",
      "autonoma",      "autonomo",      "barbara",     "bastano",
      "benessere",     "benjamin",      "bergamo",     "biasimo",
      "brindisi",      "cagliari",      "calibro",     "cambiano",
      "camera",        "camere",        "canada",      "canone",
      "capitoli",      "capitolo",      "caratteri",   "caratteristico",
      "carichi",       "cattedra",      "cavolo",      "celebre",
      "cellule",       "cercano",       "cesare",      "chiacchiere",
      "chiamano",      "cinema",        "circuito",    "codice",
      "codici",        "combattono",    "community",   "comoda",
      "comodo",        "compiere",      "compiti",     "compito",
      "complice",      "considera",     "considerano", "console",
      "contano",       "continuano",    "copyright",   "corrano",
      "correre",       "crediti",       "credito",     "crescita",
      "cronaca",       "cronache",      "curriculum",  "davide",
      "debbano",       "debiti",        "debito",      "debole",
      "deboli",        "decimo",        "dedica",      "deficit",
      "deposito",      "desidera",      "desideri",    "determina",
      "diagnosi",      "dialoghi",      "diavolo",     "dibattito",
      "difficile",     "difficili",     "digital",     "dimostrano",
      "diocesi",       "disputa",       "distinguere", "ditemi",
      "diventano",     "dodici",        "dollari",     "dubito",
      "eccomi",        "elena",         "elizabeth",   "entrano",
      "epoca",         "erano",         "esercito",    "esito",
      "esprimere",     "essere",        "esseri",      "estera",
      "esteri",        "ettari",        "facile",      "facili",
      "farcela",       "farmaci",       "farmaco",     "fascino",
      "fatemi",        "favola",        "fegato",      "femmina",
      "fenomeni",      "fenomeno",      "fermati",     "festival",
      "fidati",        "figurati",      "filosofo",    "follower",
      "formano",       "formula",       "fragile",     "funzionano",
      "genera",        "genere",        "generi",      "genova",
      "giacomo",       "giocano",       "giocattolo",  "giovane",
      "giovani",       "girano",        "girolamo",    "giudice",
      "giudici",       "giuridica",     "giuridico",   "gratuita",
      "gratuiti",      "gratuito",      "guardano",    "guidano",
      "immagina",      "immagino",      "incarichi",   "incontrano",
      "incubo",        "indica",        "indicano",    "indice",
      "indigeni",      "informati",     "iniziano",    "interessano",
      "international", "internet",      "interprete",  "intimo",
      "inutile",       "inutili",       "ipotesi",     "isola",
      "isole",         "italo",         "italy",       "jonathan",
      "lacrime",       "lavorano",      "leggere",     "legittima",
      "legittimo",     "lesbiche",      "lettera",     "libera",
      "libere",        "liberi",        "ligure",      "limita",
      "limite",        "limiti",        "limpido",     "liquido",
      "macchina",      "macchine",      "magazine",    "manager",
      "mancano",       "mangiano",      "mantova",     "marketing",
      "martiri",       "maschera",      "mediaset",    "medica",
      "mediche",       "medici",        "medico",      "merita",
      "meritano",      "meriti",        "merito",      "metodi",
      "metodo",        "microsoft",     "minima",      "minimi",
      "minimo",        "missili",       "modena",      "modulo",
      "monaci",        "monaco",        "monitor",     "mostrano",
      "muoiono",       "napoli",        "nascita",     "national",
      "necessita",     "nomina",        "numeri",      "numero",
      "obblighi",      "obbligo",       "occupa",      "occupano",
      "oliver",        "opera",         "ordina",      "organi",
      "organo",        "ospita",        "ospite",      "ospiti",
      "ossigeno",      "ottima",        "ottime",      "ottimi",
      "ottimo",        "padova",        "pagano",      "pagina",
      "parentesi",     "parlano",       "partecipa",   "partecipano",
      "passano",       "penisola",      "pensano",     "perdita",
      "perdite",       "piacciono",     "piccolo",     "popoli",
      "popolo",        "portano",       "portatile",   "possano",
      "possono",       "povera",        "poveri",      "preoccupa",
      "presentano",    "preside",       "prestiti",    "prestito",
      "principe",      "principi",      "privacy",     "profughi",
      "proiettili",    "proposito",     "provano",     "provoca",
      "pubblico",      "quindici",      "raccontano",  "rapida",
      "rapido",        "rappresentano", "recita",      "redditi",
      "reddito",       "regola",        "regole",      "restano",
      "ricordano",     "ricordati",     "riguardano",  "rinascita",
      "risultano",     "sabato",        "satira",      "scandalo",
      "scatola",       "scatole",       "scusami",     "sedici",
      "seguono",       "sembrano",      "semplice",    "settimo",
      "simboli",       "simbolo",       "simile",      "simili",
      "sindaci",       "sindaco",       "sindrome",    "singola",
      "singole",       "singoli",       "singolo",     "sinonimo",
      "sintesi",       "sintomi",       "solido",      "solita",
      "solite",        "soliti",        "solito",      "sottotitoli",
      "spettacolo",    "spiriti",       "spirito",     "splendida",
      "splendido",     "stefano",       "stomaco",     "stupida",
      "stupidi",       "stupido",       "subito",      "supera",
      "svizzera",      "taranto",       "tavola",      "tavolo",
      "telecamera",    "telecamere",    "telecom",     "telefono",
      "termina",       "tessera",       "titoli",      "titolo",
      "tornano",       "tramite",       "trappola",    "tredici",
      "trovano",       "ultima",        "ultime",      "ultimi",
      "ultimo",        "undici",        "university",  "usano",
      "utile",         "utili",         "utilizzano",  "vadano",
      "valida",        "valido",        "valuta",      "vedova",
      "vendita",       "vendite",       "venere",      "veneto",
      "vengano",       "vennero",       "vertice",     "vescovi",
      "vescovo",       "vigili",        "visita",      "visite",
      "vittima",       "vittime",       "zucchero",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetPenultWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "amica",     "amiche",   "aperti",      "averci",    "averla",
      "averli",    "averlo",   "avermi",      "averne",    "benefici",
      "cimitero",  "concerti", "consigliere", "davvero",   "divano",
      "diversi",   "edifici",  "enrico",      "esperti",   "esterne",
      "federico",  "felici",   "impero",      "interne",   "luciano",
      "ministero", "mistero",  "moderne",     "monastero", "nicola",
      "ovvero",    "poterlo",  "potersi",     "riunione",  "sacrifici",
      "saperlo",   "saperne",  "tenersi",     "uffici",    "vederla",
      "vederlo",   "vederti",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetOpenEWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "accademia",
      "accademico",
      "accendere",
      "accento",
      "accesso",
      "accetto",
      "adesso",
      "aerea",
      "aerei",
      "aereo",
      "affetti",
      "affetto",
      "agenda",
      "albergo",
      "alberto",
      "america",
      "andrea",
      "andrebbe",
      "andremo",
      "andrew",
      "aperta",
      "aperte",
      "aperti",
      "aperto",
      "argento",
      "arresti",
      "arresto",
      "aspetta",
      "aspetti",
      "aspetto",
      "assedio",
      "assemblea",
      "atleti",
      "atmosfera",
      "attende",
      "attendere",
      "attenta",
      "attento",
      "attraverso",
      "aver",
      "averci",
      "avere",
      "averla",
      "averli",
      "averlo",
      "avermi",
      "averne",
      "avrebbe",
      "avrei",
      "avremo",
      "avvenne",
      "baby",
      "basterebbe",
      "battesimo",
      "beach",
      "beh",
      "bei",
      "bel",
      "belgio",
      "bella",
      "belle",
      "belli",
      "bello",
      "ben",
      "bene",
      "benessere",
      "benevento",
      "beni",
      "beppe",
      "bergamo",
      "best",
      "beve",
      "biblioteca",
      "bisognerebbe",
      "black",
      "breve",
      "brevi",
      "bruxelles",
      "buenos",
      "capienza",
      "cappello",
      "cartella",
      "caserta",
      "castel",
      "cattedra",
      "celebre",
      "celeste",
      "cellule",
      "center",
      "centesimi",
      "cento",
      "centri",
      "centro",
      "cercano",
      "certa",
      "certe",
      "certi",
      "certo",
      "cesso",
      "champions",
      "check",
      "chiedeva",
      "chiese",
      "cieco",
      "cielo",
      "cimitero",
      "cinquecento",
      "collega",
      "collegio",
      "commedia",
      "commercio",
      "compenso",
      "complessi",
      "complesso",
      "completa",
      "completo",
      "comprende",
      "comprendere",
      "concerti",
      "concerto",
      "concesso",
      "concetti",
      "concetto",
      "concreta",
      "concreto",
      "congresso",
      "connessi",
      "conosceva",
      "conoscevo",
      "consenso",
      "conserva",
      "contea",
      "contempo",
      "conteneva",
      "contenta",
      "contento",
      "contesti",
      "contesto",
      "controversia",
      "controversie",
      "convento",
      "coperta",
      "coperte",
      "coperto",
      "corea",
      "corretta",
      "corretto",
      "crea",
      "creano",
      "crede",
      "credeva",
      "credevo",
      "credi",
      "credo",
      "crema",
      "criteri",
      "criterio",
      "crudele",
      "database",
      "dea",
      "debba",
      "decenni",
      "decennio",
      "decimo",
      "decreto",
      "dedica",
      "deficit",
      "dell",
      "der",
      "derby",
      "deserto",
      "desiderio",
      "destra",
      "destro",
      "detective",
      "determina",
      "deve",
      "devi",
      "devo",
      "dialetto",
      "dicembre",
      "difende",
      "difendere",
      "difetti",
      "difetto",
      "dipende",
      "direbbe",
      "direi",
      "diretta",
      "dirette",
      "diretti",
      "diretto",
      "disprezzo",
      "divenne",
      "diventa",
      "diversa",
      "diverse",
      "diversi",
      "diverso",
      "domestica",
      "domestici",
      "domestico",
      "dover",
      "dovesse",
      "dovessi",
      "doveva",
      "dovevi",
      "dovevo",
      "dovrebbe",
      "dovrei",
      "dovremo",
      "ebbe",
      "ebbene",
      "ebrei",
      "eccesso",
      "eccetto",
      "ecco",
      "eccomi",
      "eco",
      "ed",
      "effetti",
      "effetto",
      "ego",
      "elena",
      "elenco",
      "eletti",
      "eletto",
      "elettrica",
      "elettrico",
      "ella",
      "emanuele",
      "emerge",
      "emma",
      "en",
      "energetica",
      "ennesima",
      "ennesimo",
      "ente",
      "enti",
      "enzo",
      "epoca",
      "er",
      "era",
      "erano",
      "erba",
      "erede",
      "eri",
      "eric",
      "ero",
      "esempi",
      "esempio",
      "esercito",
      "esisteva",
      "esito",
      "esperti",
      "esperto",
      "espresso",
      "esse",
      "esser",
      "esserci",
      "essere",
      "esseri",
      "esserlo",
      "essermi",
      "esserne",
      "essersi",
      "essi",
      "estende",
      "estera",
      "esteri",
      "esterna",
      "esterne",
      "esterni",
      "esterno",
      "estero",
      "estetica",
      "estrema",
      "estremi",
      "estremo",
      "et",
      "eterna",
      "eterno",
      "etica",
      "etico",
      "ettari",
      "eu",
      "euro",
      "europea",
      "europee",
      "europei",
      "europeo",
      "eva",
      "evento",
      "ex",
      "extra",
      "faccenda",
      "facesse",
      "faceva",
      "facevo",
      "farebbe",
      "farei",
      "faremo",
      "febbre",
      "fecero",
      "fedeli",
      "ferie",
      "ferro",
      "festa",
      "feste",
      "festival",
      "finestra",
      "finestre",
      "foresta",
      "foreste",
      "fratello",
      "frattempo",
      "galera",
      "game",
      "games",
      "gazzetta",
      "gen",
      "genera",
      "genere",
      "generi",
      "generico",
      "geni",
      "genio",
      "genova",
      "gerusalemme",
      "gesti",
      "gesto",
      "giuseppe",
      "glielo",
      "gliene",
      "governi",
      "governo",
      "greca",
      "greci",
      "greco",
      "guerra",
      "guerre",
      "guglielmo",
      "harry",
      "henry",
      "hey",
      "hotel",
      "idea",
      "idee",
      "identico",
      "ieri",
      "immenso",
      "impero",
      "inferno",
      "ingegnere",
      "ingegneri",
      "inghilterra",
      "ingresso",
      "insetti",
      "intende",
      "intendere",
      "intendevo",
      "intendi",
      "intensa",
      "intenso",
      "intento",
      "intera",
      "interesse",
      "interessi",
      "interi",
      "interna",
      "international",
      "interne",
      "interni",
      "interno",
      "intero",
      "interprete",
      "intervento",
      "intraprendere",
      "inverno",
      "israele",
      "jack",
      "jean",
      "juventus",
      "lady",
      "lee",
      "leggenda",
      "leggera",
      "leggere",
      "leggero",
      "leggo",
      "lei",
      "lenta",
      "lento",
      "letta",
      "lettera",
      "lettere",
      "letti",
      "letto",
      "leva",
      "levi",
      "liceo",
      "lorenzo",
      "maestra",
      "maestri",
      "maestro",
      "magazine",
      "manifesta",
      "manifesto",
      "mantenere",
      "marea",
      "mary",
      "materia",
      "materie",
      "matteo",
      "media",
      "mediaset",
      "medica",
      "medici",
      "medico",
      "medio",
      "medioevo",
      "meglio",
      "mele",
      "membri",
      "membro",
      "mensa",
      "merce",
      "merci",
      "merda",
      "merita",
      "meritano",
      "meriti",
      "merito",
      "mese",
      "mesi",
      "messe",
      "metodi",
      "metodo",
      "metri",
      "metro",
      "mezza",
      "mezzi",
      "mezzo",
      "miami",
      "ministero",
      "miseria",
      "mistero",
      "moderna",
      "moderne",
      "moderni",
      "moderno",
      "monastero",
      "musei",
      "museo",
      "nel",
      "nell",
      "nelson",
      "neo",
      "nervi",
      "novecento",
      "novembre",
      "offerta",
      "offerte",
      "offerto",
      "oggetti",
      "oggetto",
      "ok",
      "onesta",
      "onesti",
      "onesto",
      "orchestra",
      "osserva",
      "page",
      "palermo",
      "palestra",
      "parallelo",
      "parentesi",
      "partecipa",
      "passeggeri",
      "peggio",
      "pen",
      "pene",
      "pensa",
      "pensi",
      "penso",
      "perche",
      "perde",
      "perdere",
      "perdi",
      "perdita",
      "perdite",
      "perfetta",
      "perfetti",
      "perfetto",
      "perla",
      "perle",
      "pero",
      "persa",
      "perse",
      "perso",
      "pesca",
      "pessima",
      "pessimo",
      "peste",
      "petto",
      "pezzi",
      "pezzo",
      "piacerebbe",
      "piaceva",
      "pianeti",
      "play",
      "poeta",
      "poeti",
      "polemica",
      "possedere",
      "possesso",
      "poter",
      "poteri",
      "poterlo",
      "poteva",
      "potevi",
      "potevo",
      "potrebbe",
      "potrei",
      "potremmo",
      "potremo",
      "potreste",
      "potresti",
      "preda",
      "prefetto",
      "prega",
      "prego",
      "premi",
      "premier",
      "premio",
      "prenda",
      "prende",
      "prendere",
      "prenderlo",
      "prendersi",
      "prendi",
      "presenta",
      "preside",
      "press",
      "pressi",
      "presso",
      "prestiti",
      "prestito",
      "presto",
      "prete",
      "pretendere",
      "pretesto",
      "preti",
      "prevedere",
      "prezzi",
      "prezzo",
      "primavera",
      "privilegi",
      "privilegio",
      "problema",
      "problemi",
      "procedimento",
      "processi",
      "processo",
      "profeta",
      "progetti",
      "progetto",
      "progressi",
      "progresso",
      "protesta",
      "proteste",
      "protetta",
      "protetto",
      "raccogliere",
      "raffaele",
      "rappresenta",
      "rappresentano",
      "ravenna",
      "re",
      "real",
      "recita",
      "record",
      "red",
      "redditi",
      "reddito",
      "referendum",
      "regge",
      "reggio",
      "regio",
      "regola",
      "regole",
      "renda",
      "rende",
      "rendere",
      "renderlo",
      "rendersi",
      "rendi",
      "rendono",
      "renzi",
      "replica",
      "report",
      "resta",
      "restano",
      "resti",
      "resto",
      "retro",
      "retta",
      "ricerche",
      "ricetta",
      "ricette",
      "riceve",
      "riflesso",
      "rimanere",
      "rimedio",
      "riprende",
      "riprendere",
      "riserva",
      "riserve",
      "rispetto",
      "rivedere",
      "rivela",
      "roberto",
      "romeo",
      "salerno",
      "sanremo",
      "saper",
      "saperlo",
      "saperne",
      "saprei",
      "sarebbe",
      "sarei",
      "saremo",
      "scena",
      "scende",
      "scendere",
      "scene",
      "scheda",
      "schede",
      "schema",
      "schemi",
      "scoperta",
      "scoperte",
      "scoperto",
      "scriveva",
      "sebbene",
      "sec",
      "secoli",
      "secolo",
      "sede",
      "sedia",
      "seggi",
      "segreti",
      "sei",
      "sembrerebbe",
      "sempre",
      "sensi",
      "senso",
      "sequestro",
      "serbia",
      "sergio",
      "seri",
      "serie",
      "serio",
      "serra",
      "serva",
      "serve",
      "server",
      "servi",
      "service",
      "servo",
      "servono",
      "sesso",
      "sesta",
      "sesto",
      "set",
      "sette",
      "settembre",
      "settimo",
      "sex",
      "sfera",
      "silenzio",
      "sistema",
      "sistemi",
      "sofferto",
      "sorella",
      "sorprende",
      "sospetti",
      "sospetto",
      "sostenere",
      "specchio",
      "specie",
      "spendere",
      "spento",
      "spettro",
      "splende",
      "splendida",
      "splendido",
      "stage",
      "statunitense",
      "statunitensi",
      "stipendi",
      "successi",
      "successo",
      "suprema",
      "supremo",
      "svezia",
      "tacere",
      "talento",
      "taylor",
      "tecnica",
      "tecnici",
      "tecnico",
      "tel",
      "tela",
      "telecom",
      "telefoni",
      "telefono",
      "tema",
      "temi",
      "tempesta",
      "tempi",
      "tempo",
      "tenda",
      "tende",
      "tennis",
      "tenta",
      "teresa",
      "terme",
      "termina",
      "termine",
      "termini",
      "terra",
      "terre",
      "terrestre",
      "terza",
      "terzi",
      "terzo",
      "tesi",
      "tessera",
      "test",
      "testa",
      "teste",
      "testi",
      "testo",
      "torneo",
      "tragedia",
      "treni",
      "treno",
      "trento",
      "tutela",
      "ue",
      "umberto",
      "universo",
      "valerio",
      "vangelo",
      "vecchia",
      "vecchio",
      "vederla",
      "vederlo",
      "vederti",
      "vedeva",
      "vedevo",
      "vedremo",
      "vende",
      "vendere",
      "venere",
      "veneto",
      "venezia",
      "venezuela",
      "venti",
      "vento",
      "verbo",
      "verrebbe",
      "versa",
      "verso",
      "vertice",
      "vertici",
      "veste",
      "vicenda",
      "vicende",
      "viceversa",
      "vincenzo",
      "violenta",
      "violento",
      "viveva",
      "voler",
      "volesse",
      "voleva",
      "volevo",
      "vorrebbe",
      "vorrei",
      "vorremmo",
      "vorresti",
      "web",
      "week",
      "west",
      "zero",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetClosedEWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "capelli",    "capello",  "chiede",   "chiedi",   "chiedo",
      "chiedono",   "commesso", "degli",    "della",    "delle",
      "dello",      "detto",    "diede",    "diego",    "fede",
      "fresco",     "legge",    "mela",     "mente",    "menti",
      "mentre",     "negli",    "nella",    "nelle",    "nello",
      "nera",       "nere",     "neri",     "nero",     "neve",
      "penna",      "pepe",     "piedi",    "possiede", "quella",
      "quelle",     "quelli",   "quello",   "questa",   "queste",
      "questi",     "questo",   "rete",     "richiede", "richiedere",
      "richiedono", "rientra",  "rientro",  "riesca",   "riesce",
      "riesci",     "riesco",   "riescono", "sente",    "senti",
      "sera",       "sete",     "siete",    "spiega",   "spieghi",
      "spiego",     "stella",   "stelle",   "tetto",    "vera",
      "vere",       "veri",     "vero",     "vetri",    "vetro",
      "vietnam",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetOpenOWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "abbandono",   "aborto",
      "accogliere",  "accolto",
      "accordi",     "accordo",
      "accorto",     "addosso",
      "adoro",       "aeroporto",
      "alloggio",    "android",
      "anonima",     "anonimo",
      "antonio",     "aosta",
      "appoggio",    "apposta",
      "approccio",   "autonoma",
      "autonomo",    "biologico",
      "blocca",      "blocco",
      "bob",         "bon",
      "book",        "bordo",
      "bosco",       "boss",
      "boston",      "botte",
      "buone",       "buongiorno",
      "buoni",       "california",
      "cattolica",   "cattolici",
      "cattolico",   "cerimonia",
      "cio",         "coca",
      "codice",      "codici",
      "cogliere",    "coi",
      "coinvolti",   "coinvolto",
      "col",         "college",
      "colloquio",   "colonia",
      "colonie",     "colto",
      "comico",      "comma",
      "comoda",      "comodo",
      "complice",    "comporta",
      "concordo",    "console",
      "consorzio",   "contano",
      "controlla",   "controlli",
      "controllo",   "copia",
      "coppa",       "coppia",
      "copre",       "copyright",
      "corda",       "corde",
      "core",        "coro",
      "corpi",       "corpo",
      "corre",       "cosa",
      "cose",        "cosi",
      "cost",        "costa",
      "coste",       "costi",
      "costo",       "crollo",
      "cronaca",     "cross",
      "cuore",       "cuori",
      "curiosa",     "custodia",
      "deliziosa",   "deposito",
      "dimora",      "dimostra",
      "dimostrano",  "dintorni",
      "diocesi",     "diploma",
      "divorzio",    "dollari",
      "donna",       "donne",
      "dorme",       "dose",
      "dosi",        "dott",
      "dottor",      "droga",
      "economica",   "economici",
      "economico",   "elettronica",
      "elettronico", "episodi",
      "episodio",    "eroe",
      "eroi",        "esplodere",
      "esporre",     "esposti",
      "esposto",     "europa",
      "europe",      "fenomeni",
      "fenomeno",    "filosofo",
      "foggia",      "fogli",
      "foglie",      "foglio",
      "folla",       "folle",
      "follower",    "ford",
      "formula",     "forte",
      "forti",       "forum",
      "forza",       "forze",
      "fossa",       "fossero",
      "fossi",       "fossimo",
      "foto",        "fotografo",
      "frode",       "funzionano",
      "george",      "ginocchio",
      "gioca",       "giocano",
      "giochi",      "gioco",
      "gioia",       "giorgio",
      "girolamo",    "gol",
      "gold",        "golf",
      "good",        "grossa",
      "grosse",      "grossi",
      "grosso",      "ho",
      "home",        "hong",
      "horror",      "idiota",
      "idioti",      "immobile",
      "immobili",    "imporre",
      "importa",     "importo",
      "imposta",     "imposte",
      "imposto",     "indossa",
      "ipotesi",     "john",
      "jones",       "jordan",
      "kong",        "lavora",
      "lavoro",      "livorno",
      "lobby",       "loco",
      "lodi",        "logica",
      "logico",      "logo",
      "lol",         "long",
      "look",        "lord",
      "loro",        "love",
      "madonna",     "manovra",
      "marocco",     "matrimonio",
      "memorie",     "meravigliosa",
      "mezzogiorno", "misericordia",
      "mobile",      "mobili",
      "moda",        "modena",
      "modi",        "modo",
      "modulo",      "monaci",
      "monaco",      "monica",
      "monitor",     "monopolio",
      "monza",       "moon",
      "morgan",      "moro",
      "morta",       "morte",
      "morti",       "morto",
      "mossa",       "mosse",
      "mostrano",    "moto",
      "muore",       "negozi",
      "negozio",     "no",
      "nobile",      "nobili",
      "nodi",        "nodo",
      "noia",        "nomi",
      "nomina",      "nonna",
      "nonni",       "nonno",
      "nord",        "norma",
      "norme",       "nostra",
      "nostre",      "nostri",
      "nostro",      "not",
      "nota",        "note",
      "noti",        "noto",
      "notte",       "nove",
      "nozze",       "o",
      "obbligo",     "occhi",
      "occhio",      "occorre",
      "occupano",    "odio",
      "off",         "oggi",
      "olio",        "oliver",
      "ong",         "opera",
      "opere",       "or",
      "ordine",      "organi",
      "organo",      "oro",
      "ospita",      "ospite",
      "ospiti",      "ottica",
      "otto",        "ovest",
      "ovvio",       "parola",
      "parole",      "parrocchia",
      "passaporto",  "patrimonio",
      "paul",        "performance",
      "pericolosi",  "petrolio",
      "pianoforte",  "pilota",
      "piloti",      "pioggia",
      "piombo",      "piove",
      "pistola",     "piuttosto",
      "poca",        "poche",
      "pochi",       "poco",
      "podio",       "poi",
      "point",       "polo",
      "polonia",     "pomodoro",
      "pop",         "popoli",
      "popolo",      "porca",
      "porco",       "porno",
      "porre",       "porta",
      "portafoglio", "portano",
      "porte",       "porti",
      "porto",       "posa",
      "possa",       "possano",
      "posso",       "possono",
      "posta",       "poste",
      "povera",      "povere",
      "poveri",      "povero",
      "preziosa",    "preziosi",
      "profughi",    "promossa",
      "promosso",    "proporre",
      "proposito",   "propri",
      "propria",     "proprie",
      "proprio",     "prossima",
      "prossime",    "prossimi",
      "prossimo",    "protocollo",
      "prova",       "prove",
      "provi",       "provo",
      "psicologica", "psicologico",
      "purtroppo",   "qualcosa",
      "quota",       "quote",
      "raccoglie",   "raccogliere",
      "raccolta",    "raccolte",
      "raccolti",    "raccolto",
      "raccontano",  "rapporti",
      "rapporto",    "retorica",
      "ricorda",     "ricordano",
      "ricordati",   "ricordi",
      "ricordo",     "ricorrere",
      "rimosso",     "rinnovo",
      "riporta",     "riposo",
      "risolto",     "ritrova",
      "ritrovo",     "rivolta",
      "rivolte",     "rivolto",
      "roba",        "robot",
      "rocca",       "roccia",
      "rocco",       "rock",
      "roger",       "rosa",
      "rose",        "sacerdote",
      "savoia",      "school",
      "sciopero",    "scopi",
      "scopo",       "scopre",
      "scopri",      "scorre",
      "scorta",      "scossa",
      "scott",       "sforzi",
      "sforzo",      "shock",
      "sinonimo",    "slogan",
      "so",          "soci",
      "socio",       "software",
      "soglia",      "sol",
      "soldi",       "soli",
      "solido",      "solita",
      "solite",      "soliti",
      "solito",      "sonora",
      "sorta",       "sorte",
      "sosta",       "spagnola",
      "spagnoli",    "spagnolo",
      "sporco",      "sport",
      "sposi",       "sposta",
      "stavolta",    "sto",
      "stomaco",     "stop",
      "storica",     "storiche",
      "storici",     "storico",
      "storie",      "studiosi",
      "supporto",    "svolta",
      "svolte",      "svolto",
      "talvolta",    "tecnologico",
      "terremoto",   "tesoro",
      "testimone",   "testimoni",
      "tifosi",      "togliere",
      "tom",         "tommy",
      "toni",        "tono",
      "top",         "tor",
      "tornano",     "toro",
      "torta",       "torto",
      "trasporti",   "trasporto",
      "troia",       "troll",
      "trono",       "troppa",
      "troppe",      "troppi",
      "troppo",      "trova",
      "trovi",       "trovo",
      "videogiochi", "viola",
      "vittorie",    "vol",
      "volta",       "volte",
      "volti",       "vostra",
      "vostre",      "vostri",
      "vostro",      "wall",
      "world",       "york",
      "zona",        "zone",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetClosedOWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "come",       "condotta", "condotte", "condotti",   "condotto",
      "corso",      "dopo",     "fuochi",   "interrotto", "introdotta",
      "introdotto", "molta",    "molte",    "molti",      "molto",
      "muove",      "muoversi", "posto",    "prodotta",   "prodotte",
      "prodotti",   "prodotto", "promuove", "ridotta",    "ridotti",
      "ridotto",    "rotta",    "rotto",    "sono",       "sotto",
      "suona",      "tradotto",
  });
  return *kSet;
}

bool HasStressedIaSuffix(absl::string_view w) {
  static const absl::NoDestructor<std::vector<absl::string_view>> kSuffixes({
      "logia",      "grafia",    "sofia",      "metria",     "fobia",
      "crazia",     "patia",     "gnia",       "filia",      "terapia",
      "nomia",      "scopia",    "fonia",      "archia",     "allegria",
      "poesia",     "cortesia",  "energia",    "strategia",  "magia",
      "bugia",      "follia",    "malattia",   "gelosia",    "simpatia",
      "armonia",    "ironia",    "agonia",     "teoria",     "categorie",
      "categoria",  "galleria",  "batteria",   "libreria",   "fattoria",
      "osteria",    "pizzeria",  "lotteria",   "biancheria", "borghesia",
      "fantasia",   "prateria",  "segreteria", "cavalleria", "artiglieria",
      "fanteria",   "scuderia",  "democrazia", "burocrazia", "aristocrazia",
      "diplomazia", "farmacia",  "polizia",    "pulizia",    "garanzia",
      "agenzia",    "profezia",  "via",        "vie",        "nostalgia",
      "allergia",   "chirurgia", "liturgia",   "regia",      "scia",
      "scie",
  });
  if (EndsWithAny(w, *kSuffixes)) return true;
  if (absl::EndsWith(w, "ie") && w.size() > 2 &&
      EndsWithAny(w.substr(0, w.size() - 1), *kSuffixes)) {
    return true;
  }
  return false;
}

bool HasSdrucciolaSuffix(absl::string_view w) {
  static const absl::NoDestructor<std::vector<absl::string_view>> kSuffixes({
      "bile",   "bili",   "colo",  "cola",   "coli",  "cole",   "fico",
      "fica",   "fici",   "fiche", "lico",   "lica",  "lici",   "liche",
      "tico",   "tica",   "tici",  "tiche",  "gico",  "gica",   "gici",
      "giche",  "mico",   "mica",  "mici",   "miche", "nico",   "nica",
      "nici",   "niche",  "pico",  "pica",   "pici",  "piche",  "rico",
      "rica",   "rici",   "riche", "sico",   "sica",  "sici",   "siche",
      "vico",   "vica",   "vici",  "viche",  "zico",  "zica",   "zici",
      "ziche",  "logo",   "loga",  "logi",   "loghe", "fono",   "fona",
      "foni",   "fone",   "grafo", "grafa",  "grafi", "grafe",  "metro",
      "metri",  "simo",   "sima",  "simi",   "sime",  "gine",   "gini",
      "dine",   "dini",   "mine",  "mini",   "vole",  "voli",   "tudine",
      "tudini", "dono",   "cono",  "gono",   "vono",  "tono",   "rono",
      "mono",   "lono",   "pono",  "gliono", "gnano", "sciano", "ciano",
      "giano",  "avano",  "evano", "ivano",  "arono", "erono",  "irono",
      "bero",   "sero",   "tero",  "cero",   "pero",  "vero",   "gero",
      "rero",   "gere",   "vere",  "tere",   "dere",  "cere",   "pere",
      "rere",   "gliere", "gnere", "scere",  "ndere", "mpere",  "rpere",
      "lgere",  "rgere",  "ngere", "ncere",  "rcere", "rdere",  "rtere",
      "stere",  "ttere",  "ddere", "ggere",  "ccere", "bbere",  "ppere",
      "ffere",  "erlo",   "erla",  "erli",   "erle",  "ersi",   "ermi",
      "erti",   "erci",   "erne",
  });
  return EndsWithAny(w, *kSuffixes);
}

bool HasLoanPenultSuffix(absl::string_view w) {
  static const absl::NoDestructor<std::vector<absl::string_view>> kSuffixes({
      "is",
      "us",
      "um",
      "id",
      "ard",
      "ing",
      "er",
      "al",
      "el",
      "il",
      "ol",
      "ul",
  });
  return EndsWithAny(w, *kSuffixes);
}

bool HasFinalDiphStress(absl::string_view w) {
  static const absl::NoDestructor<std::vector<absl::string_view>> kSuffixes({
      "rei",
      "rai",
      "sai",
      "vai",
      "dai",
      "fai",
      "stai",
      "assai",
      "ormai",
  });
  return EndsWithAny(w, *kSuffixes);
}

struct SuffixVqRule {
  absl::string_view suffix;
  absl::string_view vq;
};

absl::string_view LookupSuffixVq(absl::string_view w) {
  static const absl::NoDestructor<std::vector<SuffixVqRule>> kRules({
      {"mente", "e"}, {"zione", "o"}, {"sione", "o"}, {"zioni", "o"},
      {"sioni", "o"}, {"mento", "e"}, {"menti", "e"}, {"ente", "ɛ"},
      {"enti", "ɛ"},  {"enza", "ɛ"},  {"enze", "ɛ"},  {"endo", "ɛ"},
      {"esimo", "ɛ"}, {"iero", "ɛ"},  {"iera", "ɛ"},  {"ieri", "ɛ"},
      {"iere", "ɛ"},  {"orio", "ɔ"},  {"oria", "ɔ"},  {"ori", "o"},
      {"ore", "o"},   {"oso", "o"},   {"osa", "o"},   {"osi", "o"},
      {"ose", "o"},   {"one", "o"},   {"oni", "o"},   {"ezza", "e"},
      {"ezze", "e"},  {"ese", "e"},   {"esi", "e"},   {"essa", "e"},
      {"etto", "e"},  {"etta", "e"},  {"etti", "e"},  {"ette", "e"},
      {"ello", "ɛ"},  {"ella", "ɛ"},  {"elli", "ɛ"},  {"elle", "ɛ"},
      {"otto", "ɔ"},  {"otta", "ɔ"},  {"otti", "ɔ"},  {"otte", "ɔ"},
      {"ologo", "ɔ"}, {"ologa", "ɔ"}, {"ismo", "i"},
  });
  for (const SuffixVqRule& r : *kRules) {
    if (w.size() > r.suffix.size() && absl::EndsWith(w, r.suffix)) {
      return r.vq;
    }
  }
  return "";
}

enum class UnitKind {
  kVowel,
  kConsonant,
  kGlide,
};

struct Unit {
  UnitKind kind;
  char32_t v_ch;
  std::string ph;
  int len;
};

enum class SegKind {
  kConsonant,
  kGlide,
  kNucleusMono,
  kNucleusRise,
  kNucleusFall,
};

struct Segment {
  SegKind kind;
  std::string ph;
  int len;
  std::vector<char32_t> chars;
};

struct PhonToken {
  UnitKind kind;
  std::string gl;
  std::string body;
  std::string stress;
  int len;
};

std::vector<Unit> ParseUnits(const std::vector<char32_t>& w,
                             absl::string_view w_str) {
  static const absl::NoDestructor<std::vector<absl::string_view>> kIzzTsRoots({
      "pizz",
      "svizz",
      "tizz",
      "stizz",
      "guizz",
      "schizz",
      "frizz",
      "rizz",
      "drizz",
      "indirizz",
      "vizz",
      "bizz",
  });
  static const absl::NoDestructor<std::vector<absl::string_view>> kDblDzRoots({
      "mezz",
      "azzurr",
      "orizz",
      "bizzarr",
      "dozzin",
      "magazzin",
      "rozzo",
      "rozza",
      "rozzi",
      "rozze",
      "grezzo",
      "grezza",
  });
  static const absl::NoDestructor<std::vector<absl::string_view>> kInitTsRoots({
      "zii",
      "zio",
      "zia",
      "zie",
      "zup",
      "zuc",
      "zoc",
      "zop",
      "zamp",
      "zapp",
      "zecc",
      "zepp",
      "zing",
      "zitt",
      "zolf",
      "zucc",
      "zuff",
  });
  static const absl::NoDestructor<std::vector<absl::string_view>>
      kSingleDzRoots({
          "romanz",
          "bronz",
          "manz",
          "pranz",
          "monz",
          "fronz",
          "donzell",
          "zanzar",
          "zenzer",
          "zizzani",
          "garanz",
      });

  const bool has_stressed_ia = HasStressedIaSuffix(w_str);
  std::vector<Unit> out;
  out.reserve(w.size());
  const size_t n = w.size();
  size_t i = 0;
  while (i < n) {
    const char32_t c = w[i];
    const char32_t nx1 = (i + 1 < n) ? w[i + 1] : 0;
    const char32_t nx2 = (i + 2 < n) ? w[i + 2] : 0;
    if (IsVowel(c)) {
      out.push_back({UnitKind::kVowel, c, "", 1});
      i += 1;
      continue;
    }
    if (c == U'h') {
      i += 1;
      continue;
    }
    const bool dbl = (nx1 == c && c != U'h');
    if (c == U'c') {
      if (dbl && IsFrontVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "ʧ", 2});
        i += 2;
        if (i < n && w[i] == U'i' && i + 1 < n && IsVowel(w[i + 1]) &&
            !(has_stressed_ia && i + 2 == n)) {
          i += 1;
        }
        continue;
      }
      if (dbl) {
        out.push_back({UnitKind::kConsonant, 0, "k", 2});
        i += 2;
        if (i < n && w[i] == U'h') i += 1;
        continue;
      }
      if (nx1 == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "k", 1});
        i += 2;
        continue;
      }
      if (nx1 == U'q') {
        out.push_back({UnitKind::kConsonant, 0, "k", 2});
        i += 2;
        if (i < n && w[i] == U'u') {
          out.push_back({UnitKind::kGlide, 0, "w", 1});
          i += 1;
        }
        continue;
      }
      if (IsFrontVowel(nx1)) {
        out.push_back({UnitKind::kConsonant, 0, "ʧ", 1});
        i += 1;
        if (w[i] == U'i' && i + 1 < n && IsVowel(w[i + 1]) &&
            !(has_stressed_ia && i + 2 == n)) {
          i += 1;
        }
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "k", 1});
      i += 1;
      continue;
    }
    if (c == U'g') {
      if (dbl && IsFrontVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "ʤ", 2});
        i += 2;
        if (i < n && w[i] == U'i' && i + 1 < n && IsVowel(w[i + 1]) &&
            !(has_stressed_ia && i + 2 == n)) {
          i += 1;
        }
        continue;
      }
      if (dbl) {
        out.push_back({UnitKind::kConsonant, 0, "ɡ", 2});
        i += 2;
        continue;
      }
      if (nx1 == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "ɡ", 1});
        i += 2;
        continue;
      }
      if (nx1 == U'n') {
        out.push_back({UnitKind::kConsonant, 0, "ɲ", 2});
        i += 2;
        continue;
      }
      if (nx1 == U'l' && nx2 == U'i') {
        out.push_back({UnitKind::kConsonant, 0, "ʎ", 1});
        i += 2;
        if (i + 1 < n && IsVowel(w[i + 1])) i += 1;
        continue;
      }
      if (nx1 == U'u' && IsVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "ɡ", 1});
        out.push_back({UnitKind::kGlide, 0, "w", 1});
        i += 2;
        continue;
      }
      if (IsFrontVowel(nx1)) {
        out.push_back({UnitKind::kConsonant, 0, "ʤ", 1});
        i += 1;
        if (w[i] == U'i' && i + 1 < n && IsVowel(w[i + 1]) &&
            !(has_stressed_ia && i + 2 == n)) {
          i += 1;
        }
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "ɡ", 1});
      i += 1;
      continue;
    }
    if (c == U'q') {
      out.push_back({UnitKind::kConsonant, 0, "k", 1});
      if (nx1 == U'u') {
        out.push_back({UnitKind::kGlide, 0, "w", 1});
        i += 2;
      } else {
        i += 1;
      }
      continue;
    }
    if (c == U'n') {
      if (dbl) {
        out.push_back({UnitKind::kConsonant, 0, "n", 2});
        i += 2;
        continue;
      }
      if (nx1 == U'g' &&
          (nx2 == 0 || nx2 == U'a' || nx2 == U'o' || nx2 == U'u' ||
           nx2 == U'h' || nx2 == U'l' || nx2 == U'r' || nx2 == U'à' ||
           nx2 == U'ò' || nx2 == U'ó' || nx2 == U'ù' || nx2 == U'ú')) {
        out.push_back({UnitKind::kConsonant, 0, "ŋ", 1});
        i += 1;
        continue;
      }
      if (nx1 == U'k' && nx2 == 0) {
        out.push_back({UnitKind::kConsonant, 0, "ŋ", 1});
        i += 1;
        continue;
      }
      if (w_str == "pranzo") {
        out.push_back({UnitKind::kConsonant, 0, "ŋ", 1});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "n", 1});
      i += 1;
      continue;
    }
    if (c == U'r') {
      if (dbl) {
        out.push_back({UnitKind::kConsonant, 0, "r", 2});
        i += 2;
        continue;
      }
      const char32_t prev_ch = (i > 0) ? w[i - 1] : 0;
      bool is_tap = false;
      if (IsVowel(prev_ch) && (nx1 == U'à' || nx1 == U'ò') && i + 2 == n) {
        is_tap = true;
      } else if (prev_ch == U'a' && nx1 == U'a' && w_str != "garage" &&
                 w_str != "garanzie") {
        is_tap = true;
      } else if (absl::StartsWith(w_str, "parol") ||
                 absl::StartsWith(w_str, "operai")) {
        is_tap = true;
      }
      out.push_back({UnitKind::kConsonant, 0, is_tap ? "ɾ" : "r", 1});
      i += 1;
      continue;
    }
    if (c == U's') {
      if (nx1 == U'c' && IsFrontVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "ʃ", 1});
        i += 2;
        if (w[i] == U'i' && i + 1 < n && IsVowel(w[i + 1]) &&
            !(has_stressed_ia && i + 2 == n)) {
          i += 1;
        }
        continue;
      }
      if (nx1 == U'c' && nx2 == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "s", 1});
        out.push_back({UnitKind::kConsonant, 0, "k", 1});
        i += 3;
        continue;
      }
      if (dbl) {
        out.push_back({UnitKind::kConsonant, 0, "s", 2});
        i += 2;
        continue;
      }
      const bool prev_v = (!out.empty() && out.back().kind == UnitKind::kVowel);
      const bool voiced =
          (prev_v && IsVowel(nx1)) ||
          (nx1 == U'b' || nx1 == U'd' || nx1 == U'g' || nx1 == U'l' ||
           nx1 == U'm' || nx1 == U'n' || nx1 == U'r' || nx1 == U'v');
      out.push_back({UnitKind::kConsonant, 0, voiced ? "z" : "s", 1});
      i += 1;
      continue;
    }
    if (c == U'z') {
      if (dbl) {
        const bool is_dz = (i > 0 && w[i - 1] == U'i' &&
                            !StartsWithAny(w_str, *kIzzTsRoots)) ||
                           StartsWithAny(w_str, *kDblDzRoots);
        out.push_back({UnitKind::kConsonant, 0, is_dz ? "ʣ" : "ʦ", 2});
        i += 2;
        continue;
      }
      const bool is_dz = (i == 0 && !StartsWithAny(w_str, *kInitTsRoots)) ||
                         StartsWithAny(w_str, *kSingleDzRoots);
      out.push_back({UnitKind::kConsonant, 0, is_dz ? "ʣ" : "ʦ", 1});
      i += 1;
      continue;
    }
    if (c == U'x') {
      out.push_back({UnitKind::kConsonant, 0, "k", 1});
      out.push_back({UnitKind::kConsonant, 0, "s", 1});
      i += 1;
      continue;
    }
    if (c == U'j') {
      out.push_back({UnitKind::kGlide, 0, "j", 1});
      i += 1;
      continue;
    }
    if (c == U'y') {
      out.push_back({UnitKind::kVowel, U'i', "", 1});
      i += 1;
      continue;
    }
    if (c == U'w') {
      out.push_back({UnitKind::kGlide, 0, "w", 1});
      i += 1;
      continue;
    }
    std::string ph;
    AppendUtf8(c, ph);
    if (dbl) {
      out.push_back({UnitKind::kConsonant, 0, ph, 2});
      i += 2;
      continue;
    }
    out.push_back({UnitKind::kConsonant, 0, ph, 1});
    i += 1;
  }
  return out;
}

std::vector<Segment> BuildNuclei(const std::vector<Unit>& u_list) {
  std::vector<Segment> segs;
  segs.reserve(u_list.size());
  size_t i = 0;
  while (i < u_list.size()) {
    const Unit& u = u_list[i];
    if (u.kind != UnitKind::kVowel) {
      segs.push_back({u.kind == UnitKind::kConsonant ? SegKind::kConsonant
                                                     : SegKind::kGlide,
                      u.ph,
                      u.len,
                      {}});
      i += 1;
      continue;
    }
    std::vector<char32_t> run;
    while (i < u_list.size() && u_list[i].kind == UnitKind::kVowel) {
      run.push_back(u_list[i].v_ch);
      i += 1;
    }
    size_t j = 0;
    while (j < run.size()) {
      const char32_t ch = run[j];
      const char32_t nx = (j + 1 < run.size()) ? run[j + 1] : 0;
      const bool rise_i =
          (ch == U'i' && nx != 0 && nx != U'i' && nx != U'ì' && nx != U'í');
      const bool rise_u =
          (ch == U'u' && nx != 0 && nx != U'u' && nx != U'ù' && nx != U'ú' &&
           nx != U'i' && nx != U'ì' && nx != U'í');
      if (rise_i || rise_u) {
        std::vector<char32_t> nuc = {ch, nx};
        j += 2;
        if (j < run.size() && (run[j] == U'i' || run[j] == U'u') &&
            j + 1 >= run.size()) {
          nuc.push_back(run[j]);
          j += 1;
        }
        segs.push_back({SegKind::kNucleusRise, "", 1, nuc});
        continue;
      }
      if (ch != U'i' && ch != U'u' && ch != U'ì' && ch != U'ù' &&
          (nx == U'i' || nx == U'u') &&
          (j + 2 >= run.size() || run[j + 2] == U'i' || run[j + 2] == U'u')) {
        segs.push_back({SegKind::kNucleusFall, "", 1, {ch, nx}});
        j += 2;
        continue;
      }
      segs.push_back({SegKind::kNucleusMono, "", 1, {ch}});
      j += 1;
    }
  }
  return segs;
}

bool IsNucleusSeg(const Segment& s) {
  return s.kind == SegKind::kNucleusMono || s.kind == SegKind::kNucleusRise ||
         s.kind == SegKind::kNucleusFall;
}

int FindNucleusOrder(const std::vector<size_t>& idx, size_t si) {
  for (size_t k = 0; k < idx.size(); ++k) {
    if (idx[k] == si) return static_cast<int>(k);
  }
  return -1;
}

std::vector<PhonToken> WordTokens(absl::string_view word,
                                  absl::string_view root_hint = "") {
  const std::string w_str = ToLowerUtf8(word);
  const std::string lookup_w =
      root_hint.empty() ? w_str : ToLowerUtf8(root_hint);
  const std::vector<char32_t> w_cps = Utf8ToCodePoints(w_str);
  const std::vector<char32_t> lookup_cps = Utf8ToCodePoints(lookup_w);
  if (w_cps.empty()) return {};

  const std::vector<Unit> u_list = ParseUnits(w_cps, w_str);
  std::vector<Segment> segs = BuildNuclei(u_list);

  if (HasStressedIaSuffix(lookup_w)) {
    for (int k = static_cast<int>(segs.size()) - 1; k >= 0; --k) {
      if (IsNucleusSeg(segs[k])) {
        if (segs[k].kind == SegKind::kNucleusRise && !segs[k].chars.empty() &&
            segs[k].chars[0] == U'i') {
          std::vector<char32_t> tail(segs[k].chars.begin() + 1,
                                     segs[k].chars.end());
          segs[k] = {SegKind::kNucleusMono, "", 1, {U'ì'}};
          segs.insert(segs.begin() + k + 1,
                      {SegKind::kNucleusMono, "", 1, tail});
        }
        break;
      }
    }
  }

  if (GetHiatusFirst().contains(lookup_w)) {
    std::vector<Segment> new_segs;
    new_segs.reserve(segs.size() + 2);
    for (const Segment& sg : segs) {
      if ((sg.kind == SegKind::kNucleusRise ||
           sg.kind == SegKind::kNucleusFall) &&
          sg.chars.size() >= 2) {
        new_segs.push_back({SegKind::kNucleusMono, "", 1, {sg.chars[0]}});
        std::vector<char32_t> tail(sg.chars.begin() + 1, sg.chars.end());
        new_segs.push_back({SegKind::kNucleusMono, "", 1, tail});
      } else {
        new_segs.push_back(sg);
      }
    }
    segs = new_segs;
  }

  std::vector<size_t> idx;
  bool any_accent = false;
  int accent_k = -1;
  for (size_t si = 0; si < segs.size(); ++si) {
    if (IsNucleusSeg(segs[si])) {
      const int k = static_cast<int>(idx.size());
      idx.push_back(si);
      for (char32_t c : segs[si].chars) {
        if (IsAccVowel(c) && accent_k == -1) {
          any_accent = true;
          accent_k = k;
        }
      }
    }
  }

  int st = -1;
  if (!idx.empty()) {
    if (any_accent) {
      st = accent_k;
    } else if (idx.size() == 1) {
      st = 0;
    } else if (!w_cps.empty() && !IsVowel(w_cps.back())) {
      st = static_cast<int>(idx.size()) - 1;
    } else if (GetHiatusFirst().contains(w_str)) {
      st = 0;
    } else if (idx.size() >= 3 && HasSdrucciolaSuffix(w_str) &&
               !GetSdrucciolaExempt().contains(w_str)) {
      st = static_cast<int>(idx.size()) - 3;
    } else {
      st = static_cast<int>(idx.size()) - 2;
    }

    if (idx.size() >= 2 && !lookup_cps.empty() && !IsVowel(lookup_cps.back())) {
      if (HasLoanPenultSuffix(lookup_w) &&
          !GetApocopeFinalWords().contains(lookup_w)) {
        st = static_cast<int>(idx.size()) - 2;
      }
    }
    if (idx.size() >= 2 && HasFinalDiphStress(lookup_w)) {
      st = static_cast<int>(idx.size()) - 1;
    } else if (idx.size() >= 3 && !any_accent) {
      if (GetPenultWords().contains(lookup_w)) {
        st = static_cast<int>(idx.size()) - 2;
      } else if (GetSdrWords().contains(lookup_w)) {
        st = static_cast<int>(idx.size()) - 3;
      }
    }
  }

  for (size_t si = 0; si < segs.size(); ++si) {
    if (segs[si].kind == SegKind::kConsonant && segs[si].ph == "ɲ") {
      const bool prev_is_stressed_v = (si > 0 && IsNucleusSeg(segs[si - 1]) &&
                                       FindNucleusOrder(idx, si - 1) == st);
      segs[si].len = prev_is_stressed_v ? 2 : 1;
    }
  }

  const absl::string_view vq = LookupSuffixVq(lookup_w);
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kNjWords({
      "riunione",
      "riunioni",
      "opinione",
      "opinioni",
      "unione",
      "unioni",
      "comunione",
      "milione",
      "milioni",
      "provenienti",
      "proveniente",
  });
  static const absl::NoDestructor<std::vector<absl::string_view>> kTsjPrefixes({
      "delizi",
      "ambizi",
      "spazi",
      "vizi",
      "negozi",
      "funzionar",
  });

  std::vector<PhonToken> toks;
  toks.reserve(segs.size());
  for (size_t si = 0; si < segs.size(); ++si) {
    const Segment& s = segs[si];
    if (s.kind == SegKind::kConsonant) {
      toks.push_back({UnitKind::kConsonant, "", s.ph, "", s.len});
      continue;
    }
    if (s.kind == SegKind::kGlide) {
      toks.push_back({UnitKind::kGlide, "", s.ph, "", 1});
      continue;
    }
    const int k = FindNucleusOrder(idx, si);
    const bool stressed = (k == st);
    const PhonToken* prev = toks.empty() ? nullptr : &toks.back();
    if (s.kind == SegKind::kNucleusRise) {
      const char32_t g = s.chars[0];
      const char32_t v = s.chars[1];
      const absl::string_view pc =
          (prev != nullptr && prev->kind == UnitKind::kConsonant)
              ? absl::string_view(prev->body)
              : absl::string_view();
      const bool pc_gem =
          (prev != nullptr && prev->kind == UnitKind::kConsonant &&
           prev->len == 2);
      const bool prev2_c = (toks.size() >= 2 &&
                            toks[toks.size() - 2].kind == UnitKind::kConsonant);
      const absl::string_view prev2_ph =
          prev2_c ? absl::string_view(toks[toks.size() - 2].body)
                  : absl::string_view();
      bool word_init = true;
      for (const PhonToken& t : toks) {
        if (t.kind != UnitKind::kConsonant) {
          word_init = false;
          break;
        }
      }
      const bool is_final_seg = (si + 1 == segs.size());
      std::string vb(BaseVowel(v));
      bool glide_i = false;
      if (!stressed && is_final_seg && g == U'i') {
        glide_i =
            (pc == "ʦ" && (lookup_w == "grazie" || lookup_w == "notizie" ||
                           lookup_w == "amicizie")) ||
            (pc == "d" && lookup_w == "episodio");
      } else {
        const bool pbgfdv = (pc == "p" || pc == "b" || pc == "ɡ" || pc == "f" ||
                             pc == "d" || pc == "v");
        const bool pbgfdv_ok =
            pbgfdv && !(prev2_c && prev2_ph != "s" && prev2_ph != "n" &&
                        prev2_ph != "r" &&
                        !(prev2_ph == "m" && (vb == "a" || vb == "o")));
        glide_i =
            prev == nullptr ||
            (!pc_gem &&
             ((word_init && !prev2_c) || pbgfdv_ok || (pc == "k" && !prev2_c) ||
              (pc == "t" && prev2_c && prev2_ph == "s") ||
              (pc == "s" && prev2_c && prev2_ph == "n") ||
              (pc == "t" && (absl::EndsWith(lookup_w, "tiene") ||
                             absl::EndsWith(lookup_w, "tieni"))) ||
              (pc == "n" && kNjWords->contains(lookup_w)) ||
              (pc == "ʦ" &&
               ((vb != "o" && vb != "ɔ") ||
                StartsWithAny(lookup_w, *kTsjPrefixes)) &&
               !is_final_seg))) ||
            (pc_gem && (pc == "s" || pc == "p" || pc == "b"));
        if (word_init && pc == "r") {
          glide_i = false;
        }
      }
      std::string extra;
      for (size_t ci = 2; ci < s.chars.size(); ++ci) {
        absl::StrAppend(&extra, BaseVowel(s.chars[ci]));
      }
      if (stressed) {
        if (!IsAccVowel(v)) {
          if (vb == "e") {
            if (GetOpenEWords().contains(lookup_w)) {
              vb = "ɛ";
            } else if (GetClosedEWords().contains(lookup_w)) {
              vb = "e";
            } else if (vq == "ɛ" && k >= static_cast<int>(idx.size()) - 2) {
              vb = "ɛ";
            } else if (vq == "e" && k >= static_cast<int>(idx.size()) - 2) {
              vb = "e";
            } else if (g == U'i') {
              vb = "ɛ";
            }
          } else if (vb == "o") {
            if (GetOpenOWords().contains(lookup_w)) {
              vb = "ɔ";
            } else if (GetClosedOWords().contains(lookup_w)) {
              vb = "o";
            } else if (vq == "ɔ" && k >= static_cast<int>(idx.size()) - 2) {
              vb = "ɔ";
            } else if (vq == "o" && k >= static_cast<int>(idx.size()) - 2) {
              vb = "o";
            } else if (g == U'u') {
              vb = "ɔ";
            }
          }
        }
        const std::string gl =
            (g == U'i') ? (glide_i ? "j" : "i") : (toks.empty() ? "w" : "ʊ");
        toks.push_back({UnitKind::kVowel, gl, absl::StrCat(vb, extra), "ˈ", 1});
      } else {
        std::string gl;
        if (g == U'i') {
          if (vb == "u" && st >= 0 && k < st) {
            vb = "ʊ";
          }
          gl = glide_i ? "j" : "i";
        } else {
          gl = (st >= 0 && k < st) ? "ʊ" : "u";
        }
        toks.push_back({UnitKind::kVowel, gl, absl::StrCat(vb, extra), "", 1});
      }
      continue;
    }

    const char32_t v = s.chars[0];
    std::string vb(BaseVowel(v));
    if (stressed && !IsAccVowel(v)) {
      if (vb == "e") {
        if (GetOpenEWords().contains(lookup_w)) {
          vb = "ɛ";
        } else if (GetClosedEWords().contains(lookup_w)) {
          vb = "e";
        } else if ((vq == "ɛ" || vq == "e") &&
                   k >= static_cast<int>(idx.size()) - 2) {
          vb = std::string(vq);
        }
      } else if (vb == "o") {
        if (GetOpenOWords().contains(lookup_w)) {
          vb = "ɔ";
        } else if (GetClosedOWords().contains(lookup_w)) {
          vb = "o";
        } else if ((vq == "ɔ" || vq == "o") &&
                   k >= static_cast<int>(idx.size()) - 2) {
          vb = std::string(vq);
        }
      }
    }

    if (s.kind == SegKind::kNucleusFall) {
      const absl::string_view off = BaseVowel(s.chars[1]);
      std::string body = absl::StrCat(vb, off);
      const bool nxt_is_c =
          (si + 1 < segs.size() && segs[si + 1].kind == SegKind::kConsonant);
      if (body == "ai") {
        body = nxt_is_c ? "I" : "aj";
      } else if (body == "au") {
        body = "aʊ";
      } else if (body == "eu" || body == "ɛu") {
        body = stressed ? "ɛʊ" : "eʊ";
      } else if (body == "ei" || body == "ɛi") {
        body = stressed ? "ɛi" : "ei";
      } else if (body == "oi" || body == "ɔi") {
        const bool is_open_oi =
            stressed &&
            (lookup_w == "poi" || lookup_w == "coi" || lookup_w == "puoi" ||
             lookup_w == "vuoi" || lookup_w == "tuoi" || lookup_w == "suoi");
        body = is_open_oi ? "ɔi" : "oi";
      }
      toks.push_back({UnitKind::kVowel, "", body, stressed ? "ˈ" : "", 1});
      continue;
    }

    if (!stressed) {
      if (vb == "u" && st >= 0 && (k < st || absl::EndsWith(lookup_w, "um"))) {
        vb = "ʊ";
      } else if (vb == "i" && si + 1 == segs.size() && si > 0 &&
                 !IsNucleusSeg(segs[si - 1])) {
        vb = "ɪ";
      }
    }
    toks.push_back({UnitKind::kVowel, "", vb, stressed ? "ˈ" : "", 1});
  }
  return toks;
}

void DestressTokens(std::vector<PhonToken>& toks) {
  for (size_t i = 0; i < toks.size(); ++i) {
    PhonToken& t = toks[i];
    if (t.kind == UnitKind::kVowel) {
      t.stress.clear();
      if (t.body == "u") {
        t.body = "ʊ";
      } else if (t.body == "i" && i + 1 == toks.size() && i > 0 &&
                 toks[i - 1].kind != UnitKind::kVowel) {
        t.body = "ɪ";
      } else if (t.body == "ɛ") {
        t.body = "e";
      } else if (t.body == "ɔ") {
        t.body = "o";
      }
    }
  }
}

std::string RenderTokens(const std::vector<PhonToken>& toks) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kGemMap({
          {"t", "tː"}, {"p", "pː"}, {"k", "kː"}, {"b", "bː"}, {"d", "dː"},
          {"ɡ", "ɡː"}, {"ʧ", "ʧː"}, {"ʤ", "ʤː"}, {"ʦ", "ʦː"}, {"ʣ", "ʣː"},
          {"s", "S"},  {"r", "rɾ"}, {"l", "ll"}, {"n", "nn"}, {"m", "mm"},
          {"f", "ff"}, {"v", "vv"}, {"ʃ", "ʃ"},  {"ʎ", "ʎ"},  {"ɲ", "ɲɲ"},
          {"z", "S"},
      });
  std::string out;
  out.reserve(toks.size() * 3);
  for (const PhonToken& t : toks) {
    if (t.kind == UnitKind::kConsonant) {
      if (t.len == 2) {
        if (const auto it = kGemMap->find(t.body); it != kGemMap->end()) {
          out.append(it->second);
        } else {
          absl::StrAppend(&out, t.body, t.body);
        }
      } else {
        out.append(t.body);
      }
    } else if (t.kind == UnitKind::kGlide) {
      out.append(t.body);
    } else {
      absl::StrAppend(&out, t.gl, t.stress, t.body);
    }
  }
  return out;
}

absl::string_view ItUnitWord(int n) {
  switch (n) {
    case 0:
      return "ʣˈɛɾo";
    case 1:
      return "ˈuno";
    case 2:
      return "dˈue";
    case 3:
      return "trˈe";
    case 4:
      return "kwˈatːro";
    case 5:
      return "ʧˈinkwe";
    case 6:
      return "sˈɛi";
    case 7:
      return "sˈɛtːe";
    case 8:
      return "ˈɔtːo";
    case 9:
      return "nˈɔve";
    case 10:
      return "djˈɛʧɪ";
    case 11:
      return "ˈundiʧɪ";
    case 12:
      return "dˈodiʧɪ";
    case 13:
      return "trˈediʧɪ";
    case 14:
      return "kwatːˈordiʧɪ";
    case 15:
      return "kwˈimdiʧɪ";
    case 16:
      return "sˈediʧɪ";
    case 17:
      return "diʧaSˈɛtːe";
    case 18:
      return "diʧˈɔtːo";
    case 19:
      return "diʧannˈɔve";
    default:
      return "";
  }
}

absl::string_view ItTenWord(int t) {
  switch (t) {
    case 20:
      return "vˈentɪ";
    case 30:
      return "trˈeːnta";
    case 40:
      return "kwarˈanta";
    case 50:
      return "ʧinkwˈanta";
    case 60:
      return "seSˈanta";
    case 70:
      return "setːˈanta";
    case 80:
      return "otːˈanta";
    case 90:
      return "novˈanta";
    default:
      return "";
  }
}

absl::string_view ItTenStem(int t) {
  switch (t) {
    case 20:
      return "vent";
    case 30:
      return "trˈeːnt";
    case 40:
      return "kwarˈant";
    case 50:
      return "ʧinkwˈant";
    case 60:
      return "seSˈant";
    case 70:
      return "setːˈant";
    case 80:
      return "otːˈant";
    case 90:
      return "novˈant";
    default:
      return "";
  }
}

absl::string_view ItHundredWord(int h) {
  switch (h) {
    case 100:
      return "ʧˈɛnto";
    case 200:
      return "dʊeʧˈɛnto";
    case 300:
      return "treʧˈɛnto";
    case 400:
      return "kwatːroʧˈɛnto";
    case 500:
      return "ʧɪnkʊeʧˈɛnto";
    case 600:
      return "seiʧˈɛnto";
    case 700:
      return "sɛtːeʧˈɛnto";
    case 800:
      return "ɔtːoʧˈɛnto";
    case 900:
      return "noveʧˈɛnto";
    default:
      return "";
  }
}

std::string ItUnder100(int n) {
  if (n < 20) return std::string(ItUnitWord(n));
  const int t = (n / 10) * 10;
  const int u = n % 10;
  if (u == 0) return std::string(ItTenWord(t));
  if (n == 26) return "ventisˈɛj";
  if (u == 1 || u == 8) {
    return absl::StrCat(ItTenStem(t), ItUnitWord(u));
  }
  if (t == 20) {
    return absl::StrCat("venti", ItUnitWord(u));
  }
  return absl::StrCat(ItTenWord(t), ItUnitWord(u));
}

std::string ItUnder1000(int n) {
  if (n < 100) return ItUnder100(n);
  const int h = (n / 100) * 100;
  const int r = n % 100;
  if (r == 0) return std::string(ItHundredWord(h));
  return absl::StrCat(ItHundredWord(h), " ", ItUnder100(r));
}

std::string ItNumber(absl::string_view s) {
  if (s.size() == 2 && s[0] == '0') {
    if (s == "00") return "ʣˈɛɾo ʣˈɛɾo";
    return absl::StrCat("ʣˈɛɾo ", ItUnitWord(s[1] - '0'));
  }
  if (s == "000") {
    return "ʣˈɛɾoʣˈɛɾo ʣˈɛɾo";
  }
  int n = 0;
  if (!absl::SimpleAtoi(s, &n) || n < 0) {
    return std::string(s);
  }
  if (n < 1000) return ItUnder1000(n);
  if (n < 1000000) {
    const int q = n / 1000;
    const int r = n % 1000;
    std::string out =
        (q == 1) ? "mˈille" : absl::StrCat(ItUnder1000(q), " mˈiːla");
    if (r > 0) {
      absl::StrAppend(&out, " ", ItUnder1000(r));
    }
    return out;
  }
  if (n < 1000000000) {
    const int m = n / 1000000;
    const int r = n % 1000000;
    std::string out =
        (m == 1) ? "ʊnmiliˈone" : absl::StrCat(ItUnder1000(m), " miliˈonɪ");
    if (r > 0) {
      absl::StrAppend(&out, " ", ItNumber(absl::StrCat(r)));
    }
    return out;
  }
  return std::string(s);
}

struct RawToken {
  std::string text;
};

std::vector<RawToken> TokenizeText(absl::string_view text) {
  std::string norm;
  norm.reserve(text.size() + 8);
  size_t pos = 0;
  while (pos < text.size()) {
    if (absl::StartsWith(text.substr(pos), "S.p.A.")) {
      norm.append("SPA_TOKEN.");
      pos += 6;
      continue;
    }
    char32_t cp = 0;
    ConsumeUtf8CodePoint(text, pos, cp);
    if (cp == U'’') cp = U'\'';
    AppendUtf8(cp, norm);
  }

  const std::vector<char32_t> cps = Utf8ToCodePoints(norm);
  std::vector<RawToken> tokens;
  const size_t n = cps.size();
  size_t i = 0;
  while (i < n) {
    const char32_t c = cps[i];
    if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r') {
      i += 1;
      continue;
    }
    if (c >= U'0' && c <= U'9') {
      size_t j = i;
      while (j < n && cps[j] >= U'0' && cps[j] <= U'9') j += 1;
      if (j < n && cps[j] == U'º') {
        j += 1;
      } else if (j + 1 < n &&
                 (cps[j] == U'.' || cps[j] == U',' || cps[j] == U':') &&
                 cps[j + 1] >= U'0' && cps[j + 1] <= U'9') {
        j += 1;
        while (j < n && cps[j] >= U'0' && cps[j] <= U'9') j += 1;
      }
      std::vector<char32_t> slice(cps.begin() + i, cps.begin() + j);
      tokens.push_back({CodePointsToUtf8(slice)});
      i = j;
      continue;
    }
    if (IsWordChar(c)) {
      size_t j = i;
      while (j < n && IsWordChar(cps[j])) j += 1;
      if (j < n && cps[j] == U'\'') j += 1;
      std::vector<char32_t> slice(cps.begin() + i, cps.begin() + j);
      tokens.push_back({CodePointsToUtf8(slice)});
      i = j;
      continue;
    }
    std::string single;
    AppendUtf8(c, single);
    tokens.push_back({single});
    i += 1;
  }
  return tokens;
}

bool IsClosingPunct(absl::string_view s) {
  if (s.empty()) return false;
  size_t pos = 0;
  char32_t cp = 0;
  if (!ConsumeUtf8CodePoint(s, pos, cp)) return false;
  switch (cp) {
    case U'.':
    case U',':
    case U';':
    case U':':
    case U'!':
    case U'?':
    case U'…':
    case U')':
    case U'”':
    case U'»':
    case U'"':
      return true;
    default:
      return false;
  }
}

bool IsOpeningPunct(absl::string_view s) {
  return s == "¡" || s == "¿" || s == "(" || s == "“" || s == "«";
}

}  // namespace

ItalianG2p::ItalianG2p(int /*unused*/) {}

absl::StatusOr<ItalianG2p> ItalianG2p::Create() { return ItalianG2p(0); }

absl::StatusOr<std::string> ItalianG2p::DoWordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::string w = ToLowerUtf8(word);
  if (w.empty()) return std::string();
  if (const auto it = custom_lexicon.find(w); it != custom_lexicon.end()) {
    return it->second;
  }
  if (const auto it = GetWordOverrides().find(w);
      it != GetWordOverrides().end()) {
    return it->second;
  }
  if (const auto it = GetLetterNames().find(w); it != GetLetterNames().end()) {
    return it->second;
  }
  if (w == "in") return "ˈiːn";
  if (w == "è") return "ˈɛː";
  if (w == "dei") return "dˌA";
  return RenderTokens(WordTokens(w));
}

absl::StatusOr<std::string> ItalianG2p::DoTextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::vector<RawToken> raw = TokenizeText(text);
  if (raw.empty()) return std::string();

  std::vector<std::string> out;
  out.reserve(raw.size());
  std::string pending;
  const size_t n = raw.size();
  size_t i = 0;
  while (i < n) {
    const std::string& tok = raw[i].text;
    const absl::string_view nxt_tok =
        (i + 1 < n) ? absl::string_view(raw[i + 1].text) : absl::string_view();
    const absl::string_view prev_tok =
        (i > 0) ? absl::string_view(raw[i - 1].text) : absl::string_view();

    if (tok == "SPA_TOKEN") {
      out.push_back("ˈɛSe.pˈi.ˈaː");
      i += 1;
      continue;
    }
    if (tok == "€" && !nxt_tok.empty() && nxt_tok[0] >= '0' &&
        nxt_tok[0] <= '9') {
      out.push_back("ˈɛʊro");
      i += 1;
      continue;
    }
    if (tok == "-" && !nxt_tok.empty() && nxt_tok[0] >= '0' &&
        nxt_tok[0] <= '9') {
      out.push_back("mˈeno");
      i += 1;
      continue;
    }
    if (tok == "+" && !nxt_tok.empty() && nxt_tok[0] >= '0' &&
        nxt_tok[0] <= '9') {
      out.push_back("pjˈu");
      i += 1;
      continue;
    }
    if (tok == "°") {
      out.push_back("ɡrˈaːdɪ");
      i += 1;
      continue;
    }
    if (tok == "%") {
      out.push_back("perʧˈɛŋto");
      i += 1;
      continue;
    }
    if (tok == "/") {
      out.push_back("bˈarɾa");
      i += 1;
      continue;
    }
    if (tok == "1º") {
      out.push_back("prˈimo");
      i += 1;
      continue;
    }
    if (tok == "2º") {
      out.push_back("sekˈondo");
      i += 1;
      continue;
    }
    if (!tok.empty() && tok[0] >= '0' && tok[0] <= '9') {
      size_t sep_pos = std::string::npos;
      for (size_t k = 0; k < tok.size(); ++k) {
        if (tok[k] == '.' || tok[k] == ',' || tok[k] == ':') {
          sep_pos = k;
          break;
        }
      }
      if (sep_pos != std::string::npos) {
        const absl::string_view p1 = absl::string_view(tok).substr(0, sep_pos);
        const char sep = tok[sep_pos];
        const absl::string_view p2 = absl::string_view(tok).substr(sep_pos + 1);
        std::string s_num = absl::StrCat(
            ItNumber(p1), absl::string_view(&sep, 1), ItNumber(p2));
        if (i + 2 < n && raw[i + 1].text == "," &&
            IsAllDigits(raw[i + 2].text)) {
          absl::StrAppend(&s_num, ",", ItNumber(raw[i + 2].text));
          i += 2;
        }
        out.push_back(s_num);
        i += 1;
        continue;
      }
      if (IsAllDigits(tok)) {
        out.push_back(ItNumber(tok));
        i += 1;
        continue;
      }
    }

    if (IsAlphaWord(tok)) {
      if (absl::EndsWith(tok, "'")) {
        pending.append(tok.substr(0, tok.size() - 1));
        i += 1;
        continue;
      }
      const std::string lw = pending.empty()
                                 ? ToLowerUtf8(tok)
                                 : ToLowerUtf8(absl::StrCat(pending, tok));
      const std::string head = ToLowerUtf8(pending);
      pending.clear();
      const std::string key = ToLowerUtf8(tok);

      if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
        out.push_back(it->second);
        i += 1;
        continue;
      }
      if (head.empty()) {
        const std::string prev_low = ToLowerUtf8(prev_tok);
        const std::string nxt_low = ToLowerUtf8(nxt_tok);
        if (key == "dei") {
          out.push_back("dˌA");
          i += 1;
          continue;
        }
        if (key == "stasera") {
          out.push_back("stˌa sˈera");
          i += 1;
          continue;
        }
        if (key == "subito" && (prev_low == "ha" || prev_low == "aveva")) {
          out.push_back("sʊbˈiːto");
          i += 1;
          continue;
        }
        if (key == "me" && prev_low == "da" && IsAlphaWord(nxt_tok)) {
          out.push_back("me");
          i += 1;
          continue;
        }
        if (key == "vuole" && nxt_low == "con") {
          out.push_back("vʊˌɔle");
          i += 1;
          continue;
        }
        if (const auto it = GetWordOverrides().find(key);
            it != GetWordOverrides().end()) {
          out.push_back(it->second);
          i += 1;
          continue;
        }
        if (const auto it = GetLetterNames().find(key);
            it != GetLetterNames().end() &&
            ((key.size() == 1 && !GetAtoneMono().contains(key)) ||
             key == "km" || key == "kg" || key == "cm")) {
          out.push_back(it->second);
          i += 1;
          continue;
        }
      }

      std::vector<PhonToken> toks = WordTokens(
          lw, head.empty() ? absl::string_view() : absl::string_view(key));
      if (head.empty()) {
        if (GetAtoneMono().contains(key)) {
          DestressTokens(toks);
        } else if (GetAtonePoly().contains(key) &&
                   !(key == "quello" && ToLowerUtf8(nxt_tok) == "che")) {
          for (PhonToken& t : toks) {
            if (t.kind == UnitKind::kVowel && t.stress == "ˈ") {
              t.stress = "ˌ";
              if (t.body == "u") t.body = "ʊ";
            }
          }
        }
      }
      out.push_back(RenderTokens(toks));
    } else {
      if (!pending.empty()) {
        out.push_back(RenderTokens(WordTokens(pending)));
        pending.clear();
      }
      out.push_back(tok);
    }
    i += 1;
  }

  std::string s;
  s.reserve(text.size() * 2);
  for (const std::string& o : out) {
    if (IsClosingPunct(o)) {
      while (!s.empty() && s.back() == ' ') s.pop_back();
      absl::StrAppend(&s, o, " ");
    } else if (IsOpeningPunct(o)) {
      continue;
    } else {
      absl::StrAppend(&s, o, " ");
    }
  }
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

}  // namespace litert::omni::tts::kokoro
