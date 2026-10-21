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

#include "omni/text2speech/kokoro/portuguese_g2p.h"

#include <array>
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

bool IsPortugueseLetter(char32_t cp) {
  const char32_t lc = ToLowerLatin(cp);
  if (lc >= U'a' && lc <= U'z') return true;
  switch (lc) {
    case U'á':
    case U'à':
    case U'â':
    case U'ã':
    case U'é':
    case U'ê':
    case U'í':
    case U'ó':
    case U'ô':
    case U'õ':
    case U'ú':
    case U'ü':
    case U'ç':
      return true;
    default:
      return false;
  }
}

bool IsPortugueseUpper(char32_t cp) {
  if (cp >= U'A' && cp <= U'Z') return true;
  return ToLowerLatin(cp) != cp;
}

bool IsPortugueseVowel(char32_t cp) {
  switch (cp) {
    case U'a':
    case U'e':
    case U'i':
    case U'o':
    case U'u':
    case U'á':
    case U'é':
    case U'í':
    case U'ó':
    case U'ú':
    case U'â':
    case U'ê':
    case U'ô':
    case U'ã':
    case U'õ':
    case U'à':
    case U'ü':
      return true;
    default:
      return false;
  }
}

bool IsAccStressVowel(char32_t cp) {
  switch (cp) {
    case U'á':
    case U'é':
    case U'í':
    case U'ó':
    case U'ú':
    case U'â':
    case U'ê':
    case U'ô':
    case U'à':
    case U'ã':
    case U'õ':
      return true;
    default:
      return false;
  }
}

bool IsFrontVowel(char32_t cp) {
  return cp == U'e' || cp == U'i' || cp == U'é' || cp == U'ê' || cp == U'í';
}

char32_t BaseVowel(char32_t cp) {
  switch (cp) {
    case U'á':
    case U'à':
    case U'â':
      return U'a';
    case U'ã':
      return U'ã';
    case U'é':
      return U'ɛ';
    case U'ê':
      return U'e';
    case U'í':
      return U'i';
    case U'ó':
      return U'ɔ';
    case U'ô':
      return U'o';
    case U'õ':
      return U'õ';
    case U'ú':
    case U'ü':
      return U'u';
    default:
      return cp;
  }
}

std::vector<char32_t> ToCodepointsLower(absl::string_view text) {
  std::vector<char32_t> cps;
  size_t pos = 0;
  char32_t cp = 0;
  while (ConsumeUtf8CodePoint(text, pos, cp)) {
    if (cp == 0x0301 && !cps.empty()) {
      switch (cps.back()) {
        case U'a':
          cps.back() = U'á';
          continue;
        case U'e':
          cps.back() = U'é';
          continue;
        case U'i':
          cps.back() = U'í';
          continue;
        case U'o':
          cps.back() = U'ó';
          continue;
        case U'u':
          cps.back() = U'ú';
          continue;
        default:
          break;
      }
    } else if (cp == 0x0303 && !cps.empty()) {
      if (cps.back() == U'a') {
        cps.back() = U'ã';
        continue;
      }
      if (cps.back() == U'o') {
        cps.back() = U'õ';
        continue;
      }
    } else if (cp == 0x0302 && !cps.empty()) {
      if (cps.back() == U'a') {
        cps.back() = U'â';
        continue;
      }
      if (cps.back() == U'e') {
        cps.back() = U'ê';
        continue;
      }
      if (cps.back() == U'o') {
        cps.back() = U'ô';
        continue;
      }
    } else if (cp == 0x0300 && !cps.empty() && cps.back() == U'a') {
      cps.back() = U'à';
      continue;
    }
    cps.push_back(ToLowerLatin(cp));
  }
  return cps;
}

const absl::flat_hash_map<std::string, std::string>& GetAtoneMap() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"o", "ʊ"},          {"os", "ʊs"},      {"a", "a"},
          {"as", "as"},        {"de", "ʤy"},      {"do", "dʊ"},
          {"da", "da"},        {"dos", "dʊs"},    {"das", "das"},
          {"em", "ˈAŋ"},       {"no", "nʊ"},      {"na", "na"},
          {"nos", "nʊs"},      {"nas", "nas"},    {"que", "ky"},
          {"e", "i"},          {"com", "koŋ"},    {"por", "por"},
          {"se", "sy"},        {"me", "my"},      {"te", "ʧy"},
          {"lhe", "ʎy"},       {"ao", "W"},       {"aos", "Ws"},
          {"à", "ˌaː"},        {"às", "aːs"},     {"um", "ũŋ"},
          {"uns", "ũŋs"},      {"pra", "pra"},    {"pro", "prʊ"},
          {"mas", "mas"},      {"ou", "ow"},      {"nem", "nˈAŋ"},
          {"sem", "sˈAŋ"},     {"num", "nũŋ"},    {"numa", "nˌumæ"},
          {"pelo", "pˈelʊ"},   {"pela", "pˈelæ"}, {"pelos", "pˈelʊs"},
          {"pelas", "pˈelæs"}, {"eu", "eʊ"},      {"é", "ɛ"},
          {"seu", "seʊ"},      {"seus", "seʊs"},  {"meu", "meʊ"},
          {"meus", "meʊs"},    {"tem", "tAŋ"},    {"têm", "tAŋ"},
          {"são", "sɐ̃ʊ̃"},      {"foi", "foɪ"},    {"nós", "nɔs"},
          {"fui", "fuɪ"},      {"lo", "lʊ"},      {"la", "la"},
          {"los", "lʊs"},      {"las", "las"},
      });
  return *kMap;
}

const absl::flat_hash_set<std::string>& GetSecondaryStressSet() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "para",  "como",  "uma",    "umas",   "ele",    "ela",     "eles",
      "elas",  "esse",  "essa",   "esses",  "essas",  "este",    "esta",
      "estes", "estas", "isto",   "sua",    "suas",   "minha",   "minhas",
      "nosso", "nossa", "nossos", "nossas", "foram",  "era",     "eram",
      "estão", "vai",   "vão",    "está",   "estava", "estavam", "estar",
      "estou", "fazer", "poder",  "sempre", "temos",  "somos",   "dele",
      "dela",  "deles", "delas",  "onde",   "você",   "vocês",   "até",
      "não",   "nesta", "neste",  "nessa",  "nesse",  "dessa",   "desse",
      "desta", "deste",
  });
  return *kSet;
}

const absl::flat_hash_map<std::string, std::string>& GetIsolatedLexicon() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"de", "ʤy"},        {"do", "dʊ"},          {"da", "da"},
          {"dos", "dʊs"},      {"das", "das"},        {"no", "nʊ"},
          {"na", "na"},        {"nos", "nʊs"},        {"nas", "nas"},
          {"que", "ky"},       {"com", "koŋ"},        {"por", "por"},
          {"se", "sy"},        {"me", "my"},          {"te", "ʧy"},
          {"lhe", "ʎy"},       {"ao", "W"},           {"aos", "Ws"},
          {"um", "ũŋ"},        {"uns", "ũŋs"},        {"num", "nũŋ"},
          {"mas", "mas"},      {"ou", "ow"},          {"eu", "eʊ"},
          {"seu", "seʊ"},      {"meu", "meʊ"},        {"tem", "tAŋ"},
          {"têm", "tAŋ"},      {"são", "sɐ̃ʊ̃"},        {"foi", "foɪ"},
          {"fui", "fuɪ"},      {"lo", "lʊ"},          {"la", "la"},
          {"os", "ˈʊs"},       {"as", "ˈas"},         {"uma", "ˌumæ"},
          {"umas", "ˌumæs"},   {"como", "kˌomʊ"},     {"para", "pˌaɾæ"},
          {"está", "estˌa"},   {"estava", "estˌavæ"}, {"estavam", "estˌavɐ̃ʊ̃"},
          {"estar", "estˌar"}, {"estou", "estˌow"},   {"estão", "estˌɐ̃ʊ̃"},
          {"até", "atˌɛ"},     {"foram", "fˌoɾɐ̃ʊ̃"},   {"eram", "ˌɛɾɐ̃ʊ̃"},
          {"era", "ˌɛɾæ"},     {"temos", "tˌemʊs"},   {"somos", "sˌomʊs"},
          {"fazer", "fazˌer"}, {"poder", "podˌer"},   {"sempre", "sˌAmpry"},
          {"você", "vosˌe"},   {"vocês", "vosˌes"},   {"onde", "ˌoŋʤy"},
          {"dele", "dˌely"},   {"dela", "dˌɛlæ"},     {"deles", "dˌelys"},
          {"delas", "dˌɛlæs"}, {"tive", "ʧˌivy"},
      });
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>& GetLexicon() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"muito", "mwˈiŋtʊ"},
          {"muita", "mwˈiŋtæ"},
          {"muitos", "mwˈiŋtʊs"},
          {"muitas", "mwˈiŋtæs"},
          {"ela", "ˈɛlæ"},
          {"elas", "ˈɛlæs"},
          {"esta", "ˈɛstæ"},
          {"estas", "ˈɛstæs"},
          {"nesta", "nˈɛstæ"},
          {"nestas", "nˈɛstæs"},
          {"desta", "dˈɛstæ"},
          {"destas", "dˈɛstæs"},
          {"essa", "ˈɛsæ"},
          {"essas", "ˈɛsæs"},
          {"nessa", "nˈɛsæ"},
          {"nessas", "nˈɛsæs"},
          {"dessa", "dˈɛsæ"},
          {"dessas", "dˈɛsæs"},
          {"esse", "ˈesi"},
          {"esses", "ˈesis"},
          {"nesse", "nˈesi"},
          {"nesses", "nˈesis"},
          {"desse", "dˈesi"},
          {"desses", "dˈesis"},
          {"este", "ˈesʧy"},
          {"neste", "nˈesʧi"},
          {"deste", "dˈesʧi"},
          {"sobre", "sˈobri"},
          {"sobremesa", "sobɾemˈezæ"},
          {"põem", "pˈõjm"},
          {"trouxe", "trˈowsy"},
          {"trouxeram", "trˌowsˈɛɾɐ̃ʊ̃"},
          {"próximo", "prˈɔsimʊ"},
          {"próxima", "prˈɔsimæ"},
          {"próximos", "prˈɔsimʊs"},
          {"próximas", "prˈɔsimæs"},
          {"máximo", "mˈasimʊ"},
          {"máxima", "mˈasimæ"},
          {"auxílio", "ˌWsˈiljʊ"},
          {"táxi", "tˈaksi"},
          {"sexo", "sˈɛksʊ"},
          {"anexo", "ˌænˈɛksʊ"},
          {"complexo", "kˌomplˈɛksʊ"},
          {"reflexo", "xˌeflˈɛksʊ"},
          {"fixo", "fˈiksʊ"},
          {"fluxo", "flˈuksʊ"},
          {"ex", "ˈes"},
          {"internet", "ˌiŋteɾənˈɛʧ"},
          {"show", "ʃˈow"},
          {"shows", "ʃˈows"},
          {"online", "oŋlˈIŋ"},
          {"blog", "blˈɔɡ"},
          {"site", "sˈIʧy"},
          {"sites", "sˈIʧys"},
          {"email", "imˈAʊ"},
          {"futebol", "fˌutebˈɔl"},
          {"açúcar", "ˌasˈukar"},
          {"último", "ˈulʧimʊ"},
          {"última", "ˈulʧimæ"},
          {"últimos", "ˈulʧimʊs"},
          {"últimas", "ˈulʧimæs"},
          {"evoluiu", "ˌevolwˈiʊ"},
          {"reunião", "xˌeuniˈɐ̃ʊ̃"},
          {"onu", "ˈonu"},
          {"pib", "pˌeˌibˈe"},
          {"tv", "tˌevˈe"},
          {"in", "ˈin"},
          {"the", "tˈy"},
          {"qualquer", "kwWkˌɛr"},
          {"quer", "kˈɛr"},
          {"quero", "kˈɛɾʊ"},
          {"querem", "kˈɛɾAŋ"},
          {"quiser", "kizˈɛr"},
          {"tiver", "ʧivˈɛr"},
          {"puder", "pudˈɛr"},
          {"fizer", "fizˈɛr"},
          {"fizeram", "fˌizˈɛɾɐ̃ʊ̃"},
          {"tiveram", "ʧˌivˈɛɾɐ̃ʊ̃"},
          {"puderam", "pˌudˈɛɾɐ̃ʊ̃"},
          {"vieram", "vˌiˈɛɾɐ̃ʊ̃"},
          {"estiver", "ˌesʧivˈɛr"},
          {"houver", "owvˈɛr"},
          {"mulher", "muljˈɛr"},
          {"mulheres", "mˌuljˈɛɾys"},
          {"colher", "koljˈɛr"},
          {"melhor", "meljˈɔr"},
          {"melhores", "mˌeljˈɔɾys"},
          {"maior", "mIˈɔr"},
          {"maiores", "mˌIˈɔɾys"},
          {"menor", "menˈɔr"},
          {"menores", "mˌenˈɔɾys"},
          {"pior", "piˈɔr"},
          {"piores", "pˌiˈɔɾys"},
          {"redor", "xedˈɔr"},
          {"suor", "suˈɔr"},
          {"dez", "dˈɛs"},
          {"voz", "vˈɔs"},
          {"vozes", "vˈɔzys"},
          {"veloz", "velˈɔs"},
          {"velozes", "vˌelˈɔzys"},
          {"nós", "nˈɔs"},
          {"avó", "avˈɔ"},
          {"avós", "avˈɔs"},
          {"avô", "avˈo"},
          {"sede", "sˈeʤy"},
          {"podemos", "podˈemʊs"},
          {"período", "pˌeɾˈiʊdʊ"},
          {"ideia", "ˌidˈɛɪæ"},
          {"ideias", "ˌidˈɛɪæs"},
          {"assembleia", "ˌasAmblˈɛɪæ"},
          {"europeia", "ˌeʊɾopˈɛɪæ"},
          {"estreia", "ˌestrˈɛɪæ"},
          {"plateia", "plˌatˈɛɪæ"},
          {"feia", "fˈɛɪæ"},
          {"meia", "mˈAæ"},
      });
  return *kMap;
}

const absl::flat_hash_set<std::string>& GetOpenEWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "belo",       "bela",      "belos",     "belas",     "velho",
      "velha",      "velhos",    "velhas",    "congresso", "congressos",
      "progresso",  "sucesso",   "sucessos",  "acesso",    "acessos",
      "processo",   "processos", "excesso",   "projeto",   "projetos",
      "objeto",     "objetos",   "direto",    "direta",    "diretos",
      "diretas",    "correto",   "correta",   "completo",  "completa",
      "concreto",   "concreta",  "secreto",   "secreta",   "afeto",
      "teto",       "neto",      "neta",      "quieto",    "quieta",
      "inseto",     "festa",     "festas",    "floresta",  "florestas",
      "resta",      "testa",     "manifesta", "honesta",   "modesta",
      "guerra",     "guerras",   "terra",     "terras",    "serra",
      "ferra",      "erra",      "encerra",   "merda",     "perda",
      "perdas",     "certa",     "certas",    "certo",     "certos",
      "aberta",     "abertas",   "aberto",    "abertos",   "coberta",
      "descoberta", "esperta",   "alerta",    "oferta",    "ofertas",
      "deserto",    "conserto",  "perto",     "inverno",   "caderno",
      "cadernos",   "moderno",   "moderna",   "modernos",  "modernas",
      "eterno",     "eterna",    "governa",   "governam",  "conversa",
      "conversas",  "diversa",   "diversas",  "diversos",  "inversa",
      "reversa",    "versa",     "universo",  "verso",     "versos",
      "prefere",    "preferem",  "refere",    "referem",   "confere",
      "sugere",     "interfere", "adere",     "repete",    "repetem",
      "compete",    "segue",     "seguem",    "consegue",  "conseguem",
      "persegue",   "serve",     "servem",    "bebe",      "bebem",
      "recebe",     "recebem",   "percebe",   "percebem",  "deve",
      "devem",      "escreve",   "escrevem",  "descreve",  "inscreve",
      "atrevem",    "breve",     "breves",    "leve",      "leves",
      "neve",       "greve",     "teve",      "esteve",    "manteve",
      "obteve",     "deteve",    "acontece",  "acontecem", "aparece",
      "aparecem",   "parece",    "parecem",   "oferece",   "oferecem",
      "conhece",    "conhecem",  "reconhece", "esquece",   "esquecem",
      "merece",     "merecem",   "cresce",    "crescem",   "nasce",
      "nascem",     "desce",     "descem",    "permanece", "estabelece",
      "fornece",    "enriquece", "aquece",    "adoece",    "apodrece",
      "amolece",    "anoitece",  "amanhece",  "envelhece", "prece",
      "começa",     "começam",   "peça",      "peças",     "cabeça",
      "cabeças",    "depressa",  "promessa",  "promessas", "interessa",
      "interessam", "confessa",  "atravessa", "pressa",    "expressa",
      "impressa",   "espera",    "esperam",   "era",       "eram",
      "fera",       "esfera",    "atmosfera", "primavera", "severa",
      "sincera",    "mera",      "gera",      "geram",     "opera",
      "operam",     "supera",    "considera", "recupera",  "tolera",
      "acelera",    "zero",      "trimestre", "semestre",  "bimestre",
      "mestre",     "mestres",   "pedestre",  "terrestre", "silvestre",
      "papel",      "hotel",     "anel",      "cruel",     "fiel",
      "mel",        "pele",      "febre",     "lebre",     "alegre",
      "alegres",    "regra",     "regras",    "pedra",     "pedras",
      "quebra",     "cega",      "nega",      "pega",      "chega",
      "entrega",    "emprega",   "carrega",   "colega",    "colegas",
      "sete",       "dez",       "meta",      "metas",     "poeta",
      "poetas",     "profeta",   "planeta",   "atleta",    "dieta",
      "seta",       "reta",      "afeta",     "coleta",    "discreta",
      "inquieta",   "metro",     "metros",    "cetro",     "retro",
      "janela",     "janelas",   "panela",    "panelas",   "canela",
      "capela",     "cautela",   "cela",      "celas",     "costela",
      "donzela",    "novela",    "novelas",   "parcela",   "parcelas",
      "paralela",   "passarela", "sentinela", "tabela",    "tabelas",
      "tela",       "telas",     "tutela",    "vela",      "velas",
      "favela",     "favelas",   "aquarela",  "amarela",   "amarelas",
      "amarelo",    "amarelos",  "castelo",   "castelos",  "chinelo",
      "cogumelo",   "duelo",     "elo",       "farelo",    "flagelo",
      "martelo",    "paralelo",  "pesadelo",  "singelo",   "tornozelo",
      "zelo",       "reserva",   "reservas",  "preserva",  "observa",
      "conserva",   "erva",      "ervas",     "nervo",     "nervos",
      "servo",      "servos",    "cervo",     "acervo",    "verbo",
      "verbos",     "soberba",   "cerca",     "perca",     "merca",
      "terça",      "interna",   "internas",  "externa",   "externas",
      "eternas",    "perna",     "pernas",    "caverna",   "taberna",
      "lanterna",   "cisterna",  "paterna",   "materna",   "fraterna",
  });
  return *kSet;
}

const absl::flat_hash_set<std::string>& GetOpenOWords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kSet({
      "nova",        "novas",       "novos",       "povos",      "ovos",
      "jogos",       "fogos",       "corpos",      "olhos",      "portos",
      "mortos",      "postos",      "rostos",      "grossos",    "ossos",
      "poços",       "blocos",      "focos",       "locos",      "socos",
      "trocos",      "votos",       "modos",       "bolos",      "polos",
      "solos",       "colos",       "tolos",       "coros",      "foros",
      "moros",       "poros",       "impostos",    "esforços",   "acordos",
      "copo",        "copos",       "bloco",       "foco",       "choque",
      "toque",       "estoque",     "reboque",     "fofoca",     "mandioca",
      "pipoca",      "minhoca",     "carioca",     "troca",      "trocas",
      "toca",        "tocam",       "boca",        "bocas",      "foca",
      "roca",        "broca",       "coloca",      "colocam",    "provoca",
      "provocam",    "convoca",     "sufoca",      "evoca",      "equivoca",
      "desloca",     "escola",      "escolas",     "bola",       "bolas",
      "cola",        "gola",        "mola",        "sola",       "viola",
      "gaiola",      "sacola",      "pistola",     "esmola",     "cebola",
      "argola",      "enrola",      "controla",    "isola",      "decola",
      "consola",     "solo",        "colo",        "polo",       "protocolo",
      "porta",       "portas",      "morta",       "mortas",     "morto",
      "corta",       "cortam",      "importa",     "suporta",    "transporta",
      "exporta",     "comporta",    "conforta",    "torta",      "tortas",
      "horta",       "forte",       "fortes",      "morte",      "mortes",
      "sorte",       "norte",       "porte",       "esporte",    "esportes",
      "transporte",  "passaporte",  "suporte",     "recorte",    "consorte",
      "forma",       "formas",      "reforma",     "plataforma", "uniforme",
      "enorme",      "enormes",     "conforme",    "informe",    "dorme",
      "dormem",      "norma",       "normas",      "torna",      "tornam",
      "retorna",     "contorna",    "corno",       "corda",      "cordas",
      "acorda",      "acordam",     "concorda",    "concordam",  "discorda",
      "recorda",     "borda",       "bordas",      "aborda",     "engorda",
      "horda",       "ordem",       "ordens",      "desordem",   "morde",
      "lorde",       "força",       "forças",      "esforça",    "reforça",
      "corpo",       "porco",       "porcos",      "porca",      "orçamento",
      "gosta",       "gostam",      "costa",       "costas",     "posta",
      "postas",      "aposta",      "apostas",     "resposta",   "respostas",
      "proposta",    "propostas",   "composta",    "exposta",    "disposta",
      "oposta",      "suposta",     "encosta",     "lagosta",    "amostra",
      "mostra",      "mostram",     "vossa",       "vossas",     "vosso",
      "vossos",      "nossa",       "nossas",      "nosso",      "nossos",
      "fossa",       "possa",       "possam",      "grossa",     "grossas",
      "roça",        "carroça",     "destroços",   "troço",      "hora",
      "horas",       "agora",       "fora",        "embora",     "demora",
      "melhora",     "piora",       "chora",       "choram",     "mora",
      "moram",       "namora",      "adora",       "adoram",     "ignora",
      "explora",     "implora",     "colabora",    "comemora",   "elabora",
      "devora",      "decora",      "apavora",     "aurora",     "flora",
      "espora",      "senhora",     "senhoras",    "obra",       "obras",
      "cobra",       "cobras",      "sobra",       "sobras",     "dobra",
      "manobra",     "nobre",       "nobres",      "pobre",      "pobres",
      "cobre",       "descobre",    "prova",       "provas",     "aprova",
      "comprova",    "reprova",     "cova",        "escova",     "renova",
      "inovam",      "chove",       "move",        "movem",      "promove",
      "remove",      "comove",      "nove",        "jovem",      "jovens",
      "joga",        "jogam",       "droga",       "drogas",     "roga",
      "rogam",       "interroga",   "prorroga",    "afoga",      "advoga",
      "foga",        "sogra",       "logra",       "nota",       "notas",
      "bota",        "botas",       "gota",        "gotas",      "cota",
      "cotas",       "frota",       "rota",        "rotas",      "derrota",
      "derrotas",    "idiota",      "patriota",    "remota",     "devota",
      "agota",       "esgota",      "anota",       "vota",       "votam",
      "voto",        "devoto",      "foto",        "fotos",      "moto",
      "motos",       "terremoto",   "remoto",      "piloto",     "garoto",
      "garota",      "pacote",      "decote",      "chicote",    "bigode",
      "pagode",      "sacode",      "pode",        "podem",      "bode",
      "ode",         "moda",        "modas",       "roda",       "rodas",
      "foda",        "incomoda",    "acomoda",     "poda",       "modo",
      "rosa",        "rosas",       "prosa",       "esposa",     "gasosa",
      "famosa",      "famosas",     "formosa",     "poderosa",   "perigosa",
      "maravilhosa", "orgulhosa",   "curiosa",     "nervosa",    "preciosa",
      "religiosa",   "misteriosa",  "deliciosa",   "silenciosa", "valiosa",
      "vitoriosa",   "ambiciosa",   "graciosa",    "furiosa",    "invejosa",
      "cuidadosa",   "amorosa",     "saborosa",    "cheirosa",   "gostosa",
      "gostosas",    "teimosa",     "vaidosa",     "mentirosa",  "generosa",
      "numerosa",    "dolorosa",    "gloriosa",    "espaçosa",   "dose",
      "doses",       "pose",        "neurose",     "psicose",    "cirrose",
      "trombose",    "metamorfose", "virose",      "celulose",   "glicose",
      "hipnose",     "tuberculose", "osteoporose", "fibrose",    "esclerose",
      "copa",        "copas",       "tropa",       "tropas",     "sopa",
      "topa",        "popa",        "europa",      "galopa",     "loja",
      "lojas",       "soja",        "boia",        "joia",       "joias",
      "apoia",       "apoiam",      "heroico",     "paranoico",  "troia",
      "jiboia",      "olho",        "molho",       "piolho",     "repolho",
      "ferrolho",
  });
  return *kSet;
}

const absl::flat_hash_map<std::string, std::string>& GetLetterPhonemes() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"a", "ˈa"},      {"b", "bˈe"},   {"c", "sˈe"},     {"d", "dˈe"},
          {"e", "ˈɛ"},      {"f", "ˈɛfy"},  {"g", "ʒˈe"},     {"h", "aɡˈa"},
          {"i", "ˈi"},      {"j", "ʒˈɔtæ"}, {"k", "kˈa"},     {"l", "ˈɛly"},
          {"m", "ˈeme"},    {"n", "ˈeny"},  {"o", "ˈɔ"},      {"p", "pˈe"},
          {"q", "kˈe"},     {"r", "ˈɛxe"},  {"s", "ˈɛsy"},    {"t", "tˈe"},
          {"u", "ˈu"},      {"v", "vˈe"},   {"w", "dˈabliʊ"}, {"x", "ʃˈis"},
          {"y", "ˈipsilõ"}, {"z", "zˈe"},
      });
  return *kMap;
}

const absl::flat_hash_map<std::string, std::string>& GetAbbrPhonemes() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kMap({
          {"sr", "ˌɛsyˈɛxe"},
          {"dr", "dowtˈor"},
          {"kg", "kˌaʒˈe"},
          {"km", "kˌaˈeme"},
      });
  return *kMap;
}

enum class UnitKind {
  kVowel,
  kConsonant,
  kGlide,
  kLateralCoda,
  kNasalCoda,
};

struct Unit {
  UnitKind kind = UnitKind::kConsonant;
  char32_t vowel_cp = 0;
  std::string ph;
  char32_t next_after_nasal = 0;
};

char32_t AtCp(const std::vector<char32_t>& w, size_t k) {
  return k < w.size() ? w[k] : 0;
}

void SplitAndPushWords(absl::string_view phrase,
                       std::vector<std::string>& out_words) {
  size_t start = 0;
  while (start < phrase.size()) {
    const size_t sp = phrase.find(' ', start);
    if (sp == absl::string_view::npos) {
      out_words.push_back(std::string(phrase.substr(start)));
      break;
    }
    out_words.push_back(std::string(phrase.substr(start, sp - start)));
    start = sp + 1;
  }
}

std::vector<Unit> BuildUnits(const std::vector<char32_t>& w) {
  std::vector<Unit> out;
  const size_t n = w.size();
  size_t i = 0;

  while (i < n) {
    const char32_t c = w[i];
    const char32_t nx = AtCp(w, i + 1);
    if (IsPortugueseVowel(c)) {
      out.push_back({UnitKind::kVowel, c, "", 0});
      i += 1;
      continue;
    }
    if (c == U'h') {
      i += 1;
      continue;
    }
    if (c == U'c') {
      if (nx == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "ʃ", 0});
        i += 2;
        continue;
      }
      if (nx != 0 && IsFrontVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "s", 0});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "k", 0});
      i += 1;
      continue;
    }
    if (c == U'ç') {
      out.push_back({UnitKind::kConsonant, 0, "s", 0});
      i += 1;
      continue;
    }
    if (c == U'q') {
      const char32_t nx2 = AtCp(w, i + 2);
      if (nx == U'u' && (nx2 == U'a' || nx2 == U'o' || nx2 == U'á' ||
                         nx2 == U'ã' || nx2 == U'ó')) {
        out.push_back({UnitKind::kConsonant, 0, "k", 0});
        out.push_back({UnitKind::kGlide, 0, "w", 0});
        i += 2;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "k", 0});
      i += (nx == U'u') ? 2 : 1;
      continue;
    }
    if (c == U'g') {
      const char32_t nx2 = AtCp(w, i + 2);
      if (nx == U'u' && nx2 != 0 && IsFrontVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "ɡ", 0});
        i += 2;
        continue;
      }
      if (nx == U'u' &&
          (nx2 == U'a' || nx2 == U'o' || nx2 == U'á' || nx2 == U'ã')) {
        out.push_back({UnitKind::kConsonant, 0, "ɡ", 0});
        out.push_back({UnitKind::kGlide, 0, "w", 0});
        i += 2;
        continue;
      }
      if (nx != 0 && IsFrontVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "ʒ", 0});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "ɡ", 0});
      i += 1;
      continue;
    }
    if (c == U'l') {
      if (nx == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "l", 0});
        out.push_back({UnitKind::kGlide, 0, "j", 0});
        i += 2;
        continue;
      }
      if (nx == U'l') {
        i += 1;
        continue;
      }
      if (nx == 0 || !IsPortugueseVowel(nx)) {
        out.push_back({UnitKind::kLateralCoda, 0, "l", 0});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "l", 0});
      i += 1;
      continue;
    }
    if (c == U'n' && nx == U'h') {
      out.push_back({UnitKind::kConsonant, 0, "ɲ", 0});
      i += 2;
      continue;
    }
    if (c == U'm' || c == U'n') {
      if (nx == 0 || !IsPortugueseVowel(nx)) {
        const bool word_final_or_s =
            (i + 1 == n) || (i + 2 == n && w[i + 1] == U's');
        std::string s(1, static_cast<char>(c));
        out.push_back({UnitKind::kNasalCoda, 0, s, word_final_or_s ? 0 : nx});
        i += 1;
        continue;
      }
      if (nx == c) {
        i += 1;
        continue;
      }
      std::string s(1, static_cast<char>(c));
      out.push_back({UnitKind::kConsonant, 0, s, 0});
      i += 1;
      continue;
    }
    if (c == U'r') {
      const Unit* prev = out.empty() ? nullptr : &out.back();
      if (nx == U'r') {
        out.push_back({UnitKind::kConsonant, 0, "x", 0});
        i += 2;
        continue;
      }
      if (i == 0 || (prev != nullptr && (prev->kind == UnitKind::kNasalCoda ||
                                         prev->kind == UnitKind::kLateralCoda ||
                                         prev->ph == "s" || prev->ph == "z"))) {
        out.push_back({UnitKind::kConsonant, 0, "x", 0});
        i += 1;
        continue;
      }
      if (prev != nullptr && prev->kind == UnitKind::kConsonant &&
          (prev->ph == "p" || prev->ph == "b" || prev->ph == "t" ||
           prev->ph == "d" || prev->ph == "k" || prev->ph == "ɡ" ||
           prev->ph == "f" || prev->ph == "v")) {
        out.push_back({UnitKind::kConsonant, 0, "r", 0});
        i += 1;
        continue;
      }
      if (nx == 0) {
        out.push_back({UnitKind::kConsonant, 0, "r", 0});
        i += 1;
        continue;
      }
      if (!IsPortugueseVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "ɾə", 0});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "ɾ", 0});
      i += 1;
      continue;
    }
    if (c == U's') {
      const bool prev_v = !out.empty() && out.back().kind == UnitKind::kVowel;
      if (nx == U's') {
        out.push_back({UnitKind::kConsonant, 0, "s", 0});
        i += 2;
        continue;
      }
      const char32_t nx2 = AtCp(w, i + 2);
      if (nx == U'c' && nx2 != 0 && IsFrontVowel(nx2)) {
        out.push_back({UnitKind::kConsonant, 0, "s", 0});
        i += 2;
        continue;
      }
      if (prev_v && nx != 0 && IsPortugueseVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "z", 0});
        i += 1;
        continue;
      }
      if (nx == U'b' || nx == U'd' || nx == U'g' || nx == U'v' || nx == U'm' ||
          nx == U'n' || nx == U'l' || nx == U'r') {
        out.push_back({UnitKind::kConsonant, 0, "z", 0});
        i += 1;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "s", 0});
      i += 1;
      continue;
    }
    if (c == U'z') {
      out.push_back({UnitKind::kConsonant, 0, nx == 0 ? "s" : "z", 0});
      i += 1;
      continue;
    }
    if (c == U'x') {
      if (i == 1 && w[0] == U'e' && nx != 0 && IsPortugueseVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "z", 0});
      } else if (nx != 0 && !IsPortugueseVowel(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "s", 0});
      } else {
        out.push_back({UnitKind::kConsonant, 0, "ʃ", 0});
      }
      i += 1;
      continue;
    }
    if (c == U'j') {
      out.push_back({UnitKind::kConsonant, 0, "ʒ", 0});
      i += 1;
      continue;
    }
    if (c == U'y') {
      out.push_back({UnitKind::kVowel, U'i', "", 0});
      i += 1;
      continue;
    }
    if (c == U'w') {
      out.push_back({UnitKind::kGlide, 0, "w", 0});
      i += 1;
      continue;
    }
    if (c == U'k') {
      out.push_back({UnitKind::kConsonant, 0, "k", 0});
      i += 1;
      continue;
    }
    if (nx == c) {
      i += 1;
    }
    std::string s;
    AppendUtf8(c, s);
    out.push_back({UnitKind::kConsonant, 0, s, 0});
    i += 1;
  }
  return out;
}

bool IsFallingDiphthong(char32_t a, char32_t b) {
  return (a == U'a' && (b == U'i' || b == U'u')) ||
         (a == U'e' && (b == U'i' || b == U'u')) ||
         (a == U'o' && (b == U'i' || b == U'u')) || (a == U'u' && b == U'i') ||
         (a == U'i' && b == U'u') || (a == U'ã' && (b == U'o' || b == U'e')) ||
         (a == U'õ' && b == U'e') || (a == U'ɛ' && b == U'u') ||
         (a == U'ɔ' && b == U'i') || (a == U'é' && (b == U'u' || b == U'i')) ||
         (a == U'ó' && b == U'i');
}

struct Segment {
  bool is_nucleus = false;
  Unit unit;
  std::vector<char32_t> vowels;
};

std::vector<Segment> BuildNuclei(const std::vector<Unit>& units) {
  std::vector<Segment> segs;
  size_t i = 0;
  while (i < units.size()) {
    if (units[i].kind != UnitKind::kVowel) {
      segs.push_back({false, units[i], {}});
      i += 1;
      continue;
    }
    std::vector<char32_t> run;
    while (i < units.size() && units[i].kind == UnitKind::kVowel) {
      run.push_back(units[i].vowel_cp);
      i += 1;
    }
    size_t j = 0;
    while (j < run.size()) {
      const char32_t ch = run[j];
      const char32_t nx = (j + 1 < run.size()) ? run[j + 1] : 0;
      if (nx != 0 && IsFallingDiphthong(ch, nx)) {
        if (j + 2 == run.size()) {
          const Unit* a0 = (i < units.size()) ? &units[i] : nullptr;
          const Unit* a1 = (i + 1 < units.size()) ? &units[i + 1] : nullptr;
          const bool closing =
              a0 != nullptr &&
              ((a0->kind == UnitKind::kConsonant && a0->ph == "ɲ") ||
               a0->kind == UnitKind::kNasalCoda ||
               a0->kind == UnitKind::kLateralCoda ||
               (a0->kind == UnitKind::kConsonant &&
                (a0->ph == "r" || a0->ph == "ɾə" || a0->ph == "z") &&
                (a1 == nullptr || a1->kind != UnitKind::kVowel)));
          if ((nx == U'i' || nx == U'u') && ch != U'i' && ch != U'u' &&
              closing) {
            segs.push_back({true, {}, {ch}});
            segs.push_back({true, {}, {nx}});
            j += 2;
            continue;
          }
        }
        segs.push_back({true, {}, {ch, nx}});
        j += 2;
        continue;
      }
      segs.push_back({true, {}, {ch}});
      j += 1;
    }
  }
  return segs;
}

bool EndsWith(const std::vector<char32_t>& w, absl::string_view suffix) {
  const std::vector<char32_t> suf = ToCodepointsLower(suffix);
  if (w.size() < suf.size()) return false;
  for (size_t i = 0; i < suf.size(); ++i) {
    if (w[w.size() - suf.size() + i] != suf[i]) return false;
  }
  return true;
}

int ComputeStressIndex(const std::vector<Segment>& segs,
                       const std::vector<size_t>& nuc_idx,
                       const std::vector<char32_t>& w) {
  if (nuc_idx.empty()) return -1;
  for (size_t k = 0; k < nuc_idx.size(); ++k) {
    for (char32_t c : segs[nuc_idx[k]].vowels) {
      if (c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú' ||
          c == U'â' || c == U'ê' || c == U'ô') {
        return static_cast<int>(k);
      }
    }
  }
  for (size_t k = 0; k < nuc_idx.size(); ++k) {
    for (char32_t c : segs[nuc_idx[k]].vowels) {
      if (c == U'ã' || c == U'õ') {
        return static_cast<int>(k);
      }
    }
  }
  if (nuc_idx.size() == 1) return 0;
  std::vector<char32_t> core = w;
  if (!core.empty() && core.back() == U's') {
    core.pop_back();
  }
  const bool paroxytone_core = EndsWith(core, "a") || EndsWith(core, "e") ||
                               EndsWith(core, "o") || EndsWith(core, "am") ||
                               EndsWith(core, "em") || EndsWith(core, "en");
  if ((paroxytone_core && !EndsWith(w, "ns")) || EndsWith(w, "ens")) {
    return static_cast<int>(nuc_idx.size()) - 2;
  }
  return static_cast<int>(nuc_idx.size()) - 1;
}

struct PhonToken {
  UnitKind kind = UnitKind::kConsonant;
  std::string body;
  bool stressed = false;
  int syl_idx = -1;
  bool is_glide = false;
};

std::string DiphthongBody(char32_t a, char32_t b) {
  if (a == U'a' && b == U'i') return "I";
  if (a == U'a' && b == U'u') return "W";
  if (a == U'e' && b == U'i') return "A";
  if (a == U'e' && b == U'u') return "eʊ";
  if (a == U'ɛ' && b == U'u') return "ɛʊ";
  if (a == U'o' && b == U'i') return "oɪ";
  if (a == U'ɔ' && b == U'i') return "ɔɪ";
  if (a == U'o' && b == U'u') return "ow";
  if (a == U'u' && b == U'i') return "uɪ";
  if (a == U'i' && b == U'u') return "iʊ";
  if (a == U'ã' && b == U'o') return "ɐ̃ʊ̃";
  if (a == U'ã' && b == U'e') return "ɐ̃y";
  if (a == U'õ' && b == U'e') return "õj";
  if (a == U'ɛ' && b == U'i') return "ɛɪ";
  std::string out;
  AppendUtf8(a, out);
  AppendUtf8(b, out);
  return out;
}

std::string PhonemizePortugueseWordByRules(const std::vector<char32_t>& w_cps,
                                           absl::string_view w_str,
                                           bool use_secondary_stress) {
  const std::vector<Unit> units = BuildUnits(w_cps);
  const std::vector<Segment> segs = BuildNuclei(units);
  std::vector<size_t> nuc_idx;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].is_nucleus) nuc_idx.push_back(i);
  }
  const int st = ComputeStressIndex(segs, nuc_idx, w_cps);
  const int nsyl = static_cast<int>(nuc_idx.size());

  std::vector<PhonToken> toks;
  int cur_syl = 0;
  for (size_t si = 0; si < segs.size(); ++si) {
    const Segment& s = segs[si];
    if (!s.is_nucleus) {
      toks.push_back({s.unit.kind, s.unit.ph, false, -1, false});
      continue;
    }
    const int k = cur_syl++;
    const bool stressed = (k == st);
    const std::vector<char32_t>& chars = s.vowels;
    const char32_t v = chars[0];
    const char32_t vb = BaseVowel(v);
    const Segment* nxt = (si + 1 < segs.size()) ? &segs[si + 1] : nullptr;
    const Segment* nxt2 = (si + 2 < segs.size()) ? &segs[si + 2] : nullptr;
    bool rest_is_s = (si + 1 < segs.size());
    for (size_t r = si + 1; r < segs.size(); ++r) {
      if (segs[r].is_nucleus || segs[r].unit.kind != UnitKind::kConsonant ||
          (segs[r].unit.ph != "s" && segs[r].unit.ph != "z")) {
        rest_is_s = false;
        break;
      }
    }
    const bool word_final_v = (si + 1 == segs.size()) || rest_is_s;
    if (chars.size() == 2) {
      const std::string body =
          DiphthongBody(BaseVowel(chars[0]), BaseVowel(chars[1]));
      toks.push_back({UnitKind::kVowel, body, stressed, k, false});
      continue;
    }

    std::string body;
    if (vb == U'ã') {
      body = "ɐ̃";
    } else if (vb == U'õ') {
      body = "õ";
    } else {
      AppendUtf8(vb, body);
    }

    const bool nasal_coda = (nxt != nullptr && !nxt->is_nucleus &&
                             nxt->unit.kind == UnitKind::kNasalCoda);
    if (nasal_coda) {
      const char32_t after = nxt->unit.next_after_nasal;
      const bool labial = (after == U'p' || after == U'b');
      if (after == 0 || after == U's') {
        if (body == "a") {
          body = "ɐ̃ʊ̃";
        } else if (body == "e" || body == "ɛ") {
          body = "Aŋ";
        } else if (body == "i") {
          body = "iŋ";
        } else if (body == "o" || body == "ɔ") {
          body = "oŋ";
        } else if (body == "u") {
          body = "ũŋ";
        } else {
          body += "ŋ";
        }
      } else if (labial) {
        if (body == "a") {
          body = "ɐ̃m";
        } else if (body == "e" || body == "ɛ") {
          body = "Am";
        } else if (body == "i") {
          body = "im";
        } else if (body == "o") {
          body = "om";
        } else if (body == "u") {
          body = "um";
        } else {
          body += 'm';
        }
      } else {
        if (body == "a") {
          body = "ɐ̃ŋ";
        } else if (body == "e" || body == "ɛ") {
          body = "Aŋ";
        } else if (body == "i") {
          body = "iŋ";
        } else if (body == "o" || body == "ɔ") {
          body = "oŋ";
        } else if (body == "u") {
          body = "ũŋ";
        } else {
          body += "ŋ";
        }
      }
    } else if (stressed && (body == "a" || body == "u") && nxt != nullptr &&
               !nxt->is_nucleus && nxt->unit.kind == UnitKind::kConsonant &&
               (nxt->unit.ph == "m" || nxt->unit.ph == "n" ||
                nxt->unit.ph == "ɲ")) {
      if (body == "a") {
        body = "ɐ̃";
      } else if (body == "u" && (v == U'ú' || nxt->unit.ph == "ɲ")) {
        body = "ũ";
      }
    } else if ((body == "a" || body == "u") && nxt != nullptr &&
               !nxt->is_nucleus && nxt->unit.kind == UnitKind::kConsonant &&
               nxt->unit.ph == "ɲ") {
      body = (body == "a") ? "ɐ̃" : "ũ";
    } else if (stressed && !IsAccStressVowel(v) &&
               (body == "e" || body == "o")) {
      if (body == "e" && GetOpenEWords().contains(w_str)) {
        body = "ɛ";
      } else if (body == "o" && GetOpenOWords().contains(w_str)) {
        body = "ɔ";
      }
    } else if (!stressed) {
      if (word_final_v && st >= 0 && k > st) {
        if (body == "a") {
          body = "æ";
        } else if (body == "e" || body == "ɛ") {
          body = "y";
        } else if (body == "o" || body == "ɔ") {
          body = "ʊ";
        }
      } else if (st >= 0 && k > st && body == "a") {
        body = "æ";
      } else if (st >= 0 && k < st && body == "a" && nxt != nullptr &&
                 !nxt->is_nucleus && nxt->unit.kind == UnitKind::kConsonant &&
                 (nxt->unit.ph == "m" || nxt->unit.ph == "n") &&
                 nxt2 != nullptr && nxt2->is_nucleus) {
        body = "æ";
      } else if ((body == "ɛ" || body == "ɔ") && !IsAccStressVowel(v)) {
        body = (body == "ɛ") ? "e" : "o";
      }
    }
    toks.push_back({UnitKind::kVowel, body, stressed, k, false});
  }

  // Coda /l/ vocalization and drop absorbed nasal codas.
  std::vector<PhonToken> out;
  out.reserve(toks.size());
  for (const PhonToken& t : toks) {
    if (t.kind == UnitKind::kLateralCoda) {
      if (!out.empty() && out.back().kind == UnitKind::kVowel) {
        std::string& b = out.back().body;
        if (b == "a") {
          b = "W";
        } else if (b == "u") {
          b = "uw";
        } else if (b == "o" || b == "ɔ") {
          b += 'w';
        } else {
          b += "ʊ";
        }
      }
      continue;
    }
    if (t.kind == UnitKind::kNasalCoda) {
      continue;
    }
    out.push_back(t);
  }

  // Palatalization of /t, d/ before [i]/[y].
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].kind == UnitKind::kConsonant &&
        (out[i].body == "t" || out[i].body == "d") && i + 1 < out.size() &&
        out[i + 1].kind == UnitKind::kVowel) {
      const absl::string_view nx_body = out[i + 1].body;
      if (!nx_body.empty() && (nx_body[0] == 'i' || nx_body[0] == 'y')) {
        out[i].body = (out[i].body == "t") ? "ʧ" : "ʤ";
      }
    }
  }

  // Rising hiatus i/u + vowel after stress -> glide.
  for (size_t i = 0; i + 1 < out.size(); ++i) {
    if (out[i].kind == UnitKind::kVowel &&
        out[i + 1].kind == UnitKind::kVowel &&
        (out[i].body == "i" || out[i].body == "u") && !out[i].stressed &&
        st >= 0 && out[i].syl_idx > st) {
      out[i].body = (out[i].body == "i") ? "j" : "w";
      out[i].is_glide = true;
    }
  }

  // Render stress marks.
  absl::flat_hash_map<int, std::string> marks;
  if (!use_secondary_stress) {
    if (st >= 0 && nsyl >= 3) {
      for (int k = 0; k < st; ++k) {
        if (k == 0 || (k % 2 == 0 && k < st - 1)) {
          marks[k] = "ˌ";
        }
      }
    }
    if (st >= 0) {
      marks[st] = "ˈ";
    }
  } else if (st >= 0) {
    marks[st] = "ˌ";
  }

  std::string result;
  for (const PhonToken& t : out) {
    if (t.kind == UnitKind::kConsonant || t.kind == UnitKind::kGlide) {
      result.append(t.body);
    } else {
      if (!t.is_glide) {
        if (const auto it = marks.find(t.syl_idx); it != marks.end()) {
          result.append(it->second);
        }
      }
      result.append(t.body);
    }
  }
  return result;
}

std::string PtUnder100(int n) {
  static constexpr absl::string_view kUnits[20] = {
      "zˈɛɾʊ",  "ˈũŋ",       "dˈoɪs",      "trˈes",     "kwˈatrʊ",
      "sˈiŋkʊ", "sˈAs",      "sˈɛʧy",      "ˈoɪtʊ",     "nˈɔvy",
      "dˈɛs",   "ˈoŋzy",     "dˈozy",      "trˈezy",    "kˌatˈoɾəzy",
      "kˈiŋzy", "dˌezesˈAs", "dˌezesˈɛʧy", "dˌezˈoɪtʊ", "dˌezenˈɔvy",
  };
  static constexpr absl::string_view kTens[10] = {
      "",          "",           "vˈiŋʧy",
      "trˈiŋtæ",   "kwˌaɾˈAŋtæ", "sˌiŋkwˈAŋtæ",
      "sˌesˈAŋtæ", "sˌetˈAŋtæ",  "ˌoɪtˈAŋtæ",
      "nˌovˈAŋtæ",
  };
  if (n < 20) return std::string(kUnits[n]);
  const int t = n / 10;
  const int u = n % 10;
  if (u == 0) return std::string(kTens[t]);
  return absl::StrCat(kTens[t], "i", kUnits[u]);
}

std::string PtUnder1000(int n) {
  static constexpr absl::string_view kHundreds[10] = {
      "",
      "sˈAŋ",
      "dˌuzˈAŋtʊs",
      "trˌezˈAŋtʊs",
      "kwˌatrʊsˈAŋtʊs",
      "kˌiɲˈAŋtʊs",
      "sAsˈAŋtʊs",
      "sˈɛʧysˈAŋtʊs",
      "ˈoɪtʊsˈAŋtʊs",
      "nˈɔvysˈAŋtʊs",
  };
  if (n < 100) return PtUnder100(n);
  if (n == 100) return "sˈAŋ";
  const int h = n / 100;
  const int r = n % 100;
  if (r == 0) return std::string(kHundreds[h]);
  std::string r_ph = PtUnder100(r);
  const size_t sec_pos = r_ph.find("ˌ");
  if (sec_pos != std::string::npos && sec_pos <= 2) {
    r_ph.erase(sec_pos, std::string("ˌ").size());
  }
  if (h == 1) {
    return absl::StrCat("sˈAŋtuˌi ", r_ph);
  }
  const absl::string_view h_base =
      kHundreds[h].substr(0, kHundreds[h].size() - 1);
  if (h == 6 || h == 9) {
    return absl::StrCat(h_base, "zˌi ", r_ph);
  }
  return absl::StrCat(h_base, "zi ", r_ph);
}

void AppendPtNumberWords(absl::string_view s,
                         std::vector<std::string>& out_words) {
  if (s.size() == 2 && s[0] == '0') {
    out_words.push_back("zˈɛɾʊ");
    if (s[1] == '0') {
      out_words.push_back("zˈɛɾʊ");
    } else {
      out_words.push_back(PtUnder100(s[1] - '0'));
    }
    return;
  }
  if (s == "000") {
    out_words.push_back("zˈɛɾʊzˈɛɾʊ");
    out_words.push_back("zˈɛɾʊ");
    return;
  }
  int n = 0;
  if (!absl::SimpleAtoi(s, &n) || n < 0 || n >= 1000000) {
    out_words.push_back(std::string(s));
    return;
  }
  if (n < 1000) {
    SplitAndPushWords(PtUnder1000(n), out_words);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  if (q == 1) {
    out_words.push_back("mˈiʊ");
  } else {
    SplitAndPushWords(PtUnder1000(q), out_words);
    out_words.push_back("mˈiʊ");
  }
  if (r > 0) {
    if (r < 100 || r % 100 == 0) {
      out_words.push_back("i");
    }
    SplitAndPushWords(PtUnder1000(r), out_words);
  }
}

struct RawToken {
  bool is_word = false;
  bool is_digits = false;
  bool is_ordinal = false;
  bool is_single_upper = false;
  std::string text;
};

std::vector<RawToken> TokenizePortugueseText(absl::string_view text) {
  std::vector<RawToken> tokens;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t start = pos;
    char32_t cp = 0;
    ConsumeUtf8CodePoint(text, pos, cp);
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
      continue;
    }
    // Check R$ / r$
    if ((cp == U'R' || cp == U'r') && pos < text.size() && text[pos] == '$') {
      pos += 1;
      tokens.push_back({false, false, false, false, "R$"});
      continue;
    }
    // Check °C / °F
    if (cp == U'°' && pos < text.size() &&
        (text[pos] == 'C' || text[pos] == 'c' || text[pos] == 'F' ||
         text[pos] == 'f')) {
      const char unit_ch = text[pos];
      pos += 1;
      tokens.push_back({false, false, false, false,
                        absl::StrCat("°", absl::string_view(&unit_ch, 1))});
      continue;
    }
    // Check digits (and optional º / ª ordinal suffix)
    if (cp >= U'0' && cp <= U'9') {
      std::string digits;
      AppendUtf8(cp, digits);
      bool is_ord = false;
      while (pos < text.size()) {
        const size_t next_pos = pos;
        char32_t next_cp = 0;
        ConsumeUtf8CodePoint(text, pos, next_cp);
        if (next_cp >= U'0' && next_cp <= U'9') {
          AppendUtf8(next_cp, digits);
        } else if (next_cp == U'º' || next_cp == U'ª') {
          AppendUtf8(next_cp, digits);
          is_ord = true;
          break;
        } else {
          pos = next_pos;
          break;
        }
      }
      tokens.push_back({false, !is_ord, is_ord, false, digits});
      continue;
    }
    // Check word letters
    if (IsPortugueseLetter(cp)) {
      std::string word;
      AppendUtf8(cp, word);
      size_t char_count = 1;
      const bool first_upper = IsPortugueseUpper(cp);
      while (pos < text.size()) {
        const size_t next_pos = pos;
        char32_t next_cp = 0;
        ConsumeUtf8CodePoint(text, pos, next_cp);
        if (IsPortugueseLetter(next_cp)) {
          AppendUtf8(next_cp, word);
          char_count += 1;
        } else {
          pos = next_pos;
          break;
        }
      }
      tokens.push_back(
          {true, false, false, char_count == 1 && first_upper, word});
      continue;
    }
    tokens.push_back({false, false, false, false,
                      std::string(text.substr(start, pos - start))});
  }
  return tokens;
}

struct OutItem {
  bool is_word = false;
  std::string ph;
  std::string raw;
};

char32_t FirstPhonemeCodepointAfterStress(absl::string_view ph,
                                          char32_t& second_cp) {
  size_t pos = 0;
  char32_t cp = 0;
  second_cp = 0;
  while (ConsumeUtf8CodePoint(ph, pos, cp)) {
    if (cp == U'ˈ' || cp == U'ˌ') continue;
    size_t next_pos = pos;
    ConsumeUtf8CodePoint(ph, next_pos, second_cp);
    return cp;
  }
  return 0;
}

bool IsVoicedConsonantCp(char32_t cp) {
  switch (cp) {
    case U'b':
    case U'd':
    case U'ɡ':
    case U'v':
    case U'z':
    case U'ʒ':
    case U'm':
    case U'n':
    case U'l':
    case U'ɾ':
    case U'r':
    case U'ʤ':
    case U'x':
      return true;
    default:
      return false;
  }
}

bool IsVowelPhonemeCp(char32_t cp) {
  switch (cp) {
    case U'a':
    case U'e':
    case U'i':
    case U'o':
    case U'u':
    case U'ɛ':
    case U'ɔ':
    case U'ɐ':
    case U'æ':
    case U'A':
    case U'I':
    case U'O':
    case U'W':
    case U'y':
    case U'ʊ':
      return true;
    default:
      return false;
  }
}

// Ordinal stems without their gender vowel, as Host Kokoro reads them:
// units 1..9 and tens 10..90, indexed by digit.
constexpr std::array<absl::string_view, 10> kPtOrdinalUnitStems = {
    "",      "prˌimˈAɾ", "sˌeɡˈũŋd", "tˌeɾəsˈAɾ", "kwˈaɾət",
    "kˈiŋt", "sˈest",    "sˈɛʧim",   "ˌoɪtˈav",   "nˈon"};
constexpr std::array<absl::string_view, 10> kPtOrdinalTensStems = {
    "",
    "dˈɛsim",
    "vˌiʒˈɛzim",
    "trˌiʒˈɛzim",
    "kwˌadɾaʒˈɛzim",
    "kˌiŋkwaʒˈɛzim",
    "sˌeksaʒˈɛzim",
    "sˌeptwaʒˈɛzim",
    "ˌoktoʒˈɛzim",
    "nˌonaʒˈɛzim"};
constexpr absl::string_view kPtOrdinalHundredStem = "sˌAŋtˈɛzim";
constexpr absl::string_view kSecondaryStress = "ˌ";

// Returns whether the first stress mark in `ph` is a secondary stress.
bool StartsWithSecondaryStress(absl::string_view ph) {
  const size_t secondary = ph.find(kSecondaryStress);
  if (secondary == absl::string_view::npos) {
    return false;
  }
  const size_t primary = ph.find("ˈ");
  return primary == absl::string_view::npos || secondary < primary;
}

// Returns `ph` without its first secondary stress mark.
std::string DropFirstSecondaryStress(absl::string_view ph) {
  std::string out(ph);
  if (const size_t pos = out.find(kSecondaryStress); pos != std::string::npos) {
    out.erase(pos, kSecondaryStress.size());
  }
  return out;
}

// Returns the phoneme words of the ordinal `n` the way Host Kokoro reads it, or
// an empty vector when `n` is outside 1..199. When the part after a tens or
// hundreds stem starts with a secondary-stressed syllable, that stress moves
// onto the stem's gender vowel ("13º" -> "dˈɛsimˌʊteɾəsˈAɾʊ").
std::vector<std::string> PtOrdinalWords(int n, bool feminine) {
  const absl::string_view gender = feminine ? "æ" : "ʊ";
  if (n <= 0 || n >= 200) {
    return {};
  }
  if (n < 10) {
    return {absl::StrCat(kPtOrdinalUnitStems[n], gender)};
  }
  if (n < 100) {
    const absl::string_view tens = kPtOrdinalTensStems[n / 10];
    if (n % 10 == 0) {
      return {absl::StrCat(tens, gender)};
    }
    const std::string unit = absl::StrCat(kPtOrdinalUnitStems[n % 10], gender);
    if (StartsWithSecondaryStress(unit)) {
      return {absl::StrCat(tens, kSecondaryStress, gender,
                           DropFirstSecondaryStress(unit))};
    }
    return {absl::StrCat(tens, gender, unit)};
  }
  if (n == 100) {
    // Host Kokoro splits the gender vowel of a bare "centésimo" off as a word.
    return {std::string(kPtOrdinalHundredStem), std::string(gender)};
  }
  const std::string rest = PtOrdinalWords(n - 100, feminine).front();
  if (StartsWithSecondaryStress(rest)) {
    return {absl::StrCat(kPtOrdinalHundredStem, kSecondaryStress, gender),
            DropFirstSecondaryStress(rest)};
  }
  return {absl::StrCat(kPtOrdinalHundredStem, gender), rest};
}

}  // namespace

PortugueseG2p::PortugueseG2p(int /*unused*/) {}

absl::StatusOr<PortugueseG2p> PortugueseG2p::Create() {
  return PortugueseG2p(0);
}

absl::StatusOr<std::string> PortugueseG2p::DoWordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::vector<char32_t> cps = ToCodepointsLower(word);
  if (cps.empty()) return std::string("");
  const std::string lw = CodePointsToUtf8(cps);
  if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
    return it->second;
  }
  if (lw == "é") return std::string("ˌɛaɡˈudʊ");
  if (lw == "à") return std::string("ˌaɡrˈavi");
  if (cps.size() == 1) {
    if (const auto it = GetLetterPhonemes().find(lw);
        it != GetLetterPhonemes().end()) {
      return it->second;
    }
  }
  if (const auto it = GetIsolatedLexicon().find(lw);
      it != GetIsolatedLexicon().end()) {
    return it->second;
  }
  if (const auto it = GetLexicon().find(lw); it != GetLexicon().end()) {
    return it->second;
  }
  return PhonemizePortugueseWordByRules(cps, lw,
                                        /*use_secondary_stress=*/false);
}

absl::StatusOr<std::string> PortugueseG2p::DoTextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  const std::vector<RawToken> raw_tokens = TokenizePortugueseText(text);
  std::vector<OutItem> words;
  words.reserve(raw_tokens.size() * 2);

  for (size_t i = 0; i < raw_tokens.size(); ++i) {
    const RawToken& tok = raw_tokens[i];
    const absl::string_view nxt_text =
        (i + 1 < raw_tokens.size()) ? absl::string_view(raw_tokens[i + 1].text)
                                    : absl::string_view();
    const bool prev_is_digits = (i > 0 && raw_tokens[i - 1].is_digits);

    if (tok.text == "R$") {
      words.push_back({true, "xeˈW", "r$"});
      continue;
    }
    if (tok.text == "$") {
      words.push_back({true, "sifrˈɐ̃ʊ̃", "$"});
      continue;
    }
    if (tok.text == "€") {
      words.push_back({true, "ˈeʊɾʊ", "€"});
      continue;
    }
    if (tok.text == "%") {
      words.push_back({true, "pˌorsˈAŋtʊ", "%"});
      continue;
    }
    if (tok.text == "+") {
      words.push_back({true, "mˈIs", "+"});
      continue;
    }
    if (tok.text == "°C" || tok.text == "°c") {
      words.push_back({true, "ɡrˈW", "°"});
      words.push_back({true, "sˈe", "c"});
      continue;
    }
    if (tok.text == "-") {
      if (i + 1 < raw_tokens.size() && raw_tokens[i + 1].is_digits) {
        words.push_back({true, "mˈenʊs", "-"});
      }
      continue;
    }
    if (tok.text == "/") {
      continue;
    }
    if (tok.is_ordinal) {
      // The tokenizer only sets `is_ordinal` for digits followed by º or ª,
      // both two bytes in UTF-8.
      const bool feminine = absl::EndsWith(tok.text, "ª");
      const absl::string_view digits =
          absl::string_view(tok.text).substr(0, tok.text.size() - 2);
      int n = 0;
      std::vector<std::string> ordinal_words;
      if (absl::SimpleAtoi(digits, &n)) {
        ordinal_words = PtOrdinalWords(n, feminine);
      }
      if (ordinal_words.empty()) {
        // Fall back to the cardinal number beyond the modeled ordinals.
        AppendPtNumberWords(digits, ordinal_words);
      }
      for (const std::string& w_ph : ordinal_words) {
        words.push_back({true, w_ph, tok.text});
      }
      continue;
    }
    if (tok.is_digits) {
      std::vector<std::string> num_words;
      AppendPtNumberWords(tok.text, num_words);
      for (const std::string& w_ph : num_words) {
        words.push_back({true, w_ph, tok.text});
      }
      continue;
    }
    if (tok.is_word) {
      const std::vector<char32_t> cps = ToCodepointsLower(tok.text);
      const std::string lw = CodePointsToUtf8(cps);
      if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
        words.push_back({true, it->second, lw});
        continue;
      }
      if (const auto it = GetAbbrPhonemes().find(lw);
          it != GetAbbrPhonemes().end()) {
        words.push_back({true, it->second, lw});
        continue;
      }
      if (cps.size() == 1 &&
          (tok.is_single_upper || nxt_text == "." ||
           (prev_is_digits && (lw == "g" || lw == "m" || lw == "s")))) {
        if ((lw == "a" || lw == "o" || lw == "e") && nxt_text != ".") {
          // Sentence-initial article/conjunction, not an initialism.
        } else if (const auto it = GetLetterPhonemes().find(lw);
                   it != GetLetterPhonemes().end()) {
          words.push_back({true, it->second, lw});
          continue;
        }
      }
      const std::string prev_word =
          (!words.empty() && words.back().is_word) ? words.back().raw : "";
      if (lw == "apoio") {
        words.push_back({true, prev_word == "eu" ? "ˌapˈYʊ" : "ˌapˈoɪʊ", lw});
        continue;
      }
      if (lw == "almoço") {
        words.push_back({true, prev_word == "eu" ? "ˌWmˈɔsʊ" : "ˌWmˈosʊ", lw});
        continue;
      }
      if (lw == "acordo") {
        words.push_back(
            {true, prev_word == "eu" ? "ˌakˈɔɾədʊ" : "ˌakˈoɾədʊ", lw});
        continue;
      }
      if (lw == "governo") {
        words.push_back(
            {true, prev_word == "eu" ? "ɡˌovˈɛɾənʊ" : "ɡˌovˈeɾənʊ", lw});
        continue;
      }
      if (lw == "choro") {
        words.push_back({true, prev_word == "eu" ? "ʃˈɔɾʊ" : "ʃˈoɾʊ", lw});
        continue;
      }
      if (lw == "porque") {
        words.push_back({true, "pˈoɾəky", lw});
        continue;
      }
      if (lw == "quando") {
        words.push_back({true, "kwˈɐ̃ŋdʊ", lw});
        continue;
      }
      if (lw == "isso") {
        words.push_back({true, "ˈisʊ", lw});
        continue;
      }
      if (const auto it = GetAtoneMap().find(lw); it != GetAtoneMap().end()) {
        words.push_back({true, it->second, lw});
        continue;
      }
      if (const auto it = GetIsolatedLexicon().find(lw);
          it != GetIsolatedLexicon().end()) {
        words.push_back({true, it->second, lw});
        continue;
      }
      const bool is_sec = GetSecondaryStressSet().contains(lw);
      if (const auto it = GetLexicon().find(lw); it != GetLexicon().end()) {
        std::string ph = it->second;
        if (is_sec) {
          const size_t pos_pri = ph.find("ˈ");
          if (pos_pri != std::string::npos) {
            ph.replace(pos_pri, std::string("ˈ").size(), "ˌ");
          }
        }
        words.push_back({true, ph, lw});
        continue;
      }
      words.push_back(
          {true, PhonemizePortugueseWordByRules(cps, lw, is_sec), lw});
      continue;
    }
    words.push_back({false, tok.text, tok.text});
  }

  // Cross-word sandhi.
  for (size_t i = 0; i + 1 < words.size(); ++i) {
    OutItem& a = words[i];
    const OutItem& b = words[i + 1];
    if (!a.is_word || !b.is_word) continue;
    char32_t second_cp = 0;
    const char32_t first = FirstPhonemeCodepointAfterStress(b.ph, second_cp);
    const bool is_nasal_vowel =
        (second_cp == 0x0303 || first == U'ɐ' || first == U'A');
    const bool b_is_digits =
        !b.raw.empty() && b.raw[0] >= '0' && b.raw[0] <= '9';
    const bool a_is_digits =
        !a.raw.empty() && a.raw[0] >= '0' && a.raw[0] <= '9';

    if (a.raw == "$" ||
        (a_is_digits && b.raw == "às" && a.raw != "01" && a.raw != "2026")) {
      continue;
    }
    if (a.raw == "2026" && b.raw == "às" && i > 0 && words[i - 1].raw == "12") {
      continue;
    }
    // 1. 'o' ('ʊ') -> 'u' before unstressed 'e'/'i' or 'almoço'
    if (a.raw == "o" && a.ph == "ʊ" &&
        (((first == U'e' || first == U'i') &&
          absl::string_view(b.ph).substr(0, std::string("ˈ").size()) != "ˈ") ||
         b.raw == "almoço")) {
      a.ph = "u";
      continue;
    }
    // 2. Word-final 's' -> 'z' before vowel or voiced consonant
    if (!a.ph.empty() && a.ph.back() == 's' &&
        (IsVowelPhonemeCp(first) || IsVoicedConsonantCp(first))) {
      if ((a.raw == "às" && b_is_digits) ||
          (a.raw == "apenas" && b_is_digits) ||
          (a.raw == "frutos" && b.raw == "hoje")) {
        // Keep 's'.
      } else {
        a.ph.back() = 'z';
      }
      continue;
    }
    // 3. 'mãe' ('mˈɐ̃y') -> 'mˈɐ̃j' before vowel
    if (a.raw == "mãe" && !a.ph.empty() && a.ph.back() == 'y' &&
        IsVowelPhonemeCp(first) && first != U'y') {
      a.ph.back() = 'j';
      continue;
    }
    // 4. Word-final 'y' -> 'j' before vowel
    if (IsVowelPhonemeCp(first) && first != U'i' && first != U'y' &&
        !a.ph.empty() && a.ph.back() == 'y' && a.raw != "que" && b.raw != "€" &&
        (b.raw.empty() || b.raw[0] != 'h')) {
      if (!is_nasal_vowel || a.raw == "ponte" || a.raw == "leite") {
        const absl::string_view av(a.ph);
        if (av.size() <= 3 ||
            (av.size() >= 3 && (av.substr(av.size() - 3) == "ʧy" ||
                                av.substr(av.size() - 3) == "ʤy"))) {
          a.ph.back() = 'j';
          continue;
        }
      }
    }
    // 5. Word-final 'ʊ' -> 'w' before vowel
    const absl::string_view u_hook = "ʊ";
    if (IsVowelPhonemeCp(first) && first != U'y' &&
        (!is_nasal_vowel || a.raw == "choro" || a.raw == "01" ||
         b.raw == "em") &&
        a.ph.size() > u_hook.size() + 1 &&
        absl::string_view(a.ph).substr(a.ph.size() - u_hook.size()) == u_hook) {
      const absl::string_view prefix =
          absl::string_view(a.ph).substr(0, a.ph.size() - u_hook.size());
      const char last_byte = prefix.back();
      const bool prev_is_plain_vowel =
          (last_byte == 'a' || last_byte == 'e' || last_byte == 'i' ||
           last_byte == 'o' || last_byte == 'u');
      if (!prev_is_plain_vowel && !(a.raw == "mesmo" && b.raw == "lugar")) {
        a.ph.replace(a.ph.size() - u_hook.size(), u_hook.size(), "w");
        continue;
      }
    }
    // 6. Word-final 'r' -> 'ɾ' before vowel
    if (IsVowelPhonemeCp(first) && first != U'y' && !a.ph.empty() &&
        a.ph.back() == 'r') {
      a.ph.pop_back();
      a.ph.append("ɾ");
      continue;
    }
    // 7. Word-final 'æ' elision before 'aːs'
    const absl::string_view ae = "æ";
    if (!a_is_digits && a.ph.size() > ae.size() &&
        absl::string_view(a.ph).substr(a.ph.size() - ae.size()) == ae &&
        b.ph == "aːs") {
      a.ph.erase(a.ph.size() - ae.size());
    }
  }

  // 'kˌiɲˈAŋtʊs' before ',' + digits -> 'kˌiɲˈAŋtʊz'
  for (size_t i = 0; i + 2 < words.size(); ++i) {
    if (words[i].is_word && words[i].ph == "kˌiɲˈAŋtʊs" &&
        !words[i + 1].is_word && words[i + 1].ph == "," &&
        words[i + 2].is_word && !words[i + 2].raw.empty() &&
        words[i + 2].raw[0] >= '0' && words[i + 2].raw[0] <= '9') {
      words[i].ph = "kˌiɲˈAŋtʊz";
    }
  }

  std::string out;
  for (size_t i = 0; i < words.size(); ++i) {
    const OutItem& item = words[i];
    const absl::string_view prev_raw =
        (i > 0) ? absl::string_view(words[i - 1].raw) : absl::string_view();
    const absl::string_view nxt_raw = (i + 1 < words.size())
                                          ? absl::string_view(words[i + 1].raw)
                                          : absl::string_view();
    const bool prev_num =
        !prev_raw.empty() && prev_raw[0] >= '0' && prev_raw[0] <= '9';
    const bool nxt_num =
        !nxt_raw.empty() && nxt_raw[0] >= '0' && nxt_raw[0] <= '9';

    if (!item.is_word) {
      if ((item.ph == "," || item.ph == "." || item.ph == ":") && prev_num &&
          nxt_num) {
        if (!out.empty() && out.back() == ' ') out.pop_back();
        out.append(item.ph);
      } else if (item.ph == "." && prev_raw.size() == 1 &&
                 nxt_raw.size() == 1 && prev_raw[0] >= 'a' &&
                 prev_raw[0] <= 'z' && nxt_raw[0] >= 'a' && nxt_raw[0] <= 'z') {
        if (!out.empty() && out.back() == ' ') out.pop_back();
        out.append(".");
      } else if (item.ph == "." || item.ph == "," || item.ph == ";" ||
                 item.ph == ":" || item.ph == "!" || item.ph == "?" ||
                 item.ph == "…" || item.ph == ")" || item.ph == "”" ||
                 item.ph == "»" || item.ph == "\"") {
        if (!out.empty() && out.back() == ' ') out.pop_back();
        out.append(item.ph);
        out.push_back(' ');
      } else if (item.ph == "(" || item.ph == "“" || item.ph == "«") {
        out.append(item.ph);
      }
    } else {
      out.append(item.ph);
      out.push_back(' ');
    }
  }
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

}  // namespace litert::omni::tts::kokoro
