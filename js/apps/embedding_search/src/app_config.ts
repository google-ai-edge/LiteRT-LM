/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @fileoverview Deployment knobs for the demo. Each URL can be overridden at
 * runtime with a query parameter so the same build can point at staging
 * assets, e.g. `?model=https://example.com/model.litertlm&samples=https://example.com/samples`.
 */

function readUrlParam(name: string): string|null {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get(name);
  return value ? value : null;
}

/**
 * Reads a URL-valued query parameter and resolves it against the page, so a
 * relative override such as `?model=/models/x.litertlm` becomes an absolute
 * URL that `new URL()` accepts everywhere else in the app. Values that do not
 * parse are ignored in favour of `fallback`.
 */
function readUrlParamAsUrl(name: string, fallback: string|null): string|null {
  const value = readUrlParam(name);
  if (!value) return fallback;
  try {
    return new URL(value, window.location.href).toString();
  } catch {
    console.warn(`[EmbeddingSearch] Ignoring invalid ?${name}= value: ${value}`);
    return fallback;
  }
}

/**
 * Where to download the EmbeddingGemma model from on startup (external
 * weights variant). Until the model repository is public on Hugging Face,
 * upload a local `.litertlm` file or override with `?model=`.
 */
const DEFAULT_MODEL_URL =
    'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/main/embeddinggemma-2-740m.litertlm';

/**
 * Bucket directory of sample photos to index out of the box, fetched as
 * `<SAMPLES_BASE_URL>/<filename>`.
 */
const DEFAULT_SAMPLES_BASE_URL =
    'https://storage.googleapis.com/litert/litert_lm/gallery_sample_images';

/** The model URL to fetch, or null if the user must upload a model file. */
export const MODEL_URL: string|null =
    readUrlParamAsUrl('model', DEFAULT_MODEL_URL);

/** Base URL under which the sample album images are served. */
export const SAMPLES_BASE_URL: string =
    (readUrlParam('samples') ?? DEFAULT_SAMPLES_BASE_URL).replace(/\/+$/, '');

/**
 * Filenames of the sample album photos: a curated 100-photo subset of the
 * AI-generated AI Edge Gallery Smart Album sample set.
 */
export const SAMPLE_IMAGE_FILES: readonly string[] = [
  'acoustic_guitar.jpg',
  'aquarium_fish.jpg',
  'birthday_cake.jpg',
  'city_canal_footbridge.jpg',
  'city_cobblestone_street.jpg',
  'city_neon_street.jpg',
  'city_night_traffic_trails.jpg',
  'city_rainy_pavement_reflection.jpg',
  'city_skyline_from_park.jpg',
  'city_subway_entrance.jpg',
  'city_suspension_bridge.jpg',
  'doc_business_card_wood.jpg',
  'doc_chalkboard_cafe_menu.jpg',
  'doc_coffee_shop_receipt.jpg',
  'doc_doctor_appointment_card.jpg',
  'doc_grocery_list_fridge.jpg',
  'doc_grocery_receipt.jpg',
  'doc_handwritten_recipe_notes.jpg',
  'doc_handwritten_todo_list.jpg',
  'doc_restaurant_check_tray.jpg',
  'doc_shipping_box_label.jpg',
  'event_balloon_arch.jpg',
  'event_cheers_toast.jpg',
  'event_holiday_lights.jpg',
  'event_party_balloons.jpg',
  'event_sparkler_cake.jpg',
  'food_avocado_toast.jpg',
  'food_caprese_salad.jpg',
  'food_croissant_plate.jpg',
  'food_fruit_salad.jpg',
  'food_iced_latte.jpg',
  'food_margherita_pizza.jpg',
  'food_pancakes_syrup.jpg',
  'food_spaghetti_plate.jpg',
  'forest_waterfall.jpg',
  'hobby_camping_tent.jpg',
  'hobby_chess_board.jpg',
  'hobby_guitar_picks.jpg',
  'hobby_mountain_bike.jpg',
  'hobby_watercolor_palette.jpg',
  'land_banff_turquoise_lake.jpg',
  'land_foggy_beach_walk.jpg',
  'land_fuji_mountain_lake.jpg',
  'land_grand_canyon_overlook.jpg',
  'land_iceland_black_beach.jpg',
  'land_kyoto_bamboo_temple.jpg',
  'land_lavender_field_hills.jpg',
  'land_misty_lake_pier.jpg',
  'land_swiss_alps_meadow.jpg',
  'land_tropical_palm_beach.jpg',
  'land_tuscany_rolling_hills.jpg',
  'macro_frost_crystals.jpg',
  'macro_peacock_feather.jpg',
  'macro_spider_web_dew.jpg',
  'person_cafe_barista.jpg',
  'person_cycling_street.jpg',
  'person_dog_fetch_beach.jpg',
  'person_dog_walk_park.jpg',
  'person_farmers_market.jpg',
  'person_guitar_campfire.jpg',
  'person_hiking_selfie.jpg',
  'person_jogging_beach.jpg',
  'person_kayak_paddling_lake.jpg',
  'person_picnic_blanket.jpg',
  'person_rainy_umbrella.jpg',
  'person_yoga_sunset.jpg',
  'pet_cat_in_box.jpg',
  'pet_corgi_playing_ball.jpg',
  'pet_dachshund_in_yard.jpg',
  'pet_golden_retriever_nap.jpg',
  'pet_hedgehog_curled_up.jpg',
  'pet_puppy_chewing_toy.jpg',
  'pet_rabbit_on_grass.jpg',
  'pet_siamese_cat_window.jpg',
  'pet_tabby_cat_sunbeam.jpg',
  'pet_turtle_basking_rock.jpg',
  'sim_camping_twilight_campfire.jpg',
  'sim_luna_backyard_ball.jpg',
  'sim_luna_couch_nap.jpg',
  'sim_luna_hiking_trail.jpg',
  'sim_milo_cardboard_box.jpg',
  'sim_milo_on_laptop.jpg',
  'sim_milo_windowbox.jpg',
  'sim_people_gathering_birthday_cheers.jpg',
  'sim_people_gathering_picnic_park.jpg',
  'sim_people_gathering_pizza_night.jpg',
  'sim_person_hiker_scenic_overlook.jpg',
  'sim_person_hiker_summit_selfie.jpg',
  'sim_receipt_cafe_coffee.jpg',
  'sim_receipt_gas_station.jpg',
  'sim_receipt_grocery_supermarket.jpg',
  'sim_sourdough_baked_cooling.jpg',
  'tech_laptop_coffee.jpg',
  'tech_monitor_code.jpg',
  'tech_smartwatch_charging.jpg',
  'tech_vr_headset.jpg',
  'veh_city_taxi_street.jpg',
  'veh_commuter_train.jpg',
  'veh_electric_scooter.jpg',
  'veh_food_truck.jpg',
  'veh_harbor_sail_boat.jpg',
];

/** Cache Storage bucket for model weights, shared with the chat demo. */
export const MODEL_CACHE_NAME = 'litertlm-models';

/** Human-readable labels for the prompt template options. */
export const PROMPT_TEMPLATE_LABELS = {
  'none': 'Raw (Gallery default)',
  'retrieval': 'EmbeddingGemma retrieval',
} as const;

/**
 * How text is wrapped before embedding.
 *
 * - `none`: Raw text. This is what the AI Edge Gallery Smart Album does for
 *   its multimodal search (it sets `disable_instruction_prefix`).
 * - `retrieval`: The EmbeddingGemma retrieval prompts (see the "Prompt
 *   instructions" section of the EmbeddingGemma model card). Only applied to
 *   text-only inputs; queries and documents containing media stay raw.
 */
export type PromptTemplateId = keyof typeof PROMPT_TEMPLATE_LABELS;

/** Wraps a text-only search query according to the template. */
export function applyQueryTemplate(
    template: PromptTemplateId, text: string): string {
  return template === 'retrieval' ? `task: search result | query: ${text}` :
                                    text;
}

/** Wraps a text document according to the template. */
export function applyDocumentTemplate(
    template: PromptTemplateId, text: string, title = 'none'): string {
  return template === 'retrieval' ? `title: ${title} | text: ${text}` : text;
}

/**
 * One element of an example query. `image` parts are resolved against the
 * indexed images at click time: the first image whose path contains `hint`
 * is used, otherwise the `fallbackIndex`-th image (so different examples
 * pick different photos).
 */
export type ExampleQueryPart = {
  text: string;
}|{
  image: {hint?: string; fallbackIndex: number};
};

/** A clickable example query shown before the first search. */
export interface ExampleQuery {
  parts: ExampleQueryPart[];
}

/**
 * Example queries. The text-only ones are the Smart Album suggestion chips
 * from the AI Edge Gallery; the rest show off multimodal and interleaved
 * inputs.
 */
export const EXAMPLE_QUERIES: ExampleQuery[] = [
  {parts: [{text: 'grocery store receipt'}]},
  {parts: [{text: 'friends hiking selfie'}]},
  {parts: [{text: 'dog in backyard'}]},
  {parts: [{text: 'walking on cloudy beach'}]},
  {parts: [{image: {hint: 'pet_corgi', fallbackIndex: 0}}]},
  {
    parts: [
      {image: {hint: 'city_skyline', fallbackIndex: 1}},
      {text: 'but at night'},
    ],
  },
  {
    parts: [
      {text: 'a drawing of'},
      {image: {hint: 'pet_golden_retriever', fallbackIndex: 2}},
    ],
  },
  {
    parts: [
      {text: 'the same place as'},
      {image: {hint: 'land_tropical_palm_beach', fallbackIndex: 3}},
      {text: 'with more people'},
    ],
  },
];
