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

// Example usage:
// bazel run -c opt \
//   //omni/text2image:text2image_runner -- \
//   --model_path="$HOME/models/flux2/bonsai" \
//   --model_type="bonsai_flux2" \
//   --prompt="a red fox sitting in fresh snow at sunrise" \
//   --img_size=256 \
//   --steps=4 \
//   --seed=42 \
//   --output_png="/tmp/fox.png" \
//   --num_threads=8 --v=2 --alsologtostderr

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/initialize.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "stb_image_write.h"  // from @stb

ABSL_FLAG(std::string, model_path, "",
          "Path to the text-to-image model (.litertlm file or directory).");
ABSL_FLAG(std::string, model_type, "bonsai_flux2",
          "Text-to-image model architecture type: 'bonsai_flux2', "
          "'flux2_klein' (or empty to auto-detect).");
ABSL_FLAG(std::string, prompt, "a red fox sitting in fresh snow at sunrise",
          "Text prompt to generate an image from.");
ABSL_FLAG(std::string, output_png, "/tmp/output.png",
          "Path to save the generated RGB888 PNG image file.");
ABSL_FLAG(int, img_size, 0,
          "Target output image width and height in pixels (e.g., 512 or 256). "
          "If 0, uses the model's configured img_size.");
ABSL_FLAG(int, steps, 4, "Number of FlowMatch Euler denoising steps.");
ABSL_FLAG(uint64_t, seed, 42, "Random seed for initial Gaussian latent noise.");
ABSL_FLAG(int, num_threads, 4,
          "Number of CPU threads for LiteRT model execution.");
ABSL_FLAG(std::string, backend, "cpu",
          "Backend to use for model execution: 'cpu', 'gpu', or 'npu'.");
ABSL_FLAG(std::string, textenc_backend, "",
          "Optional backend override for text encoder ('cpu', 'gpu', or "
          "'npu'). If empty, inherits the engine backend.");
ABSL_FLAG(std::string, dit_backend, "",
          "Optional backend override for DiT denoiser ('cpu', 'gpu', or "
          "'npu'). If empty, inherits the engine backend.");
ABSL_FLAG(std::string, vae_backend, "",
          "Optional backend override for VAE decoder ('cpu', 'gpu', or 'npu'). "
          "If empty, inherits the engine backend.");
ABSL_FLAG(std::string, cache_dir, "",
          "Directory to store/load XNNPack weight cache.");

namespace {

absl::Status WritePngFile(absl::string_view filename,
                          const litert::omni::ImageOutput& image) {
  if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid image dimensions: %dx%dx%d", image.width,
                        image.height, image.channels));
  }
  const int stride_in_bytes = image.width * image.channels;
  if (stbi_write_png(std::string(filename).c_str(), image.width, image.height,
                     image.channels, image.rgb_data.data(),
                     stride_in_bytes) == 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to write PNG file: %s", filename));
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string model_path = absl::GetFlag(FLAGS_model_path);

  const std::string prompt_text = absl::GetFlag(FLAGS_prompt);
  if (prompt_text.empty()) {
    return absl::InvalidArgumentError("--prompt flag must not be empty.");
  }

  const int img_size = absl::GetFlag(FLAGS_img_size);
  const int steps = absl::GetFlag(FLAGS_steps);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);

  litert::omni::OmniEngine::Settings engine_settings;
  engine_settings.cache_dir = absl::GetFlag(FLAGS_cache_dir);
  engine_settings.num_threads = absl::GetFlag(FLAGS_num_threads);
  ABSL_ASSIGN_OR_RETURN(
      engine_settings.backend,
      litert::omni::ParseOmniBackend(absl::GetFlag(FLAGS_backend)));

  litert::lm::proto::Text2ImageSessionConfig session_config;
  if (img_size > 0) {
    session_config.set_width(img_size);
    session_config.set_height(img_size);
  }
  session_config.set_num_inference_steps(steps);
  session_config.set_seed(seed);

  std::string model_type_str = absl::GetFlag(FLAGS_model_type);
  if (model_type_str.empty()) {
    model_type_str = "bonsai_flux2";
  }
  litert::lm::proto::Text2ImageMetadata metadata;
  litert::lm::proto::Flux2Params* flux2_params = nullptr;
  std::string model_name;
  if (model_type_str == "flux2_klein" || model_type_str == "flux2-klein" ||
      model_type_str == "flux.2-klein-4b") {
    model_name = "flux2-klein";
    flux2_params = metadata.mutable_image_gen_model_type()
                       ->mutable_flux2_klein()
                       ->mutable_flux2_params();
  } else if (model_type_str == "bonsai_flux2" ||
             model_type_str == "bonsai-flux2" || model_type_str == "flux2") {
    model_name = "bonsai-flux2";
    flux2_params = metadata.mutable_image_gen_model_type()
                       ->mutable_bonsai_flux2()
                       ->mutable_flux2_params();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown model_type: ", model_type_str));
  }

  const std::string effective_cache_dir = engine_settings.cache_dir.empty()
                                              ? "/tmp/omni_models"
                                              : engine_settings.cache_dir;
  if (!model_path.empty()) {
    engine_settings.model_path = model_path;
  } else {
    ABSL_ASSIGN_OR_RETURN(
        litert::omni::OmniRegistry::RegisteredModel reg_model,
        litert::omni::OmniRegistry::Get().GetRegisteredModel(model_name));
    engine_settings.model_remote_url = reg_model.model_remote_url;
    std::string model_ext =
        std::filesystem::path(engine_settings.model_remote_url)
            .extension()
            .string();
    if (model_ext.empty()) {
      model_ext = ".litertlm";
    }
    engine_settings.model_path = (std::filesystem::path(effective_cache_dir) /
                                  absl::StrCat(model_name, model_ext))
                                     .string();
    if (!engine_settings.model_remote_url.empty() &&
        !std::filesystem::exists(engine_settings.model_path)) {
      ABSL_RETURN_IF_ERROR(litert::omni::DownloadFileWithCurl(
          engine_settings.model_remote_url, engine_settings.model_path));
    }
  }

  if (img_size > 0) {
    flux2_params->set_img_size(img_size);
  }
  flux2_params->set_default_steps(steps);

  std::cout << "Loading OmniEngine from: " << engine_settings.model_path
            << "\n";
  absl::Time load_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(auto engine, litert::omni::OmniEngine::Create(
                                         model_name, engine_settings,
                                         &session_config, &metadata));
  double load_time_sec = absl::ToDoubleSeconds(absl::Now() - load_start);

  auto input_source = std::make_unique<litert::omni::PushInputSource>();
  auto* raw_input_source = input_source.get();
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(input_source)));

  std::cout << "Generating image for prompt: \"" << prompt_text
            << "\" (steps=" << steps << ", seed=" << seed << ")\n";
  absl::Time gen_start = absl::Now();

  ABSL_RETURN_IF_ERROR(raw_input_source->PushInput(
      litert::omni::OmniSession::TextInput{.text = prompt_text}));
  ABSL_ASSIGN_OR_RETURN(auto output, session->ProcessNext());
  const auto& image = std::get<litert::omni::ImageOutput>(output);

  double gen_time_sec = absl::ToDoubleSeconds(absl::Now() - gen_start);

  std::cout << "\n============================================================="
               "===================\n";
  std::cout << "                    Text2ImageEngine Execution Summary         "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << absl::StrFormat("  Model Load Latency           : %8.3f s\n",
                               load_time_sec);
  std::cout << absl::StrFormat("  Generation Wall-Clock Latency: %8.3f s\n",
                               gen_time_sec);
  std::cout << absl::StrFormat(
      "  Generated Image Dimensions   : %d x %d x %d (%zu bytes)\n",
      image.width, image.height, image.channels, image.rgb_data.size());
  std::cout << "==============================================================="
               "=================\n\n";

  const std::string output_png = absl::GetFlag(FLAGS_output_png);
  if (!output_png.empty()) {
    ABSL_RETURN_IF_ERROR(WritePngFile(output_png, image));
    std::cout << "Saved image to: " << output_png << "\n";
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  absl::Status status = Run();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Text-to-image execution failed: " << status;
    std::cerr << "ERROR: " << status << "\n";
    return 1;
  }

  return 0;
}
