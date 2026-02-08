// Copyright 2025 Google LLC.
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

// MeZO Optimizer Benchmark
//
// Compares Vanilla MeZO, ConMeZO, and AGZO on:
//   1. Synthetic ill-conditioned quadratic (default, no model needed)
//   2. Real model LoRA fine-tuning (with --model_path, requires LoRA model)
//
// The synthetic benchmark uses d=10K parameters with condition number 100,
// with learning rate 1e-7 (stable for ZO-SGD: lr < C/(lambda_max * d)).
//
// Usage:
//   mezo_wikitext_bench                              # synthetic benchmark
//   mezo_wikitext_bench --num_steps=5000 --dim=50000 # custom synthetic
//   mezo_wikitext_bench --model_path=<path>          # real model (needs LoRA)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "c/engine.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct BenchConfig {
  // Synthetic mode settings.
  int dim = 10000;            // Parameter dimension (simulates LoRA params).
  float noise_std = 0.01f;    // Stochastic noise on loss evaluations.
  float condition_number = 100.0f;  // Eigenvalue spread of quadratic.

  // Model mode settings (optional).
  std::string model_path;
  std::string lora_path;
  std::string backend = "cpu";
  std::string train_data_path = "benchmark/data/wiki.train.raw";
  std::string eval_data_path = "benchmark/data/wiki.test.raw";
  int seq_len = 256;
  int num_eval_samples = 10;

  // Shared settings.
  int num_steps = 3000;
  int eval_every = 300;
  float lr = 1e-7f;
  float epsilon = 1e-3f;
  uint64_t seed = 42;
  int agzo_rank = 16;
};

// ---------------------------------------------------------------------------
// Run result
// ---------------------------------------------------------------------------
struct RunResult {
  std::string optimizer_name;
  std::vector<float> step_losses;
  std::vector<float> eval_losses;
  std::vector<int> eval_steps;
  double total_time_sec;
};

// ---------------------------------------------------------------------------
// Synthetic ill-conditioned quadratic loss
// ---------------------------------------------------------------------------
// f(w) = 0.5 * sum(lambda_i * (w_i - w*_i)^2) + noise
//
// The eigenvalues lambda_i are logarithmically spaced from 1 to
// condition_number, creating an ill-conditioned landscape that is hard for
// vanilla ZO methods but easier for momentum-based or subspace methods.

struct SyntheticLossContext {
  const std::vector<float>* eigenvalues;  // lambda_i for each dimension.
  const std::vector<float>* optimum;      // w*_i target values.
  float noise_std;
  std::mt19937* rng;
};

// Called by MeZO during optimization. Parameters point to the current w.
int SyntheticLossFn(void* user_data, float* loss_out) {
  auto* ctx = static_cast<SyntheticLossContext*>(user_data);
  // We don't have direct access to the parameter values here since MeZO
  // manages them internally. Instead, we store a pointer to the params
  // and compute the loss over the global parameter buffer.
  //
  // But actually the C API design passes params separately to the step
  // function. The loss_fn callback doesn't receive params — it evaluates
  // the "model" which has been perturbed in-place. So we need the params
  // pointer accessible here.
  //
  // We store it in the context.
  return -1;  // Placeholder — see SyntheticBenchLossFn below.
}

// The actual synthetic loss function with access to parameter data.
struct SyntheticBenchContext {
  float* params;              // Current parameter values (mutated by MeZO).
  int dim;
  const float* eigenvalues;   // lambda_i for each dimension.
  const float* optimum;       // w*_i target values.
  float noise_std;
  std::mt19937 rng;
};

int SyntheticBenchLossFn(void* user_data, float* loss_out) {
  auto* ctx = static_cast<SyntheticBenchContext*>(user_data);

  double loss = 0.0;
  for (int i = 0; i < ctx->dim; ++i) {
    float diff = ctx->params[i] - ctx->optimum[i];
    loss += 0.5 * ctx->eigenvalues[i] * diff * diff;
  }

  // Add stochastic noise to simulate mini-batch loss variance.
  if (ctx->noise_std > 0.0f) {
    std::normal_distribution<float> noise_dist(0.0f, ctx->noise_std);
    loss += noise_dist(ctx->rng);
  }

  *loss_out = static_cast<float>(loss);
  return 0;
}

// Compute exact loss (no noise) for evaluation.
float SyntheticEvalLoss(const float* params, int dim,
                        const float* eigenvalues, const float* optimum) {
  double loss = 0.0;
  for (int i = 0; i < dim; ++i) {
    float diff = params[i] - optimum[i];
    loss += 0.5 * eigenvalues[i] * diff * diff;
  }
  return static_cast<float>(loss);
}

// ---------------------------------------------------------------------------
// Run synthetic benchmark for one optimizer
// ---------------------------------------------------------------------------
RunResult RunSyntheticOptimizer(const BenchConfig& config,
                                const std::string& name,
                                int optimizer_mode,
                                const std::vector<float>& eigenvalues,
                                const std::vector<float>& optimum) {
  RunResult result;
  result.optimizer_name = name;

  printf("\n=== %s (lr=%.1e, mode=%d, dim=%d) ===\n",
         name.c_str(), config.lr, optimizer_mode, config.dim);

  // Allocate parameter buffer, initialized to zero (far from optimum).
  std::vector<float> params(config.dim, 0.0f);

  // Create MeZO parameter descriptor (single parameter block).
  LiteRtLmMeZoParameter mezo_param;
  mezo_param.name = "synthetic_params";
  mezo_param.data = params.data();
  mezo_param.num_elements = config.dim;
  mezo_param.apply_weight_decay = false;

  // Create MeZO config.
  LiteRtLmMeZoConfig* mezo_config = litert_lm_mezo_config_create();
  litert_lm_mezo_config_set_learning_rate(mezo_config, config.lr);
  litert_lm_mezo_config_set_epsilon(mezo_config, config.epsilon);
  litert_lm_mezo_config_set_seed(mezo_config, config.seed);
  litert_lm_mezo_config_set_optimizer_mode(mezo_config, optimizer_mode);
  if (optimizer_mode == 1) {  // ConMeZO
    litert_lm_mezo_config_set_momentum_decay(mezo_config, 0.9f);
    litert_lm_mezo_config_set_cone_angle(mezo_config, 0.7854f);
  }
  if (optimizer_mode == 2) {  // AGZO
    litert_lm_mezo_config_set_agzo_subspace_rank(mezo_config, config.agzo_rank);
  }

  LiteRtLmMeZoFineTuner* finetuner =
      litert_lm_mezo_finetuner_create(mezo_config);
  litert_lm_mezo_config_delete(mezo_config);

  if (!finetuner) {
    fprintf(stderr, "ERROR: Cannot create fine-tuner for %s\n", name.c_str());
    return result;
  }

  // Set up loss context.
  SyntheticBenchContext ctx;
  ctx.params = params.data();
  ctx.dim = config.dim;
  ctx.eigenvalues = eigenvalues.data();
  ctx.optimum = optimum.data();
  ctx.noise_std = config.noise_std;
  ctx.rng.seed(config.seed + optimizer_mode * 1000);

  // Initial evaluation.
  float eval_loss = SyntheticEvalLoss(params.data(), config.dim,
                                      eigenvalues.data(), optimum.data());
  printf("  Step %5d | eval_loss=%.6f\n", 0, eval_loss);
  result.eval_losses.push_back(eval_loss);
  result.eval_steps.push_back(0);

  auto t_start = std::chrono::steady_clock::now();

  for (int step = 1; step <= config.num_steps; ++step) {
    float loss = 0.0f;
    int rc = litert_lm_mezo_finetuner_step(
        finetuner, &mezo_param, 1, SyntheticBenchLossFn, &ctx, &loss);

    if (rc != 0) {
      fprintf(stderr, "  Step %d: MeZO step failed (rc=%d)\n", step, rc);
      break;
    }

    result.step_losses.push_back(loss);

    if (step % (config.eval_every / 4) == 0 && step < config.eval_every) {
      // More frequent early reporting.
      printf("  Step %5d | train_loss=%.6f\n", step, loss);
    } else if (step % config.eval_every == 0) {
      printf("  Step %5d | train_loss=%.6f\n", step, loss);
    }

    if (step % config.eval_every == 0 || step == config.num_steps) {
      eval_loss = SyntheticEvalLoss(params.data(), config.dim,
                                    eigenvalues.data(), optimum.data());
      printf("  Step %5d | eval_loss=%.6f\n", step, eval_loss);
      result.eval_losses.push_back(eval_loss);
      result.eval_steps.push_back(step);
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  result.total_time_sec =
      std::chrono::duration<double>(t_end - t_start).count();

  printf("  Done: %d steps in %.1f sec (%.2f ms/step)\n",
         config.num_steps, result.total_time_sec,
         result.total_time_sec / config.num_steps * 1000.0);

  // Compute distance to optimum.
  double dist_sq = 0.0;
  for (int i = 0; i < config.dim; ++i) {
    float d = params[i] - optimum[i];
    dist_sq += d * d;
  }
  printf("  ||w - w*|| = %.6f\n", std::sqrt(dist_sq));

  litert_lm_mezo_finetuner_delete(finetuner);
  return result;
}

// ---------------------------------------------------------------------------
// Model-based benchmark (requires LoRA model)
// ---------------------------------------------------------------------------
struct DataSample {
  std::string prompt;
  std::string target;
};

std::vector<DataSample> LoadWikiText(const std::string& path, int seq_len,
                                     int max_samples = 0) {
  std::ifstream file(path);
  if (!file.is_open()) {
    fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str());
    return {};
  }

  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());

  int chars_per_sample = seq_len * 4;
  int half = chars_per_sample / 2;

  std::vector<DataSample> samples;
  size_t pos = 0;
  while (pos + chars_per_sample <= text.size()) {
    while (pos < text.size() && (text[pos] == '\n' || text[pos] == '=')) {
      pos++;
    }
    if (pos + chars_per_sample > text.size()) break;

    DataSample sample;
    sample.prompt = text.substr(pos, half);
    sample.target = text.substr(pos + half, half);
    samples.push_back(std::move(sample));
    pos += chars_per_sample;

    if (max_samples > 0 && static_cast<int>(samples.size()) >= max_samples) {
      break;
    }
  }

  return samples;
}

struct ModelLossContext {
  LiteRtLmEngine* engine;
  LiteRtLmSessionConfig* session_config;
  const std::vector<DataSample>* train_data;
  int current_sample_idx;
};

int ModelLossFn(void* user_data, float* loss_out) {
  auto* ctx = static_cast<ModelLossContext*>(user_data);

  LiteRtLmSession* session =
      litert_lm_engine_create_session(ctx->engine, ctx->session_config);
  if (!session) return -1;

  const auto& sample =
      ctx->train_data->at(ctx->current_sample_idx % ctx->train_data->size());

  int rc = litert_lm_session_run_prefill(session, sample.prompt.c_str());
  if (rc != 0) {
    litert_lm_session_delete(session);
    return -1;
  }

  const char* target = sample.target.c_str();
  float score = 0.0f;
  int n = litert_lm_session_run_text_scoring(session, &target, 1, &score);
  litert_lm_session_delete(session);

  if (n < 1) return -1;
  *loss_out = -score;
  return 0;
}

RunResult RunModelOptimizer(const BenchConfig& config, const std::string& name,
                            int optimizer_mode, LiteRtLmEngine* engine,
                            LiteRtLmSessionConfig* session_config,
                            const std::vector<DataSample>& train_data) {
  RunResult result;
  result.optimizer_name = name;

  printf("\n=== %s (lr=%.1e, mode=%d) ===\n", name.c_str(), config.lr,
         optimizer_mode);

  LiteRtLmSession* param_session =
      litert_lm_engine_create_session(engine, session_config);
  if (!param_session) {
    fprintf(stderr, "ERROR: Cannot create session\n");
    return result;
  }

  LiteRtLmTrainableParams* params =
      litert_lm_session_get_trainable_parameters(param_session);
  if (!params) {
    fprintf(stderr,
            "ERROR: Cannot get trainable parameters. "
            "Model may not have LoRA signatures.\n");
    litert_lm_session_delete(param_session);
    return result;
  }

  size_t num_params = litert_lm_trainable_params_count(params);
  printf("  Trainable parameters: %zu tensors\n", num_params);

  std::vector<LiteRtLmMeZoParameter> mezo_params(num_params);
  size_t total_elements = 0;
  for (size_t i = 0; i < num_params; ++i) {
    const LiteRtLmMeZoParameter* p = litert_lm_trainable_params_get(params, i);
    mezo_params[i] = *p;
    total_elements += p->num_elements;
  }
  printf("  Total trainable elements: %zu (%.2f MB)\n", total_elements,
         total_elements * sizeof(float) / (1024.0 * 1024.0));

  LiteRtLmMeZoConfig* mezo_config = litert_lm_mezo_config_create();
  litert_lm_mezo_config_set_learning_rate(mezo_config, config.lr);
  litert_lm_mezo_config_set_epsilon(mezo_config, config.epsilon);
  litert_lm_mezo_config_set_seed(mezo_config, config.seed);
  litert_lm_mezo_config_set_optimizer_mode(mezo_config, optimizer_mode);
  if (optimizer_mode == 1) {
    litert_lm_mezo_config_set_momentum_decay(mezo_config, 0.9f);
    litert_lm_mezo_config_set_cone_angle(mezo_config, 0.7854f);
  }
  if (optimizer_mode == 2) {
    litert_lm_mezo_config_set_agzo_subspace_rank(mezo_config, config.agzo_rank);
  }

  LiteRtLmMeZoFineTuner* finetuner =
      litert_lm_mezo_finetuner_create(mezo_config);
  litert_lm_mezo_config_delete(mezo_config);

  if (!finetuner) {
    fprintf(stderr, "ERROR: Cannot create fine-tuner\n");
    litert_lm_trainable_params_delete(params);
    litert_lm_session_delete(param_session);
    return result;
  }

  ModelLossContext ctx;
  ctx.engine = engine;
  ctx.session_config = session_config;
  ctx.train_data = &train_data;
  ctx.current_sample_idx = 0;

  auto t_start = std::chrono::steady_clock::now();

  for (int step = 1; step <= config.num_steps; ++step) {
    ctx.current_sample_idx = step - 1;
    float loss = 0.0f;
    int rc = litert_lm_mezo_finetuner_step(finetuner, mezo_params.data(),
                                           num_params, ModelLossFn, &ctx,
                                           &loss);
    if (rc != 0) {
      fprintf(stderr, "  Step %d: MeZO step failed (rc=%d)\n", step, rc);
      break;
    }
    result.step_losses.push_back(loss);
    if (step % 10 == 0) {
      printf("  Step %5d | train_loss=%.4f\n", step, loss);
    }
    if (step % config.eval_every == 0 || step == config.num_steps) {
      result.eval_losses.push_back(loss);
      result.eval_steps.push_back(step);
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  result.total_time_sec =
      std::chrono::duration<double>(t_end - t_start).count();

  printf("  Done: %d steps in %.1f sec (%.2f sec/step)\n", config.num_steps,
         result.total_time_sec, result.total_time_sec / config.num_steps);

  litert_lm_mezo_finetuner_delete(finetuner);
  litert_lm_trainable_params_delete(params);
  litert_lm_session_delete(param_session);

  return result;
}

// ---------------------------------------------------------------------------
// Print comparison table
// ---------------------------------------------------------------------------
void PrintComparison(const std::vector<RunResult>& results,
                     bool is_synthetic) {
  printf("\n");
  printf("================================================================\n");
  if (is_synthetic) {
    printf("  MeZO Optimizer Benchmark (Synthetic Ill-conditioned Quadratic)\n");
  } else {
    printf("  MeZO WikiText-2 Benchmark Results\n");
  }
  printf("================================================================\n\n");

  printf("%-15s | %12s | %12s | %10s | %10s\n", "Optimizer", "Init Loss",
         "Final Loss", "Reduction", "Time (s)");
  printf("%-15s-+-%12s-+-%12s-+-%10s-+-%10s\n", "---------------",
         "------------", "------------", "----------", "----------");

  for (const auto& r : results) {
    if (r.eval_losses.size() < 2) {
      printf("%-15s | %12s | %12s | %10s | %10.1f\n",
             r.optimizer_name.c_str(), "N/A", "N/A", "N/A", r.total_time_sec);
      continue;
    }
    float init = r.eval_losses.front();
    float final_loss = r.eval_losses.back();
    float reduction = (init > 0.0f) ? (init - final_loss) / init * 100.0f
                                    : 0.0f;
    printf("%-15s | %12.6f | %12.6f | %9.2f%% | %10.1f\n",
           r.optimizer_name.c_str(), init, final_loss, reduction,
           r.total_time_sec);
  }

  // If synthetic, show step-losses at select checkpoints for convergence plot.
  if (is_synthetic && !results.empty() &&
      results[0].eval_losses.size() >= 2) {
    printf("\nConvergence trajectory (eval loss at checkpoints):\n");
    printf("%-7s", "Step");
    for (const auto& r : results) {
      printf(" | %15s", r.optimizer_name.c_str());
    }
    printf("\n");
    printf("%-7s", "-------");
    for (size_t i = 0; i < results.size(); ++i) {
      printf("-+-%15s", "---------------");
    }
    printf("\n");

    // Print all eval checkpoints (assuming same eval_steps across runs).
    size_t max_evals = 0;
    for (const auto& r : results) {
      max_evals = std::max(max_evals, r.eval_losses.size());
    }
    for (size_t e = 0; e < max_evals; ++e) {
      int step = (e < results[0].eval_steps.size())
                     ? results[0].eval_steps[e]
                     : -1;
      if (step >= 0) printf("%-7d", step);
      else printf("%-7s", "?");
      for (const auto& r : results) {
        if (e < r.eval_losses.size()) {
          printf(" | %15.6f", r.eval_losses[e]);
        } else {
          printf(" | %15s", "N/A");
        }
      }
      printf("\n");
    }
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
BenchConfig ParseArgs(int argc, char** argv) {
  BenchConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto eq = arg.find('=');
    if (eq == std::string::npos) continue;
    std::string key = arg.substr(0, eq);
    std::string val = arg.substr(eq + 1);
    if (key == "--model_path") config.model_path = val;
    else if (key == "--lora_path") config.lora_path = val;
    else if (key == "--backend") config.backend = val;
    else if (key == "--train_data") config.train_data_path = val;
    else if (key == "--eval_data") config.eval_data_path = val;
    else if (key == "--seq_len") config.seq_len = std::stoi(val);
    else if (key == "--num_steps") config.num_steps = std::stoi(val);
    else if (key == "--eval_every") config.eval_every = std::stoi(val);
    else if (key == "--lr") config.lr = std::stof(val);
    else if (key == "--epsilon") config.epsilon = std::stof(val);
    else if (key == "--seed") config.seed = std::stoull(val);
    else if (key == "--dim") config.dim = std::stoi(val);
    else if (key == "--noise") config.noise_std = std::stof(val);
    else if (key == "--condition") config.condition_number = std::stof(val);
    else if (key == "--agzo_rank") config.agzo_rank = std::stoi(val);
  }
  return config;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  BenchConfig config = ParseArgs(argc, argv);
  bool synthetic_mode = config.model_path.empty();

  printf("MeZO Optimizer Benchmark\n");
  printf("  Mode: %s\n", synthetic_mode ? "synthetic" : "model");
  printf("  Steps: %d\n", config.num_steps);
  printf("  Learning rate: %.1e\n", config.lr);
  printf("  Epsilon: %.1e\n", config.epsilon);
  printf("  Seed: %lu\n", config.seed);
  printf("  AGZO rank: %d\n", config.agzo_rank);

  std::vector<RunResult> results;

  if (synthetic_mode) {
    printf("  Dimension: %d\n", config.dim);
    printf("  Condition number: %.0f\n", config.condition_number);
    printf("  Noise std: %.4f\n", config.noise_std);
    printf("  Eval every: %d steps\n", config.eval_every);

    // Generate ill-conditioned quadratic: eigenvalues log-spaced from 1 to κ.
    std::vector<float> eigenvalues(config.dim);
    std::vector<float> optimum(config.dim);
    std::mt19937 setup_rng(config.seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);

    for (int i = 0; i < config.dim; ++i) {
      float t = static_cast<float>(i) / std::max(config.dim - 1, 1);
      eigenvalues[i] = std::pow(config.condition_number, t);
      optimum[i] = normal(setup_rng);  // Random target near zero.
    }

    printf("\n  Eigenvalue range: [%.1f, %.1f]\n", eigenvalues[0],
           eigenvalues[config.dim - 1]);

    // Initial loss at w=0.
    float init_loss = SyntheticEvalLoss(
        std::vector<float>(config.dim, 0.0f).data(), config.dim,
        eigenvalues.data(), optimum.data());
    printf("  Initial loss at w=0: %.6f\n", init_loss);

    // Run each optimizer with the same starting point.
    results.push_back(RunSyntheticOptimizer(config, "Vanilla MeZO",
                                            /*mode=*/0, eigenvalues, optimum));
    results.push_back(RunSyntheticOptimizer(config, "ConMeZO",
                                            /*mode=*/1, eigenvalues, optimum));
    results.push_back(RunSyntheticOptimizer(config, "AGZO (k=" +
                                            std::to_string(config.agzo_rank) +
                                            ")",
                                            /*mode=*/2, eigenvalues, optimum));

    PrintComparison(results, /*is_synthetic=*/true);

  } else {
    // Model mode.
    printf("  Model: %s\n", config.model_path.c_str());
    printf("  Backend: %s\n", config.backend.c_str());

    printf("\nLoading WikiText-2...\n");
    auto train_data =
        LoadWikiText(config.train_data_path, config.seq_len, 2048);
    printf("  Train samples: %zu\n", train_data.size());

    if (train_data.empty()) {
      fprintf(stderr, "ERROR: No training data loaded\n");
      return 1;
    }

    printf("\nCreating engine...\n");
    LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(
        config.model_path.c_str(), config.backend.c_str(), nullptr, nullptr);
    if (!settings) {
      fprintf(stderr, "ERROR: Failed to create engine settings\n");
      return 1;
    }

    LiteRtLmEngine* engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!engine) {
      fprintf(stderr, "ERROR: Failed to create engine\n");
      return 1;
    }
    printf("  Engine created successfully\n");

    LiteRtLmSessionConfig* session_config = litert_lm_session_config_create();
    litert_lm_session_config_set_lora_id(session_config, 0);

    results.push_back(RunModelOptimizer(config, "Vanilla MeZO", 0, engine,
                                        session_config, train_data));
    results.push_back(RunModelOptimizer(config, "ConMeZO", 1, engine,
                                        session_config, train_data));
    results.push_back(RunModelOptimizer(config, "AGZO", 2, engine,
                                        session_config, train_data));

    PrintComparison(results, /*is_synthetic=*/false);

    litert_lm_session_config_delete(session_config);
    litert_lm_engine_delete(engine);
  }

  return 0;
}
