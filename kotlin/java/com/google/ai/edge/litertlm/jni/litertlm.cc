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

#include <jni.h>
#include <sys/stat.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "tflite/logger.h"  // from @litert
#include "tflite/minimal_logging.h"  // from @litert

// For Windows, __declspec( dllexport ) is required to export function in .dll.
// https://learn.microsoft.com/en-us/cpp/cpp/using-dllimport-and-dllexport-in-cpp-classes?view=msvc-170
//
// _WIN32 is defined as 1 when the compilation target is 32-bit ARM, 64-bit ARM,
// x86, x64, or ARM64EC. Otherwise, undefined.
// https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#if defined(_WIN32)
#define LITERTLM_JNIEXPORT __declspec(dllexport)
#else
#define LITERTLM_JNIEXPORT JNIEXPORT
#endif  // _WIN32

#define JNI_METHOD(METHOD_NAME) \
  Java_com_google_ai_edge_litertlm_LiteRtLmJni_##METHOD_NAME

namespace {
using litert::lm::Backend;
using litert::lm::Conversation;
using litert::lm::ConversationConfig;
using litert::lm::Engine;
using litert::lm::EngineFactory;
using litert::lm::EngineSettings;
using litert::lm::InputAudio;
using litert::lm::InputData;
using litert::lm::InputImage;
using litert::lm::InputText;
using litert::lm::JsonMessage;
using litert::lm::JsonPreface;
using litert::lm::Message;
using litert::lm::ModelAssets;
using litert::lm::Preface;
using litert::lm::Responses;
using litert::lm::SessionConfig;
using litert::lm::proto::SamplerParameters;

void ThrowLiteRtLmJniException(JNIEnv* env, const std::string& message) {
  jclass exClass =
      env->FindClass("com/google/ai/edge/litertlm/LiteRtLmJniException");
  if (exClass != nullptr) {
    env->ThrowNew(exClass, message.c_str());
    // Clean up local reference
    env->DeleteLocalRef(exClass);
  }
}

// Replacement of env->NewStringUTF(str.c_str()) to handle "Standard UTF-8".
//
// NewStringUTF() expects a "modified UTF-8" string. "Standard UTF-8" and
// "modified UTF-8" are mostly the same, but differ in the encoding of null
// characters and characters outside the Basic Multilingual Plane (BMP). Emojis
// often fall into this latter category. nlohmann::json::dump() also returns a
// "Standard UTF-8".
//
// https://developer.android.com/ndk/guides/jni-tips#utf-8-and-utf-16-strings
jstring NewStringStandardUTF(JNIEnv* env, std::string standard_utf8_str) {
  // Create a jbyteArray from the UTF-8 string
  jbyteArray bytes = env->NewByteArray(standard_utf8_str.length());
  env->SetByteArrayRegion(
      bytes, 0, standard_utf8_str.length(),
      reinterpret_cast<const jbyte*>(standard_utf8_str.c_str()));

  // Get the java.lang.String class
  jclass string_class = env->FindClass("java/lang/String");

  // Get the constructor for String(byte[], String)
  jmethodID string_ctor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");

  // Create a jstring for the charset name "UTF-8"
  jstring charset_name = env->NewStringUTF("UTF-8");

  // Create the new String object
  jstring result =
      (jstring)env->NewObject(string_class, string_ctor, bytes, charset_name);

  // Clean up local references
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(string_class);
  env->DeleteLocalRef(charset_name);

  return result;
}

// Helper function to convert BenchmarkInfo to Java object
jobject CreateBenchmarkInfoJni(
    JNIEnv* env, const litert::lm::BenchmarkInfo& benchmark_info) {
  int last_prefill_token_count = 0;
  if (benchmark_info.GetTotalPrefillTurns() > 0) {
    auto turn = benchmark_info.GetPrefillTurn(
        benchmark_info.GetTotalPrefillTurns() - 1);
    if (turn.ok()) {
      last_prefill_token_count = turn->num_tokens;
    }
  }

  int last_decode_token_count = 0;
  if (benchmark_info.GetTotalDecodeTurns() > 0) {
    auto turn =
        benchmark_info.GetDecodeTurn(benchmark_info.GetTotalDecodeTurns() - 1);
    if (turn.ok()) {
      last_decode_token_count = turn->num_tokens;
    }
  }

  jclass benchmark_info_cls =
      env->FindClass("com/google/ai/edge/litertlm/BenchmarkInfo");
  jmethodID benchmark_info_ctor =
      env->GetMethodID(benchmark_info_cls, "<init>", "(II)V");

  return env->NewObject(benchmark_info_cls, benchmark_info_ctor,
                        last_prefill_token_count, last_decode_token_count);
}

// Converts a Java InputData array to a C++ vector of InputData.
std::vector<InputData> GetNativeInputData(JNIEnv* env,
                                          jobjectArray input_data) {
  jclass text_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Text");
  jclass audio_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Audio");
  jclass image_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Image");

  jmethodID text_get_text_mid =
      env->GetMethodID(text_class, "getText", "()Ljava/lang/String;");
  jmethodID audio_get_bytes_mid =
      env->GetMethodID(audio_class, "getBytes", "()[B");
  jmethodID image_get_bytes_mid =
      env->GetMethodID(image_class, "getBytes", "()[B");

  jsize num_inputs = env->GetArrayLength(input_data);
  std::vector<InputData> contents;
  contents.reserve(num_inputs);
  for (jsize i = 0; i < num_inputs; ++i) {
    jobject input_obj = env->GetObjectArrayElement(input_data, i);
    if (env->IsInstanceOf(input_obj, text_class)) {
      jstring text_jstr =
          (jstring)env->CallObjectMethod(input_obj, text_get_text_mid);
      const char* text_chars = env->GetStringUTFChars(text_jstr, nullptr);
      contents.emplace_back(InputText(text_chars));
      env->ReleaseStringUTFChars(text_jstr, text_chars);
      env->DeleteLocalRef(text_jstr);
    } else if (env->IsInstanceOf(input_obj, audio_class)) {
      jbyteArray bytes_jarr =
          (jbyteArray)env->CallObjectMethod(input_obj, audio_get_bytes_mid);
      jsize len = env->GetArrayLength(bytes_jarr);
      jbyte* bytes = env->GetByteArrayElements(bytes_jarr, nullptr);
      contents.emplace_back(
          InputAudio(std::string(reinterpret_cast<char*>(bytes), len)));
      env->ReleaseByteArrayElements(bytes_jarr, bytes, JNI_ABORT);
      env->DeleteLocalRef(bytes_jarr);
    } else if (env->IsInstanceOf(input_obj, image_class)) {
      jbyteArray bytes_jarr =
          (jbyteArray)env->CallObjectMethod(input_obj, image_get_bytes_mid);
      jsize len = env->GetArrayLength(bytes_jarr);
      jbyte* bytes = env->GetByteArrayElements(bytes_jarr, nullptr);
      contents.emplace_back(
          InputImage(std::string(reinterpret_cast<char*>(bytes), len)));
      env->ReleaseByteArrayElements(bytes_jarr, bytes, JNI_ABORT);
      env->DeleteLocalRef(bytes_jarr);
    } else {
      ThrowLiteRtLmJniException(env, "Unsupported InputData type");
    }
    env->DeleteLocalRef(input_obj);
  }

  env->DeleteLocalRef(text_class);
  env->DeleteLocalRef(audio_class);
  env->DeleteLocalRef(image_class);

  return contents;
}

// Helper to get JNIEnv and attach to the current thread if necessary.
// Returns nullptr if an error occurs.
JNIEnv* GetJniEnvAndAttach(JavaVM* jvm, bool* attached) {
  JNIEnv* env = nullptr;
  *attached = false;
  // Requesting JNI_VERSION_1_6, but the returned JNIEnv* will support
  // the highest version the current JVM provides. This is safe because
  // newer JNI versions are backward-compatible.
  int get_env_stat = jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
  if (get_env_stat == JNI_EDETACHED) {
#if defined(__ANDROID__)
    if (jvm->AttachCurrentThread(&env, nullptr) == 0) {
#else
    if (jvm->AttachCurrentThread((void**)&env, nullptr) == 0) {
#endif
      *attached = true;
      return env;
    } else {
      ABSL_LOG(ERROR) << "Failed to attach to JVM.";
      return nullptr;
    }
  } else if (get_env_stat == JNI_OK) {
    return env;
  } else {
    ABSL_LOG(ERROR) << "Failed to get JNIEnv: GetEnv returned " << get_env_stat;
    return nullptr;
  }
}

// Helper function to create SamplerParameters from Java SamplerConfig object.
SamplerParameters CreateSamplerParamsFromJni(JNIEnv* env,
                                             jobject sampler_config_obj) {
  SamplerParameters sampler_params;

  // Based on the current engine implementation, when the SamplerConfig is
  // set, we must switch to the TOP_P sampling type.
  sampler_params.set_type(SamplerParameters::TOP_P);

  // Get the fields from SamplerConfig
  jclass sampler_config_cls = env->GetObjectClass(sampler_config_obj);

  jmethodID get_top_k_mid =
      env->GetMethodID(sampler_config_cls, "getTopK", "()I");
  sampler_params.set_k(env->CallIntMethod(sampler_config_obj, get_top_k_mid));

  jmethodID get_top_p_mid =
      env->GetMethodID(sampler_config_cls, "getTopP", "()D");
  sampler_params.set_p(
      env->CallDoubleMethod(sampler_config_obj, get_top_p_mid));

  jmethodID get_temperature_mid =
      env->GetMethodID(sampler_config_cls, "getTemperature", "()D");
  sampler_params.set_temperature(
      env->CallDoubleMethod(sampler_config_obj, get_temperature_mid));

  jmethodID get_seed_mid =
      env->GetMethodID(sampler_config_cls, "getSeed", "()I");
  sampler_params.set_seed(env->CallIntMethod(sampler_config_obj, get_seed_mid));

  env->DeleteLocalRef(sampler_config_cls);

  return sampler_params;
}
}  // namespace

extern "C" {

LITERTLM_JNIEXPORT void JNICALL
Java_com_google_ai_edge_litertlm_NativeLibraryLoader_nativeCheckLoaded(
    JNIEnv* env, jclass thiz) {}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeSetMinLogSeverity)(
    JNIEnv* env, jclass thiz, jint log_severity) {
  absl::LogSeverityAtLeast absl_log_severity;
  LiteRtLogSeverity litert_log_severity;
  tflite::LogSeverity tflite_log_severity;

  switch (log_severity) {
    case 0:  // verbose
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityVerbose;
      tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
      break;
    case 1:  // debug
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityDebug;
      tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
      break;
    case 2:  // info
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityInfo;
      tflite_log_severity = tflite::TFLITE_LOG_INFO;
      break;
    case 3:  // warning
      absl_log_severity = absl::LogSeverityAtLeast::kWarning;
      litert_log_severity = kLiteRtLogSeverityWarning;
      tflite_log_severity = tflite::TFLITE_LOG_WARNING;
      break;
    case 4:  // error
      absl_log_severity = absl::LogSeverityAtLeast::kError;
      litert_log_severity = kLiteRtLogSeverityError;
      tflite_log_severity = tflite::TFLITE_LOG_ERROR;
      break;
    case 5:  // fatal
      absl_log_severity = absl::LogSeverityAtLeast::kFatal;
      litert_log_severity = kLiteRtLogSeverityError;
      tflite_log_severity = tflite::TFLITE_LOG_ERROR;
      break;
    default:  // infinity
      absl_log_severity = absl::LogSeverityAtLeast::kInfinity;
      litert_log_severity = kLiteRtLogSeveritySilent;
      tflite_log_severity = tflite::TFLITE_LOG_SILENT;
      break;
  }

  // Update the absl logging framework, used by LiteRT-LM.
  absl::SetMinLogLevel(absl_log_severity);

  // Update the logging framework of LiteRT.
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), litert_log_severity);

  // Update the logging framework of TFLite.
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
      tflite_log_severity);
}

// __declspec( dllexport )
LITERTLM_JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateEngine)(
    JNIEnv* env, jclass thiz, jstring model_path, jstring backend,
    jstring vision_backend, jstring audio_backend, jint max_num_tokens,
    jstring cache_dir, jboolean enable_benchmark, jstring npu_libraries_dir) {
  const char* model_path_chars = env->GetStringUTFChars(model_path, nullptr);
  std::string model_path_str(model_path_chars);
  env->ReleaseStringUTFChars(model_path, model_path_chars);

  // Check if the file exists.
  struct stat buffer;
  if (stat(model_path_str.c_str(), &buffer) != 0) {
    ThrowLiteRtLmJniException(env, "Model file not found: " + model_path_str);
    return 0;
  }

  auto model_assets = ModelAssets::Create(model_path_str);
  if (!model_assets.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create model assets: " +
                                       model_assets.status().ToString());
    return 0;
  }

  const char* backend_chars = env->GetStringUTFChars(backend, nullptr);
  std::string backend_str(backend_chars);
  env->ReleaseStringUTFChars(backend, backend_chars);

  auto backend_enum = litert::lm::GetBackendFromString(backend_str);
  if (!backend_enum.ok()) {
    ThrowLiteRtLmJniException(env, backend_enum.status().ToString());
    return 0;
  }

  const char* vision_backend_chars =
      env->GetStringUTFChars(vision_backend, nullptr);
  std::string vision_backend_str(vision_backend_chars);
  env->ReleaseStringUTFChars(vision_backend, vision_backend_chars);

  std::optional<Backend> vision_backend_optional = std::nullopt;
  if (!vision_backend_str.empty()) {
    auto vision_backend_enum =
        litert::lm::GetBackendFromString(vision_backend_str);
    if (!vision_backend_enum.ok()) {
      ThrowLiteRtLmJniException(env, vision_backend_enum.status().ToString());
      return 0;
    }

    vision_backend_optional = vision_backend_enum.value();
  }

  const char* audio_backend_chars =
      env->GetStringUTFChars(audio_backend, nullptr);
  std::string audio_backend_str(audio_backend_chars);
  env->ReleaseStringUTFChars(audio_backend, audio_backend_chars);

  std::optional<Backend> audio_backend_optional = std::nullopt;
  if (!audio_backend_str.empty()) {
    auto audio_backend_enum =
        litert::lm::GetBackendFromString(audio_backend_str);
    if (!audio_backend_enum.ok()) {
      ThrowLiteRtLmJniException(env, audio_backend_enum.status().ToString());
      return 0;
    }

    audio_backend_optional = audio_backend_enum.value();
  }

  auto settings = EngineSettings::CreateDefault(*model_assets, *backend_enum,
                                                vision_backend_optional,
                                                audio_backend_optional);
  if (!settings.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create engine settings: " +
                                       settings.status().ToString());
    return 0;
  }

  const char* cache_dir_chars = env->GetStringUTFChars(cache_dir, nullptr);
  std::string cache_dir_str(cache_dir_chars);
  env->ReleaseStringUTFChars(cache_dir, cache_dir_chars);
  if (!cache_dir_str.empty()) {
    settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir_str);
    if (vision_backend_optional.has_value()) {
      settings->GetMutableVisionExecutorSettings()->SetCacheDir(cache_dir_str);
    }
    if (audio_backend_optional.has_value()) {
      settings->GetMutableAudioExecutorSettings()->SetCacheDir(cache_dir_str);
    }
  }

  const char* npu_libraries_dir_chars =
      env->GetStringUTFChars(npu_libraries_dir, nullptr);
  std::string npu_libraries_dir_str(npu_libraries_dir_chars);
  env->ReleaseStringUTFChars(npu_libraries_dir, npu_libraries_dir_chars);
  if (!npu_libraries_dir_str.empty()) {
    settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
        npu_libraries_dir_str);
    if (vision_backend_optional.has_value()) {
      settings->GetMutableVisionExecutorSettings()->SetLitertDispatchLibDir(
          npu_libraries_dir_str);
    }
    if (audio_backend_optional.has_value()) {
      settings->GetMutableAudioExecutorSettings()->SetLitertDispatchLibDir(
          npu_libraries_dir_str);
    }
  }

  if (max_num_tokens > 0) {
    settings->GetMutableMainExecutorSettings().SetMaxNumTokens(max_num_tokens);
  }

  if (enable_benchmark) {
    settings->GetMutableBenchmarkParams();
  }

  auto engine = EngineFactory::CreateAny(*settings);
  if (!engine.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to create engine: " + engine.status().ToString());
    return 0;
  }

  return reinterpret_cast<jlong>(engine->release());
}

LITERTLM_JNIEXPORT void JNICALL
JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz, jlong engine_pointer) {
  delete reinterpret_cast<Engine*>(engine_pointer);
}

LITERTLM_JNIEXPORT jlong JNICALL
JNI_METHOD(nativeCreateSession)(JNIEnv* env, jclass thiz, jlong engine_pointer,
                                jobject sampler_config_obj) {
  auto session_config = SessionConfig::CreateDefault();

  if (sampler_config_obj != nullptr) {
    session_config.GetMutableSamplerParams() =
        CreateSamplerParamsFromJni(env, sampler_config_obj);
  }

  Engine* engine = reinterpret_cast<Engine*>(engine_pointer);
  auto session = engine->CreateSession(session_config);
  if (!session.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to create session: " + session.status().ToString());
    return 0;
  }
  return reinterpret_cast<jlong>(session->release());
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSession)(
    JNIEnv* env, jclass thiz, jlong session_pointer) {
  delete reinterpret_cast<Engine::Session*>(session_pointer);
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeRunPrefill)(
    JNIEnv* env, jclass thiz, jlong session_pointer, jobjectArray input_data) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  // return if there is pending exceptions (e.g., if ThrowLiteRtLmJniException
  // called.)
  if (env->ExceptionCheck()) {
    return;
  }

  auto status = session->RunPrefill(contents);

  if (!status.ok()) {
    ThrowLiteRtLmJniException(env,
                              "Failed to run prefill: " + status.ToString());
  }
}

LITERTLM_JNIEXPORT jstring JNICALL
JNI_METHOD(nativeRunDecode)(JNIEnv* env, jclass thiz, jlong session_pointer) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  auto responses = session->RunDecode();

  if (!responses.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to run decode: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetTexts().size() != 1) {
    ThrowLiteRtLmJniException(
        env, "Number of output candidates should be 1, but got " +
                 std::to_string(responses->GetTexts().size()));
  }

  return NewStringStandardUTF(env, responses->GetTexts()[0]);
}

LITERTLM_JNIEXPORT jstring JNICALL JNI_METHOD(nativeGenerateContent)(
    JNIEnv* env, jclass thiz, jlong session_pointer, jobjectArray input_data) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  // return if there is pending exceptions (e.g., if ThrowLiteRtLmJniException
  // called.)
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  auto responses = session->GenerateContent(contents);

  if (!responses.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to generate content: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetTexts().empty()) {
    return env->NewStringUTF("");
  }

  return NewStringStandardUTF(env, responses->GetTexts()[0]);
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeGenerateContentStream)(
    JNIEnv* env, jclass thiz, jlong session_pointer, jobjectArray input_data,
    jobject callback) {
  JavaVM* jvm = nullptr;
  if (env->GetJavaVM(&jvm) != JNI_OK) {
    ThrowLiteRtLmJniException(env, "Failed to get JavaVM");
    return;
  }

  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  if (env->ExceptionCheck()) {
    return;
  }

  jobject callback_global = env->NewGlobalRef(callback);
  jclass callback_class = env->GetObjectClass(callback_global);
  jmethodID on_response_mid =
      env->GetMethodID(callback_class, "onNext", "(Ljava/lang/String;)V");
  jmethodID on_done_mid = env->GetMethodID(callback_class, "onDone", "()V");
  jmethodID on_error_mid =
      env->GetMethodID(callback_class, "onError", "(ILjava/lang/String;)V");
  env->DeleteLocalRef(callback_class);

  // This lambda is to clean up the global reference.
  absl::AnyInvocable<void()> cleanup_callback_ref = [jvm, callback_global]() {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(jvm, &attached);
    if (env) {
      env->DeleteGlobalRef(callback_global);
      // Detachment will be handled in the main callback_fn.
    }
  };

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback_fn =
      [jvm, callback_global, on_response_mid, on_done_mid, on_error_mid,
       cleanup_callback_ref = std::move(cleanup_callback_ref)](
          absl::StatusOr<Responses> responses) mutable {
        bool attached = false;
        JNIEnv* env = GetJniEnvAndAttach(jvm, &attached);
        if (!env) return;

        if (responses.ok()) {
          if (responses->GetTaskState() == litert::lm::TaskState::kDone) {
            env->CallVoidMethod(callback_global, on_done_mid);
            cleanup_callback_ref();
          } else if (responses->GetTaskState() ==
                     litert::lm::TaskState::kMaxNumTokensReached) {
            jstring message =
                NewStringStandardUTF(env, "Maximum kv-cache size reached.");
            env->CallVoidMethod(callback_global, on_error_mid,
                                (jint)absl::StatusCode::kInternal, message);
            env->DeleteLocalRef(message);
            cleanup_callback_ref();
          } else {
            jstring response_jstr =
                NewStringStandardUTF(env, responses->GetTexts()[0]);
            env->CallVoidMethod(callback_global, on_response_mid,
                                response_jstr);
            env->DeleteLocalRef(response_jstr);
          }
        } else {
          ABSL_LOG(WARNING)
              << "Receive callback OnError: " << responses.status();
          jstring message = NewStringStandardUTF(
              env, std::string(responses.status().message()));
          env->CallVoidMethod(callback_global, on_error_mid,
                              (jint)responses.status().code(), message);
          env->DeleteLocalRef(message);
          cleanup_callback_ref();
        }

        if (attached && jvm->DetachCurrentThread() != JNI_OK) {
          ABSL_LOG(ERROR) << "Failed to detach from JVM in callback_fn.";
        }
      };

  auto status =
      session->GenerateContentStream(contents, std::move(callback_fn));

  if (!status.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to start GenerateContentStream: " + status.ToString());
  }
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeCancelProcess)(
    JNIEnv* env, jclass thiz, jlong session_pointer) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);
  session->CancelProcess();
}

LITERTLM_JNIEXPORT jobject JNICALL
JNI_METHOD(nativeConversationGetBenchmarkInfo)(JNIEnv* env, jclass thiz,
                                               jlong conversation_pointer) {
  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);

  auto benchmark_info = conversation->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to get benchmark info: " +
                                       benchmark_info.status().ToString());
    return nullptr;
  }

  return CreateBenchmarkInfoJni(env, *benchmark_info);
}

LITERTLM_JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateConversation)(
    JNIEnv* env, jclass thiz, jlong engine_pointer, jobject sampler_config_obj,
    jstring messages_json_string, jstring tools_description_json_string,
    jboolean enable_constrained_decoding) {
  Engine* engine = reinterpret_cast<Engine*>(engine_pointer);

  // Create a native SessionConfig
  auto session_config = SessionConfig::CreateDefault();
  if (sampler_config_obj != nullptr) {
    session_config.GetMutableSamplerParams() =
        CreateSamplerParamsFromJni(env, sampler_config_obj);
  }

  // Create the Preface from the system instruction and tools.
  JsonPreface json_preface;

  const char* messages_chars =
      env->GetStringUTFChars(messages_json_string, nullptr);
  std::string messages_json_str(messages_chars);
  env->ReleaseStringUTFChars(messages_json_string, messages_chars);
  json_preface.messages = nlohmann::ordered_json::parse(messages_json_str);

  const char* tools_description_chars =
      env->GetStringUTFChars(tools_description_json_string, nullptr);
  auto tool_json = nlohmann::ordered_json::parse(tools_description_chars);
  env->ReleaseStringUTFChars(tools_description_json_string,
                             tools_description_chars);

  if (tool_json.is_array()) {
    nlohmann::ordered_json::array_t tool_json_array =
        tool_json.get<nlohmann::ordered_json::array_t>();
    json_preface.tools = tool_json_array;
  } else {
    ThrowLiteRtLmJniException(
        env, "tools_json should be a json array. Got: " + tool_json.dump());
    return 0;
  }

  // Create the conversation
  auto conversation_config =
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetPreface(json_preface)
          .SetEnableConstrainedDecoding(enable_constrained_decoding)
          .Build(*engine);

  if (!conversation_config.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create conversation config: " +
                                       conversation_config.status().ToString());
    return 0;
  }
  auto conversation = Conversation::Create(*engine, *conversation_config);
  if (!conversation.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create conversation: " +
                                       conversation.status().ToString());
    return 0;
  }

  return reinterpret_cast<jlong>(conversation->release());
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteConversation)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer) {
  delete reinterpret_cast<Conversation*>(conversation_pointer);
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeSendMessageAsync)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer,
    jstring messageJSONString, jobject callback) {
  JavaVM* jvm = nullptr;
  if (env->GetJavaVM(&jvm) != JNI_OK) {
    ThrowLiteRtLmJniException(env, "Failed to get JavaVM");
    return;
  }

  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);

  const char* json_chars = env->GetStringUTFChars(messageJSONString, nullptr);
  litert::lm::JsonMessage json_message =
      nlohmann::ordered_json::parse(json_chars);
  env->ReleaseStringUTFChars(messageJSONString, json_chars);

  jobject callback_global = env->NewGlobalRef(callback);
  jclass callback_class = env->GetObjectClass(callback_global);
  jmethodID on_message_mid =
      env->GetMethodID(callback_class, "onMessage", "(Ljava/lang/String;)V");
  jmethodID on_complete_mid = env->GetMethodID(callback_class, "onDone", "()V");
  jmethodID on_error_mid =
      env->GetMethodID(callback_class, "onError", "(ILjava/lang/String;)V");
  env->DeleteLocalRef(callback_class);

  absl::AnyInvocable<void(absl::StatusOr<Message>)> callback_fn =
      [jvm, callback_global, on_message_mid, on_complete_mid,
       on_error_mid](absl::StatusOr<Message> message) {
        bool attached = false;
        JNIEnv* env = GetJniEnvAndAttach(jvm, &attached);
        if (!env) return;

        // This lambda is to clean up the global reference.
        auto on_done_fn = [jvm, callback_global]() {
          bool attached = false;
          JNIEnv* env = GetJniEnvAndAttach(jvm, &attached);
          if (env) {
            env->DeleteGlobalRef(callback_global);
            if (attached && jvm->DetachCurrentThread() != JNI_OK) {
              ABSL_LOG(ERROR) << "Failed to detach from JVM in on_done_fn.";
            }
          }
        };

        if (message.ok()) {
          if (!std::holds_alternative<litert::lm::JsonMessage>(*message)) {
            ABSL_LOG(WARNING) << "Receive callback OnError: Not a JsonMessage";
            jstring err_message =
                env->NewStringUTF("Response is not a JsonMessage");
            env->CallVoidMethod(callback_global, on_error_mid,
                                (jint)absl::StatusCode::kInternal, err_message);
            env->DeleteLocalRef(err_message);
            on_done_fn();
          } else {
            auto json_message = std::get<litert::lm::JsonMessage>(*message);
            if (json_message.is_null()) {
              // Null message indicates completion.
              env->CallVoidMethod(callback_global, on_complete_mid);
              on_done_fn();
            } else {
              std::string message_str = json_message.dump();
              jstring message_jstr = NewStringStandardUTF(env, message_str);
              env->CallVoidMethod(callback_global, on_message_mid,
                                  message_jstr);
              env->DeleteLocalRef(message_jstr);
            }
          }
        } else {
          ABSL_LOG(WARNING) << "Receive callback OnError: " << message.status();
          jstring err_message =
              NewStringStandardUTF(env, message.status().message().data());
          env->CallVoidMethod(callback_global, on_error_mid,
                              (jint)message.status().code(), err_message);
          env->DeleteLocalRef(err_message);
          on_done_fn();
        }

        if (attached && jvm->DetachCurrentThread() != JNI_OK) {
          ABSL_LOG(ERROR) << "Failed to detach from JVM in callback_fn.";
        }
      };

  auto status =
      conversation->SendMessageAsync(json_message, std::move(callback_fn));

  if (!status.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to start nativeSendMessageAsync: " + status.ToString());
  }
}

LITERTLM_JNIEXPORT jstring JNICALL JNI_METHOD(nativeSendMessage)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer,
    jstring messageJSONString) {
  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);

  const char* json_chars = env->GetStringUTFChars(messageJSONString, nullptr);
  litert::lm::JsonMessage json_message =
      nlohmann::ordered_json::parse(json_chars);
  env->ReleaseStringUTFChars(messageJSONString, json_chars);

  auto response = conversation->SendMessage(json_message);
  if (!response.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to call nativeSendMessage: " +
                                       response.status().ToString());
    return nullptr;
  }

  if (!std::holds_alternative<litert::lm::JsonMessage>(*response)) {
    ThrowLiteRtLmJniException(
        env, "Failed to call nativeSendMessage: Response is not a JsonMessage");
    return nullptr;
  }

  auto json_response = std::get<litert::lm::JsonMessage>(*response);
  return NewStringStandardUTF(env, json_response.dump());
}

LITERTLM_JNIEXPORT void JNICALL JNI_METHOD(nativeConversationCancelProcess)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer) {
  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);
  conversation->CancelProcess();
}

}  // extern "C"
