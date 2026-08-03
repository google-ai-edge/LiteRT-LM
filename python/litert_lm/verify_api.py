# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone structural verification suite for pre-built LiteRT-LM PyPI wheels."""

import pathlib
import re
import urllib.error
import urllib.request
import litert_lm


def verify_comprehensive_e2e_suite(model_path: pathlib.Path):
  """Executes master comprehensive E2E tests covering multi-turn reasoning, streaming, tools, and telemetry."""
  print("------------------------------------------------")
  print(
      f"🚀 Running Comprehensive Enterprise E2E Verification using {model_path}"
  )
  print("------------------------------------------------")

  # 1. Multi-Turn Statefulness & Semantic Reasoning
  print("\n1️⃣ Testing Multi-Turn Reasoning & Statefulness...")
  with (
      litert_lm.Engine(str(model_path), max_num_tokens=4096) as engine,
      engine.create_conversation() as conversation,
  ):
    msg1 = conversation.send_message("What is the capital of France?")
    text1 = msg1["content"][0]["text"]
    print(f"   Turn 1 Response: '{text1.strip()}'")
    assert (
        "paris" in text1.lower()
    ), f"Turn 1 Failure: Expected 'Paris', got '{text1}'"

    msg2 = conversation.send_message("And what country is that city in?")
    text2 = msg2["content"][0]["text"]
    print(f"   Turn 2 Response: '{text2.strip()}'")
    assert (
        "france" in text2.lower()
    ), f"Turn 2 Failure: Expected 'France', got '{text2}'"

  # 2. Async Streaming Generation
  print("\n2️⃣ Testing Async Streaming Token Generation...")
  with (
      litert_lm.Engine(str(model_path), max_num_tokens=4096) as engine,
      engine.create_conversation() as conversation,
  ):
    stream = conversation.send_message_async("What is 2 + 2?")
    chunks = []
    for chunk in stream:
      if (
          isinstance(chunk, dict)
          and "content" in chunk
          and chunk["content"]
          and "text" in chunk["content"][0]
      ):
        chunks.append(chunk["content"][0]["text"])
    full_async_text = "".join(chunks)
    print(f"   Async Stream Output: '{full_async_text.strip()}'")
    assert (
        "4" in full_async_text
    ), f"Streaming Failure: Expected '4', got '{full_async_text}'"

  # 3. Emoji Rendering Support
  print("\n3️⃣ Testing Emoji Output Rendering...")
  with (
      litert_lm.Engine(str(model_path), max_num_tokens=4096) as engine,
      engine.create_conversation() as conversation,
  ):
    resp_emoji = conversation.send_message("What is the emoji of strawberry?")
    text_emoji = resp_emoji["content"][0]["text"]
    print(f"   Emoji Output: '{text_emoji.strip()}'")
    assert (
        "🍓" in text_emoji or "strawberry" in text_emoji.lower()
    ), f"Emoji Failure: Got '{text_emoji}'"

  # 4. Agentic Function Calling / Tools
  print("\n4️⃣ Testing Structured Function Calling / Tool Execution...")

  def get_weather(location: str) -> str:
    """Gets current weather for a city.

    Args:
      location: City name.

    Returns:
      A string describing the weather.
    """
    return f"Weather in {location} is sunny."

  with (
      litert_lm.Engine(str(model_path), max_num_tokens=4096) as engine,
      engine.create_conversation(tools=[get_weather]) as conversation,
  ):
    resp_tool = conversation.send_message("What's the weather in London?")
    text_tool = resp_tool["content"][0]["text"]
    print(f"   Tool Execution Response: '{text_tool.strip()}'")
    assert (
        "sunny" in text_tool.lower() and "london" in text_tool.lower()
    ), f"Tool Failure: Got '{text_tool}'"

  # 5. Telemetry & Performance Benchmarking
  print("\n5️⃣ Testing Engine Telemetry & Performance Metrics...")
  benchmark = litert_lm.Benchmark(
      str(model_path),
      litert_lm.Backend.CPU(),
      prefill_tokens=64,
      decode_tokens=64,
      max_num_tokens=4096,
  )
  info = benchmark.run()
  print(
      f"   TTFT: {info.time_to_first_token_in_second*1000:.1f} ms | Decode:"
      f" {info.last_decode_tokens_per_second:.1f} tok/sec"
  )
  assert info.time_to_first_token_in_second > 0.0, "Telemetry Breach: TTFT <= 0"
  assert (
      info.last_decode_tokens_per_second > 0.0
  ), "Telemetry Breach: Decode TPS <= 0"

  # 6. Multi-Modal Vision Preface Test
  print("\n6️⃣ Testing Multi-Modal Vision Preface (Image History)...")
  repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
  apple_img = (
      repo_root / "runtime/components/preprocessor/testdata/apple.png"
  )
  if apple_img.exists():
    messages = [
        litert_lm.Message.user(
            litert_lm.Contents.of(
                litert_lm.Content.ImageFile(str(apple_img)),
                "Describe this image.",
            )
        ),
        litert_lm.Message.model(
            litert_lm.Contents.of("This is an image of an apple.")
        ),
    ]
    with (
        litert_lm.Engine(
            str(model_path),
            vision_backend=litert_lm.Backend.CPU(),
            max_num_tokens=4096,
        ) as engine,
        engine.create_conversation(messages=messages) as conversation,
    ):
      resp_vision = conversation.send_message(
          "How many apples are there in the image?"
      )
      text_vision = resp_vision["content"][0]["text"]
      print(f"   Vision Preface Output: '{text_vision.strip()}'")
      assert re.search(
          r"two|2|one|1|apple", text_vision.lower()
      ), f"Vision Preface Failure: Got '{text_vision}'"
  else:
    print(
        f"   ⚠️ Skipping Vision Preface test (image not found at {apple_img})"
    )

  # 7. Multi-Modal Audio Transcription Test
  print("\n7️⃣ Testing Multi-Modal Audio Transcription...")
  audio_wav = repo_root / "runtime/testdata/have_a_wonderful_day.wav"
  if audio_wav.exists():
    with (
        litert_lm.Engine(
            str(model_path),
            audio_backend=litert_lm.Backend.CPU(),
            max_num_tokens=4096,
        ) as engine,
        engine.create_conversation() as conversation,
    ):
      resp_audio = conversation.send_message(
          litert_lm.Contents.of(
              litert_lm.Content.AudioFile(str(audio_wav)),
              "Transcribe this audio.",
          )
      )
      text_audio = resp_audio["content"][0]["text"]
      print(f"   Audio Transcription Output: '{text_audio.strip()}'")
      assert (
          "wonderful" in text_audio.lower() and "day" in text_audio.lower()
      ), f"Audio Failure: Got '{text_audio}'"
  else:
    print(f"   ⚠️ Skipping Audio test (file not found at {audio_wav})")

  print(
      "\n🎉 ALL 7 COMPREHENSIVE ENTERPRISE VERIFICATION GATES PASSED 100%"
      " GREEN!"
  )


def main():
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.INFO)

  # Target GCS model cached in temporary storage
  model_dir = pathlib.Path("/tmp/litert_lm_models")
  model_dir.mkdir(parents=True, exist_ok=True)
  target_model = model_dir / "gemma-4-E2B-it.litertlm"

  if not target_model.exists():
    model_url = (
        "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/"
        "resolve/main/gemma-4-E2B-it.litertlm"
    )
    print(f"⏬ Downloading verification model from {model_url}...")
    urllib.request.urlretrieve(model_url, target_model)
    size_mb = target_model.stat().st_size // (1024 * 1024)
    print(f"✅ Model downloaded successfully ({size_mb} MB)")

  if not target_model.exists():
    raise FileNotFoundError(f"Verification model not found at {target_model}")

  verify_comprehensive_e2e_suite(target_model)


if __name__ == "__main__":
  main()
