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

import pathlib
from unittest import mock
import warnings

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import litert_lm

FLAGS = flags.FLAGS


class LiteRtLmTestBase(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  def setUp(self):
    super().setUp()
    self.model_path = str(
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

  def _create_engine(self, max_num_tokens=10, enable_benchmark=False):
    return litert_lm.Engine(
        self.model_path,
        litert_lm.Backend.CPU(),
        max_num_tokens=max_num_tokens,
        cache_dir=":nocache",
        enable_benchmark=enable_benchmark,
    )

  @classmethod
  def _extract_text(cls, stream):
    text_pieces = []
    for chunk in stream:
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          text_pieces.append(item.get("text", ""))
    return text_pieces


class EngineTest(LiteRtLmTestBase):

  _EXPECTED_RESPONSE = "TarefaByte دارایेत्र investigaciónప్రదేశ"

  def test_engine_init_fail(self):
    with self.assertRaisesRegex(
        RuntimeError, "Failed to create LiteRT-LM engine for /non/existent/path"
    ):
      litert_lm.Engine("/non/existent/path")

  def test_backend_cpu_equality(self):
    cpu_default = litert_lm.Backend.CPU()
    cpu_default_explicit = litert_lm.Backend.CPU(thread_count=None)
    cpu_4 = litert_lm.Backend.CPU(thread_count=4)
    cpu_2 = litert_lm.Backend.CPU(thread_count=2)
    gpu = litert_lm.Backend.GPU()

    with self.subTest("CPU default equality"):
      self.assertEqual(cpu_default, cpu_default_explicit)
    with self.subTest("CPU default vs thread count inequality"):
      self.assertNotEqual(cpu_default, cpu_4)
    with self.subTest("Different CPU thread counts inequality"):
      self.assertNotEqual(cpu_4, cpu_2)
    with self.subTest("CPU vs GPU inequality"):
      self.assertNotEqual(cpu_4, gpu)

  def test_engine_init_with_cpu_thread_counts(self):
    lib = litert_lm._ffi._get_lib()
    orig_set_num_threads = lib.litert_lm_engine_settings_set_num_threads
    orig_set_audio_num_threads = (
        lib.litert_lm_engine_settings_set_audio_num_threads
    )

    mock_set_num_threads = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_num_threads",
            autospec=True,
            side_effect=orig_set_num_threads,
        )
    )
    mock_set_audio_num_threads = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_audio_num_threads",
            autospec=True,
            side_effect=orig_set_audio_num_threads,
        )
    )

    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(thread_count=4),
        audio_backend=litert_lm.Backend.CPU(thread_count=2),
        cache_dir=":nocache",
    )

    mock_set_num_threads.assert_called_once_with(mock.ANY, 4)
    mock_set_audio_num_threads.assert_called_once_with(mock.ANY, 2)

  @mock.patch("sys.platform", "win32")
  def test_engine_init_with_npu_backend(self):
    lib = litert_lm._ffi._get_lib()
    if hasattr(lib, "litert_lm_engine_settings_set_litert_dispatch_lib_dir"):
      orig_set_dir = lib.litert_lm_engine_settings_set_litert_dispatch_lib_dir

      mock_ov = mock.MagicMock()
      mock_ov.__file__ = "path/to/openvino/__init__.py"
      mock_ov.Core.return_value.available_devices = ["CPU", "NPU"]

      with mock.patch.object(
          lib,
          "litert_lm_engine_settings_set_litert_dispatch_lib_dir",
          autospec=True,
          side_effect=orig_set_dir,
      ) as mock_set_dir:
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
          with mock.patch("importlib.resources.files") as unused_mock_files:
            try:
              npu = litert_lm.Backend.NPU(
                  litert_dispatch_lib_dir="my_custom_dir"
              )
              litert_lm.Engine(
                  self.model_path,
                  npu,
                  cache_dir=":nocache",
              )
            except RuntimeError:
              pass

            mock_set_dir.assert_called_once_with(mock.ANY, "my_custom_dir")

  @mock.patch("sys.platform", "linux")
  def test_npu_backend_non_windows(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "NPU is supported only for Intel OpenVINO on Windows. Current"
        " platform is 'linux'.",
    ):
      litert_lm.Backend.NPU()

  @mock.patch("sys.platform", "win32")
  def test_npu_backend_windows_no_openvino(self):
    with mock.patch.dict("sys.modules", {"openvino": None}):
      with self.assertRaisesRegex(
          RuntimeError,
          "NPU is supported only for Intel OpenVINO on Windows. Failed to"
          " import the 'openvino' package. Please ensure 'openvino' is"
          " installed.",
      ):
        litert_lm.Backend.NPU()

  @mock.patch("sys.platform", "win32")
  def test_npu_backend_windows_openvino_no_npu(self):
    mock_ov = mock.MagicMock()
    mock_ov.Core.return_value.available_devices = ["CPU", "GPU"]
    with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
      with self.assertRaisesRegex(
          RuntimeError,
          "NPU is supported only for Intel OpenVINO on Windows. No NPU"
          r" device detected by OpenVINO \(available devices: \['CPU',"
          r" 'GPU'\]\).",
      ):
        litert_lm.Backend.NPU()

  @mock.patch("sys.platform", "win32")
  def test_npu_backend_windows_openvino_with_npu(self):
    mock_ov = mock.MagicMock()
    mock_ov.__file__ = "path/to/openvino/__init__.py"
    mock_ov.Core.return_value.available_devices = ["CPU", "NPU"]

    mock_path = mock.MagicMock()
    mock_path.__truediv__.return_value = "mocked_resolved_path"

    with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
      with mock.patch("importlib.resources.files") as mock_files:
        mock_files.return_value = mock_path
        npu = litert_lm.Backend.NPU()

        mock_files.assert_called_once_with(
            "litert_lm"
        )
        mock_path.__truediv__.assert_called_once_with(
            "vendors/intel_openvino/dispatch/"
        )
        self.assertEqual(npu.litert_dispatch_lib_dir, "mocked_resolved_path")

  def test_engine_init_with_legacy_backend_class(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      engine = litert_lm.Engine(
          self.model_path,
          litert_lm.Backend.CPU,
          max_num_tokens=10,
          cache_dir=":nocache",
      )
      self.assertIsNotNone(engine)
      self.assertLen(w, 1)
      self.assertTrue(issubclass(w[0].category, DeprecationWarning))
      self.assertIn(
          "Passing Backend class CPU is deprecated", str(w[0].message)
      )
      self.assertIsInstance(engine.backend, litert_lm.Backend.CPU)

  def test_sampler_config_validation(self):
    # Invalid top_k
    with self.assertRaisesRegex(ValueError, "top_k should be positive"):
      litert_lm.SamplerConfig(top_k=0, top_p=0.9, temperature=0.7)

    # Invalid top_p (low)
    with self.assertRaisesRegex(ValueError, "top_p should between 0 and 1"):
      litert_lm.SamplerConfig(top_k=40, top_p=-0.1, temperature=0.7)

    # Invalid top_p (high)
    with self.assertRaisesRegex(ValueError, "top_p should between 0 and 1"):
      litert_lm.SamplerConfig(top_k=40, top_p=1.1, temperature=0.7)

    # Invalid temperature
    with self.assertRaisesRegex(
        ValueError, "temperature should be non-negative"
    ):
      litert_lm.SamplerConfig(top_k=40, top_p=0.9, temperature=-0.1)

  def test_conversation_with_sampler_config(self):
    sampler_config = litert_lm.SamplerConfig(
        top_k=10, top_p=0.95, temperature=0.8, seed=123
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation(
            sampler_config=sampler_config
        ) as conversation,
    ):
      self.assertEqual(conversation.sampler_config, sampler_config)
      message = conversation.send_message("Hello world!")
      self.assertIn("role", message)
      self.assertEqual(message["role"], "assistant")

  def test_session_with_sampler_config(self):
    sampler_config = litert_lm.SamplerConfig(
        top_k=10, top_p=0.95, temperature=0.8, seed=123
    )
    with (
        self._create_engine() as engine,
        engine.create_session(sampler_config=sampler_config) as session,
    ):
      self.assertIsNotNone(session)
      session.run_prefill(["Hello world!"])
      responses = session.run_decode()
      self.assertIsNotNone(responses.texts)

  def test_conversation_send_message(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertIsNotNone(engine)
      self.assertIsNotNone(conversation)
      message = conversation.send_message("Hello world!")

      expected_message = {
          "role": "assistant",
          "content": [{"type": "text", "text": self._EXPECTED_RESPONSE}],
      }
      self.assertEqual(message, expected_message)

  def test_conversation_render_message_to_string(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertIsNotNone(engine)
      self.assertIsNotNone(conversation)
      rendered = conversation.render_message_to_string("Hello world!")
      self.assertIsInstance(rendered, str)
      self.assertIn("Hello world!", rendered)

  def test_conversation_send_message_async(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertIsNotNone(engine)
      self.assertIsNotNone(conversation)
      stream = conversation.send_message_async("Hello world!")
      text_pieces = self._extract_text(stream)

      self.assertEqual("".join(text_pieces), self._EXPECTED_RESPONSE)
      self.assertLen(text_pieces, 6)

  def test_conversation_send_message_async_cancel(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      stream = conversation.send_message_async("Hello world!")

      text_pieces = []
      for chunk in stream:
        content_list = chunk.get("content", [])
        for item in content_list:
          if item.get("type") == "text":
            text_pieces.append(item.get("text", ""))

        # Cancel the process after receiving the first chunk.
        conversation.cancel_process()

      # We only expect to receive the first piece before cancellation.
      self.assertNotEmpty(text_pieces)
      # NOTE: We don't assert len(text_pieces) < 6 here because on fast machines
      # (like Mac arm64) the generation might complete before the cancellation
      # signal is processed by the background thread.

  def test_benchmark_class(self):
    benchmark = litert_lm.Benchmark(
        self.model_path,
        litert_lm.Backend.CPU(),
        prefill_tokens=10,
        decode_tokens=10,
        cache_dir=":nocache",
    )
    self.assertIsInstance(benchmark, litert_lm.AbstractBenchmark)
    result = benchmark.run()
    self.assertIsInstance(result, litert_lm.BenchmarkInfo)
    self.assertGreater(result.init_time_in_second, 0)
    self.assertGreater(result.time_to_first_token_in_second, 0)
    self.assertGreater(result.last_prefill_token_count, 0)
    self.assertGreater(result.last_prefill_tokens_per_second, 0)
    self.assertGreater(result.last_decode_token_count, 0)
    self.assertGreater(result.last_decode_tokens_per_second, 0)

  def test_benchmark_class_with_thread_count(self):
    lib = litert_lm._ffi._get_lib()
    orig_set_num_threads = lib.litert_lm_engine_settings_set_num_threads
    with mock.patch.object(
        lib,
        "litert_lm_engine_settings_set_num_threads",
        autospec=True,
        side_effect=orig_set_num_threads,
    ) as mock_set_num_threads:
      benchmark = litert_lm.Benchmark(
          self.model_path,
          litert_lm.Backend.CPU(thread_count=4),
          prefill_tokens=10,
          decode_tokens=10,
          cache_dir=":nocache",
      )
      benchmark.run()

      mock_set_num_threads.assert_called_once_with(mock.ANY, 4)

  def test_engine_abc_inheritance(self):
    with self._create_engine() as engine:
      self.assertIsInstance(engine, litert_lm.AbstractEngine)

  def test_engine_tokenization_api(self):
    with self._create_engine() as engine:
      token_ids = engine.tokenize("Hello world!")
      self.assertNotEmpty(token_ids)
      self.assertTrue(all(isinstance(token_id, int) for token_id in token_ids))

      decoded = engine.detokenize(token_ids)
      self.assertIsInstance(decoded, str)
      self.assertNotEmpty(decoded)

  def test_engine_special_token_metadata(self):
    with self._create_engine() as engine:
      bos_token_id = engine.bos_token_id
      if bos_token_id is not None:
        self.assertIsInstance(bos_token_id, int)

      eos_token_ids = engine.eos_token_ids
      self.assertIsInstance(eos_token_ids, list)
      for stop_token_ids in eos_token_ids:
        self.assertIsInstance(stop_token_ids, list)
        self.assertTrue(
            all(isinstance(token_id, int) for token_id in stop_token_ids)
        )

  def test_conversation_abc_inheritance(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertIsInstance(conversation, litert_lm.AbstractConversation)

  def test_create_conversation_with_messages(self):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    with (
        self._create_engine() as engine,
        engine.create_conversation(messages=messages) as conversation,
    ):
      self.assertEqual(conversation.messages, messages)

  def test_create_conversation_with_message_objects(self):
    messages = [
        litert_lm.Message.system("You are a helpful assistant."),
        litert_lm.Message.user("Hello"),
    ]
    with (
        self._create_engine() as engine,
        engine.create_conversation(messages=messages) as conversation,
    ):
      self.assertEqual(conversation.messages, messages)

  def test_create_conversation_with_max_output_tokens(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation(max_output_tokens=1) as conversation,
    ):
      self.assertEqual(conversation.max_output_tokens, 1)
      message = conversation.send_message("Hello world!")
      self.assertEqual(message["role"], "assistant")
      # Response should be shorter because of max_output_tokens=1 default in
      # conversation
      text = "".join([c.get("text", "") for c in message.get("content", [])])
      self.assertLess(len(text), 10)

  def test_create_conversation_with_max_output_tokens_async(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation(max_output_tokens=1) as conversation,
    ):
      stream = conversation.send_message_async("Hello world!")
      text_pieces = self._extract_text(stream)
      self.assertLen(text_pieces, 1)
      self.assertLess(len("".join(text_pieces)), 10)

  def test_conversation_send_message_object(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = litert_lm.Message.user("Hello world!")
      message = conversation.send_message(user_message)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_send_contents_object(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_contents = litert_lm.Contents.of("Hello world!")
      message = conversation.send_message(user_contents)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_send_dict_message(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      message = conversation.send_message(user_message)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_with_thinking_config(self):
    thinking_config = litert_lm.ThinkingConfig(
        enable_thinking=True, thinking_token_budget=10
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation(
            thinking_config=thinking_config
        ) as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      message = conversation.send_message(
          user_message, thinking_config=thinking_config
      )
      self.assertEqual(message["role"], "assistant")
      self.assertEqual(conversation.thinking_config, thinking_config)

    with (
        self._create_engine() as engine,
        engine.create_conversation(
            thinking_config=thinking_config
        ) as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      responses = list(
          conversation.send_message_async(
              user_message, thinking_config=thinking_config
          )
      )
      self.assertNotEmpty(responses)

  def test_conversation_token_count(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertEqual(conversation.token_count, 0)
      user_message = {"role": "user", "content": "Hello world!"}
      conversation.send_message(user_message)
      self.assertEqual(conversation.token_count, 10)

  def test_conversation_get_benchmark_info(self):
    with (
        self._create_engine(enable_benchmark=True) as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      conversation.send_message(user_message)
      info = conversation.get_benchmark_info()
      self.assertIsInstance(info, litert_lm.BenchmarkInfo)
      self.assertGreaterEqual(info.init_time_in_second, 0.0)
      self.assertGreater(info.last_prefill_token_count, 0)
      self.assertGreater(info.last_decode_token_count, 0)

  def test_create_conversation_with_extra_context(self):
    extra_context = {"key": "value"}
    with (
        self._create_engine() as engine,
        engine.create_conversation(extra_context=extra_context) as conversation,
    ):
      self.assertEqual(conversation.extra_context, extra_context)

  def test_tool_event_handler_storage(self):

    class MyHandler(litert_lm.ToolEventHandler):

      def approve_tool_call(self, tool_call):
        return True

      def process_tool_response(self, tool_response):
        return tool_response

    handler = MyHandler()
    with (
        self._create_engine() as engine,
        engine.create_conversation(tool_event_handler=handler) as conversation,
    ):
      self.assertEqual(conversation.tool_event_handler, handler)

  def test_session_api_run_decode(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      self.assertIsInstance(session, litert_lm.AbstractSession)
      session.run_prefill(["Hello", " world!"])
      responses = session.run_decode()
      self.assertIsInstance(responses, litert_lm.Responses)
      self.assertLen(responses.texts, 1)
      self.assertEqual(responses.texts, [self._EXPECTED_RESPONSE])
      self.assertLen(responses.scores, 1)
      self.assertEmpty(responses.token_lengths)

  def test_session_api_run_text_scoring_with_token_lengths(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      self.assertIsInstance(session, litert_lm.AbstractSession)
      session.run_prefill(["Hello", " world!"])
      scoring_responses = session.run_text_scoring(
          ["Hello"], store_token_lengths=True
      )
      self.assertIsInstance(scoring_responses, litert_lm.Responses)
      self.assertEqual(scoring_responses.texts, ["Hello"])
      self.assertLen(scoring_responses.scores, 1)
      self.assertLen(scoring_responses.token_lengths, 1)
      self.assertIsInstance(scoring_responses.token_scores, list)
      self.assertLen(scoring_responses.token_scores, 1)
      self.assertIsInstance(scoring_responses.token_scores[0], list)
      self.assertLen(
          scoring_responses.token_scores[0],
          scoring_responses.token_lengths[0],
      )
      for score in scoring_responses.token_scores[0]:
        self.assertIsInstance(score, float)

  def test_session_api_run_text_scoring_no_token_lengths(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      self.assertIsInstance(session, litert_lm.AbstractSession)
      session.run_prefill(["Hello", " world!"])
      scoring_responses = session.run_text_scoring(
          ["Hello"], store_token_lengths=False
      )
      self.assertIsInstance(scoring_responses, litert_lm.Responses)
      self.assertEqual(scoring_responses.texts, ["Hello"])
      self.assertLen(scoring_responses.scores, 1)
      self.assertEmpty(scoring_responses.token_lengths)
      self.assertIsInstance(scoring_responses.token_scores, list)
      self.assertLen(scoring_responses.token_scores, 1)
      self.assertIsInstance(scoring_responses.token_scores[0], list)
      for score in scoring_responses.token_scores[0]:
        self.assertIsInstance(score, float)

  def test_session_api_run_decode_async(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      self.assertIsInstance(session, litert_lm.AbstractSession)
      session.run_prefill(["Hello", " world!"])
      stream = session.run_decode_async()
      responses = list(stream)
      self.assertNotEmpty(responses)
      self.assertLen(responses, 6)
      full_text = "".join(["".join(r.texts) for r in responses])
      self.assertEqual(full_text, self._EXPECTED_RESPONSE)

  def test_session_api_cancel_process(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      self.assertIsInstance(session, litert_lm.AbstractSession)
      session.run_prefill(["Hello world!"])
      stream = session.run_decode_async()

      responses = []
      for response in stream:
        responses.append(response)
        session.cancel_process()

      self.assertNotEmpty(responses)
      # NOTE: We don't assert len(responses) < 6 here because on fast machines
      # (like Mac arm64) the generation might complete before the cancellation
      # signal is processed by the background thread.

  def test_conversation_send_message_with_repetition_penalty_config(self):
    repetition_penalty_config = litert_lm.RepetitionPenaltyConfig(
        repetition_penalty=1.2,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        window_size=10,
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      message = conversation.send_message(
          "Hello world!",
          repetition_penalty_config=repetition_penalty_config,
      )
      self.assertIn("role", message)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_send_message_async_with_repetition_penalty_config(self):
    repetition_penalty_config = litert_lm.RepetitionPenaltyConfig(
        repetition_penalty=1.2,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        window_size=10,
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      stream = conversation.send_message_async(
          "Hello world!",
          repetition_penalty_config=repetition_penalty_config,
      )
      text_pieces = self._extract_text(stream)
      self.assertNotEmpty(text_pieces)

  def test_conversation_send_message_with_no_repeat_ngram_config(self):
    no_repeat_ngram_config = litert_lm.NoRepeatNgramConfig(
        no_repeat_ngram_size=3,
        window_size=10,
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      message = conversation.send_message(
          "Hello world!",
          no_repeat_ngram_config=no_repeat_ngram_config,
      )
      self.assertIn("role", message)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_send_message_async_with_no_repeat_ngram_config(self):
    no_repeat_ngram_config = litert_lm.NoRepeatNgramConfig(
        no_repeat_ngram_size=3,
        window_size=10,
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      stream = conversation.send_message_async(
          "Hello world!",
          no_repeat_ngram_config=no_repeat_ngram_config,
      )
      text_pieces = self._extract_text(stream)
      self.assertNotEmpty(text_pieces)

  def test_conversation_send_message_with_suppress_tokens_config(self):
    suppress_tokens_config = litert_lm.SuppressTokensConfig(
        suppress_tokens=[1, 2, 3],
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      message = conversation.send_message(
          "Hello world!",
          suppress_tokens_config=suppress_tokens_config,
      )
      self.assertIn("role", message)
      self.assertEqual(message["role"], "assistant")

  def test_conversation_send_message_async_with_suppress_tokens_config(self):
    suppress_tokens_config = litert_lm.SuppressTokensConfig(
        suppress_tokens=[1, 2, 3],
    )
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      stream = conversation.send_message_async(
          "Hello world!",
          suppress_tokens_config=suppress_tokens_config,
      )
      text_pieces = self._extract_text(stream)
      self.assertNotEmpty(text_pieces)

  def test_conversation_send_message_with_max_output_tokens(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      message = conversation.send_message("Hello world!", max_output_tokens=1)
      self.assertIn("role", message)
      self.assertEqual(message["role"], "assistant")
      # Response should be shorter because of max_output_tokens=1
      text = "".join([c.get("text", "") for c in message.get("content", [])])
      self.assertLess(len(text), 10)

  def test_conversation_send_message_async_with_max_output_tokens(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      stream = conversation.send_message_async(
          "Hello world!", max_output_tokens=1
      )
      text_pieces = self._extract_text(stream)
      self.assertLen(text_pieces, 1)
      self.assertLess(len("".join(text_pieces)), 10)

  @parameterized.parameters(True, False)
  def test_session_api_apply_prompt_template(self, apply_prompt_template):
    with self._create_engine() as engine:
      with engine.create_session(
          apply_prompt_template=apply_prompt_template
      ) as session:
        self.assertIsNotNone(session)


class FunctionCallingTest(LiteRtLmTestBase):

  def test_create_conversation_with_tools(self):

    def get_weather(location: str):
      """Gets weather for a location."""
      return f"Weather in {location} is sunny."

    tools = [get_weather]
    with (
        self._create_engine() as engine,
        engine.create_conversation(tools=tools) as conversation,
    ):
      self.assertEqual(conversation.tools, tools)

  def test_send_message_async_with_tools(self):

    def get_weather(location: str):
      """Gets weather for a location."""
      return f"Weather in {location} is sunny."

    tools = [get_weather]
    with (
        self._create_engine() as engine,
        engine.create_conversation(tools=tools) as conversation,
    ):
      user_message = {
          "role": "user",
          "content": "What's the weather in London?",
      }
      stream = conversation.send_message_async(user_message)
      text_pieces = self._extract_text(stream)
      self.assertNotEmpty(text_pieces)


if __name__ == "__main__":
  absltest.main()
