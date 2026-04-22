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

from absl import flags
from absl.testing import absltest

import litert_lm

FLAGS = flags.FLAGS


class LiteRtLmTestBase(absltest.TestCase):

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

  def _create_engine(self, max_num_tokens=10):
    return litert_lm.Engine(
        self.model_path,
        litert_lm.Backend.CPU,
        max_num_tokens=max_num_tokens,
        cache_dir=":nocache",
    )

  @staticmethod
  def _extract_text(stream):
    text_pieces = []
    for chunk in stream:
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          text_pieces.append(item.get("text", ""))
    return text_pieces


class EngineTest(LiteRtLmTestBase):

  _EXPECTED_RESPONSE = "TarefaByte دارایेत्र investigaciónప్రదేశ"

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
      user_message = {"role": "user", "content": "Hello world!"}
      message = conversation.send_message(user_message)
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
      user_message = {"role": "user", "content": "Hello world!"}
      message = conversation.send_message(user_message)

      expected_message = {
          "role": "assistant",
          "content": [{"type": "text", "text": self._EXPECTED_RESPONSE}],
      }
      self.assertEqual(message, expected_message)

  def test_conversation_send_message_async(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      self.assertIsNotNone(engine)
      self.assertIsNotNone(conversation)
      user_message = {"role": "user", "content": "Hello world!"}
      stream = conversation.send_message_async(user_message)
      text_pieces = self._extract_text(stream)

      self.assertEqual("".join(text_pieces), self._EXPECTED_RESPONSE)
      self.assertLen(text_pieces, 6)

  def test_conversation_send_message_async_cancel(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      stream = conversation.send_message_async(user_message)

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
      self.assertLess(len(text_pieces), 6)  # Cancelled before completion

  def test_benchmark_class(self):
    benchmark = litert_lm.Benchmark(
        self.model_path,
        litert_lm.Backend.CPU,
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

  def test_create_conversation_with_extra_context(self):
    extra_context = {"key": "value"}
    with (
        self._create_engine() as engine,
        engine.create_conversation(extra_context=extra_context) as conversation,
    ):
      self.assertEqual(conversation.extra_context, extra_context)

  def test_str_input_support(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      # Test with str input
      message = conversation.send_message("Hello world!")
      self.assertEqual(message["role"], "assistant")

  def test_str_input_support_async(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      # Test with str input (async)
      stream = conversation.send_message_async("Hello world!")
      text_pieces = self._extract_text(stream)
      self.assertNotEmpty(text_pieces)

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

  def test_create_session_with_apply_prompt_template(self):
    with self._create_engine() as engine:
      with engine.create_session(apply_prompt_template=True) as session:
        self.assertIsInstance(session, litert_lm.AbstractSession)
      with engine.create_session(apply_prompt_template=False) as session:
        self.assertIsInstance(session, litert_lm.AbstractSession)

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
      self.assertEmpty(scoring_responses.texts)
      self.assertLen(scoring_responses.scores, 1)
      self.assertLen(scoring_responses.token_lengths, 1)

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
      self.assertEmpty(scoring_responses.texts)
      self.assertLen(scoring_responses.scores, 1)
      self.assertEmpty(scoring_responses.token_lengths)

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
      # We expect fewer responses than a full decode (which is 6 chunks).
      self.assertLess(len(responses), 6)


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
