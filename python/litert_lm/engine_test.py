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

import os
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

  def _create_engine(self, max_num_tokens=10, enable_benchmark=False, **kwargs):
    return litert_lm.Engine(
        self.model_path,
        litert_lm.Backend.CPU(),
        max_num_tokens=max_num_tokens,
        cache_dir=":nocache",
        enable_benchmark=enable_benchmark,
        **kwargs,
    )

  @classmethod
  def _extract_text(cls, stream):
    return [str(chunk) for chunk in stream if str(chunk)]


class EngineTest(LiteRtLmTestBase):

  _EXPECTED_RESPONSE = (
      " spectrophot spectrophot<unused178><unused178><unused178><unused178>"
  )

  def test_engine_init_fail(self):
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_engine_create failed with status \w+ \(\d+\): .+",
    ):
      litert_lm.Engine("/non/existent/path")

  def test_create_checked_raises_with_last_error_message(self):
    lib = litert_lm._ffi._get_lib()
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_engine_create_session failed with status"
        r" INVALID_ARGUMENT \(3\): .+",
    ):
      litert_lm._ffi.create_checked(
          lib, "litert_lm_engine_create_session", None, None
      )

  def test_call_checked_raises_with_last_error_message(self):
    lib = litert_lm._ffi._get_lib()
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_engine_settings_set_num_threads failed with status"
        r" INVALID_ARGUMENT \(3\): .+",
    ):
      litert_lm._ffi.call_checked(
          lib, "litert_lm_engine_settings_set_num_threads", None, 4
      )

  def test_set_min_log_severity_invalid_raises(self):
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_set_min_log_level failed with status INVALID_ARGUMENT"
        r" \(3\): Unknown LiteRtLmLogSeverity",
    ):
      # 6 is within the value range of LiteRtLmLogSeverity but is not a named
      # enumerator. Values outside the range are undefined behavior in C++.
      litert_lm._ffi.set_min_log_severity(6)  # pyrefly: ignore[bad-argument-type]

  def test_constraint_provider_is_passed_by_value(self):
    lib = litert_lm._ffi._get_lib()
    config = litert_lm._ffi.create_checked(
        lib, "litert_lm_conversation_config_create"
    )
    try:
      for provider in litert_lm.LiteRtLmConstraintProviderType:
        litert_lm._ffi.call_checked(
            lib,
            "litert_lm_conversation_config_set_constraint_provider",
            config,
            provider.value,
        )
    finally:
      lib.litert_lm_conversation_config_delete(config)

  def test_constraint_provider_rejects_unknown_value(self):
    lib = litert_lm._ffi._get_lib()
    config = litert_lm._ffi.create_checked(
        lib, "litert_lm_conversation_config_create"
    )
    try:
      # A C caller can pass any int, so the library must reject values that
      # are not declared enumerators. C++ tests can't pass them without
      # undefined behavior, because every value in the enum's range is
      # declared.
      with self.assertRaisesRegex(
          RuntimeError,
          r"litert_lm_conversation_config_set_constraint_provider failed with"
          r" status INVALID_ARGUMENT \(3\): Unknown"
          r" LiteRtLmConstraintProviderType",
      ):
        litert_lm._ffi.call_checked(
            lib,
            "litert_lm_conversation_config_set_constraint_provider",
            config,
            2,
        )
    finally:
      lib.litert_lm_conversation_config_delete(config)

  def test_check_status_unknown_code(self):
    lib = litert_lm._ffi._get_lib()
    litert_lm._ffi.check_status(lib, "some_function", 0)
    with self.assertRaisesRegex(
        RuntimeError,
        r"some_function failed with status UNKNOWN_STATUS_CODE \(1234\)",
    ):
      litert_lm._ffi.check_status(lib, "some_function", 1234)

  def test_engine_init_setter_failure_raises_and_frees_settings(self):
    lib = litert_lm._ffi._get_lib()
    orig_delete = lib.litert_lm_engine_settings_delete
    self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_max_num_tokens",
            autospec=True,
            return_value=litert_lm._ffi.StatusCode.INTERNAL,
        )
    )
    mock_delete = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_delete",
            autospec=True,
            side_effect=orig_delete,
        )
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_engine_settings_set_max_num_tokens failed with status"
        r" INTERNAL \(13\)",
    ):
      self._create_engine(max_num_tokens=10)
    mock_delete.assert_called_once()

  def test_create_conversation_setter_failure_raises(self):
    engine = self._create_engine()
    lib = litert_lm._ffi._get_lib()
    self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_conversation_config_set_system_message",
            autospec=True,
            return_value=litert_lm._ffi.StatusCode.INVALID_ARGUMENT,
        )
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_conversation_config_set_system_message failed with status"
        r" INVALID_ARGUMENT \(3\)",
    ):
      engine.create_conversation(system_message="hi")

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

  def test_engine_init_with_use_ringbuffers_local_attention(self):
    lib = litert_lm._ffi._get_lib()
    orig_fn = lib.litert_lm_engine_settings_set_use_ringbuffers_local_attention

    mock_set_ringbuffers = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_use_ringbuffers_local_attention",
            autospec=True,
            side_effect=orig_fn,
        )
    )

    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        use_ringbuffers_local_attention=True,
        cache_dir=":nocache",
    )

    mock_set_ringbuffers.assert_called_once_with(mock.ANY, True)

  def test_engine_init_with_selected_signatures(self):
    lib = litert_lm._ffi._get_lib()
    selections = []

    def fake_set_selected_signatures(settings, signatures, num_signatures):
      del settings
      selections.append(
          [signatures[i].decode("utf-8") for i in range(num_signatures)]
      )
      return 0  # kLiteRtLmStatusOk

    self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_selected_signatures",
            side_effect=fake_set_selected_signatures,
        )
    )

    # Default should not select signatures.
    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        cache_dir=":nocache",
    )
    self.assertEqual(selections, [])

    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        selected_signatures=["prefill_1024", "decode"],
        cache_dir=":nocache",
    )
    self.assertEqual(selections, [["prefill_1024", "decode"]])

  def test_engine_init_with_selected_signatures_str_raises(self):
    with self.assertRaises(TypeError):
      litert_lm.Engine(
          self.model_path,
          backend=litert_lm.Backend.CPU(),
          selected_signatures="decode",
          cache_dir=":nocache",
      )

  def test_engine_init_with_empty_selected_signatures_raises(self):
    # An explicit empty selection is forwarded to the C API, which rejects it,
    # rather than silently falling back to all of the model's signatures.
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_engine_settings_set_selected_signatures failed with status"
        r" INVALID_ARGUMENT \(3\): .+",
    ):
      litert_lm.Engine(
          self.model_path,
          backend=litert_lm.Backend.CPU(),
          selected_signatures=[],
          cache_dir=":nocache",
      )

  def test_engine_init_with_enable_ynnpack(self):
    lib = litert_lm._ffi._get_lib()
    orig_fn = lib.litert_lm_engine_settings_set_enable_ynnpack

    mock_set_enable_ynnpack = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_enable_ynnpack",
            autospec=True,
            side_effect=orig_fn,
        )
    )

    # Default should not call set_enable_ynnpack.
    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        cache_dir=":nocache",
    )
    mock_set_enable_ynnpack.assert_not_called()

    # enable_ynnpack=True (succeeds if YNNPACK is compiled into the build,
    # or raises RuntimeError when YNNPACK is excluded at build time).
    try:
      litert_lm.Engine(
          self.model_path,
          backend=litert_lm.Backend.CPU(),
          enable_ynnpack=True,
          cache_dir=":nocache",
      )
    except RuntimeError:
      pass
    mock_set_enable_ynnpack.assert_called_once_with(mock.ANY, True)
    mock_set_enable_ynnpack.reset_mock()

    # enable_ynnpack=False
    litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        enable_ynnpack=False,
        cache_dir=":nocache",
    )
    mock_set_enable_ynnpack.assert_called_once_with(mock.ANY, False)

  def test_engine_init_with_max_vision_tokens_per_image(self):
    lib = litert_lm._ffi._get_lib()
    orig_fn = lib.litert_lm_engine_settings_set_max_vision_tokens_per_image

    mock_set_max_vision_tokens = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_max_vision_tokens_per_image",
            autospec=True,
            side_effect=orig_fn,
        )
    )

    # Default should not call set_max_vision_tokens_per_image.
    with litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        cache_dir=":nocache",
    ) as engine:
      self.assertIsNone(engine.max_vision_tokens_per_image)
    mock_set_max_vision_tokens.assert_not_called()

    with litert_lm.Engine(
        self.model_path,
        backend=litert_lm.Backend.CPU(),
        max_vision_tokens_per_image=70,
        cache_dir=":nocache",
    ) as engine:
      self.assertEqual(engine.max_vision_tokens_per_image, 70)
    mock_set_max_vision_tokens.assert_called_once_with(mock.ANY, 70)

  def test_create_conversation_with_visual_token_budget(self):
    lib = litert_lm._ffi._get_lib()
    orig_fn = lib.litert_lm_conversation_optional_args_set_visual_token_budget

    mock_set_visual_token_budget = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_conversation_optional_args_set_visual_token_budget",
            autospec=True,
            side_effect=orig_fn,
        )
    )

    with self.subTest("unset_budget_is_not_sent"):
      with (
          self._create_engine() as engine,
          engine.create_conversation() as conversation,
      ):
        conversation.send_message("Hello world!")
      mock_set_visual_token_budget.assert_not_called()

    # Bind before entering: `with ... as engine` is typed as AbstractEngine,
    # whose create_conversation() does not accept visual_token_budget.
    engine = litert_lm.Engine(
        self.model_path,
        litert_lm.Backend.CPU(),
        max_num_tokens=10,
        max_vision_tokens_per_image=280,
        cache_dir=":nocache",
    )
    with engine:
      with self.subTest("engine_max_is_not_used_as_default"):
        mock_set_visual_token_budget.reset_mock()
        with engine.create_conversation() as conversation:
          conversation.send_message("Hello world!")
        mock_set_visual_token_budget.assert_not_called()

      with self.subTest("send_message"):
        mock_set_visual_token_budget.reset_mock()
        with engine.create_conversation(visual_token_budget=70) as conversation:
          conversation.send_message("Hello world!")
        mock_set_visual_token_budget.assert_called_once_with(mock.ANY, 70)

      with self.subTest("send_message_async"):
        mock_set_visual_token_budget.reset_mock()
        with engine.create_conversation(visual_token_budget=70) as conversation:
          self._extract_text(conversation.send_message_async("Hello world!"))
        mock_set_visual_token_budget.assert_called_once_with(mock.ANY, 70)

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

    with (
        mock.patch.dict("sys.modules", {"openvino": mock_ov}),
        mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True),
        mock.patch("importlib.resources.files") as mock_files,
    ):
      mock_files.return_value = mock_path
      npu = litert_lm.Backend.NPU()

      mock_files.assert_called_once_with(
          "litert_lm"
      )
      mock_path.__truediv__.assert_called_once_with(
          "vendors/intel_openvino/dispatch/"
      )
      self.assertEqual(npu.litert_dispatch_lib_dir, "mocked_resolved_path")
      expected_libs_dir = os.path.join("path/to/openvino", "libs")
      self.assertEqual(
          os.environ["PATH"], f"/usr/bin{os.pathsep}{expected_libs_dir}"
      )

  @mock.patch("sys.platform", "win32")
  def test_npu_backend_windows_openvino_path_deduplication(self):
    mock_ov = mock.MagicMock()
    mock_ov.__file__ = "path/to/openvino/__init__.py"
    mock_ov.Core.return_value.available_devices = ["CPU", "NPU"]

    mock_path = mock.MagicMock()
    mock_path.__truediv__.return_value = "mocked_resolved_path"
    expected_libs_dir = os.path.join("path/to/openvino", "libs")

    with (
        mock.patch.dict("sys.modules", {"openvino": mock_ov}),
        mock.patch("importlib.resources.files", return_value=mock_path),
    ):
      with mock.patch.dict(os.environ, {}, clear=True):
        litert_lm.Backend.NPU()
        self.assertEqual(os.environ["PATH"], expected_libs_dir)
        litert_lm.Backend.NPU()
        self.assertEqual(os.environ["PATH"], expected_libs_dir)

      prefix_match_dir = f"{expected_libs_dir}_extra"
      with mock.patch.dict(
          os.environ, {"PATH": prefix_match_dir}, clear=True
      ):
        litert_lm.Backend.NPU()
        litert_lm.Backend.NPU()
        self.assertEqual(
            os.environ["PATH"],
            f"{prefix_match_dir}{os.pathsep}{expected_libs_dir}",
        )

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
      self.assertIsInstance(message, litert_lm.Message)
      self.assertEqual(str(message), self._EXPECTED_RESPONSE)

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
        if str(chunk):
          text_pieces.append(str(chunk))

        # Cancel the process after receiving the first chunk.
        conversation.cancel_process()

      # We only expect to receive the first piece before cancellation.
      self.assertNotEmpty(text_pieces)
      # NOTE: We don't assert len(text_pieces) < 6 here because on fast machines
      # (like Mac arm64) the generation might complete before the cancellation
      # signal is processed by the background thread.

  def test_benchmark_class(self):
    with litert_lm.Benchmark(
        self.model_path,
        litert_lm.Backend.CPU(),
        prefill_tokens=10,
        decode_tokens=10,
        cache_dir=":nocache",
    ) as benchmark:
      self.assertIsInstance(benchmark, litert_lm.AbstractBenchmark)
      result = benchmark.run()
      self.assertIsInstance(result, litert_lm.BenchmarkInfo)
      self.assertGreater(result.init_time_in_second, 0)
      self.assertGreater(result.time_to_first_token_in_second, 0)
      self.assertGreater(result.last_prefill_token_count, 0)
      self.assertGreater(result.last_prefill_tokens_per_second, 0)
      self.assertGreater(result.last_decode_token_count, 0)
      self.assertGreater(result.last_decode_tokens_per_second, 0)

  def test_benchmark_reuses_engine_across_runs(self):
    lib = litert_lm._ffi._get_lib()
    with (
        mock.patch.object(
            lib,
            "litert_lm_engine_create",
            wraps=lib.litert_lm_engine_create,
        ) as mock_engine_create,
        mock.patch.object(
            lib,
            "litert_lm_engine_delete",
            wraps=lib.litert_lm_engine_delete,
        ) as mock_engine_delete,
        mock.patch.object(
            lib,
            "litert_lm_engine_create_session",
            wraps=lib.litert_lm_engine_create_session,
        ) as mock_create_session,
        mock.patch.object(
            lib,
            "litert_lm_session_delete",
            wraps=lib.litert_lm_session_delete,
        ) as mock_session_delete,
    ):
      with litert_lm.Benchmark(
          self.model_path,
          litert_lm.Backend.CPU(),
          prefill_tokens=10,
          decode_tokens=10,
          cache_dir=":nocache",
      ) as benchmark:
        warmup_result = benchmark.run()
        iter_result = benchmark.run()
        self.assertIsInstance(warmup_result, litert_lm.BenchmarkInfo)
        self.assertIsInstance(iter_result, litert_lm.BenchmarkInfo)
        mock_engine_create.assert_called_once()
        mock_engine_delete.assert_not_called()
        self.assertEqual(mock_create_session.call_count, 2)
        self.assertEqual(mock_session_delete.call_count, 2)

      mock_engine_delete.assert_called_once()
      benchmark.close()
      mock_engine_delete.assert_called_once()

      # Calling run() outside a context manager deterministically deletes the
      # engine on completion.
      benchmark.run()
      self.assertEqual(mock_engine_create.call_count, 2)
      self.assertEqual(mock_engine_delete.call_count, 2)

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

  def test_benchmark_class_with_enable_speculative_decoding(self):
    lib = litert_lm._ffi._get_lib()
    orig_set_mode = lib.litert_lm_speculative_decoding_config_set_mode
    with mock.patch.object(
        lib,
        "litert_lm_speculative_decoding_config_set_mode",
        autospec=True,
        side_effect=orig_set_mode,
    ) as mock_set_mode:
      with self.assertWarnsRegex(
          DeprecationWarning, "enable_speculative_decoding is deprecated"
      ):
        benchmark = litert_lm.Benchmark(
            self.model_path,
            litert_lm.Backend.CPU(),
            prefill_tokens=10,
            decode_tokens=10,
            cache_dir=":nocache",
            enable_speculative_decoding=False,
        )
      benchmark.run()

      # The deprecated flag maps to mode OFF.
      mock_set_mode.assert_called_once_with(
          mock.ANY, int(litert_lm.SpeculativeDecodingMode.OFF)
      )

  def test_benchmark_class_with_speculative_decoding_config(self):
    lib = litert_lm._ffi._get_lib()
    orig_set_config = (
        lib.litert_lm_engine_settings_set_speculative_decoding_config
    )
    with mock.patch.object(
        lib,
        "litert_lm_engine_settings_set_speculative_decoding_config",
        autospec=True,
        side_effect=orig_set_config,
    ) as mock_set_config:
      benchmark = litert_lm.Benchmark(
          self.model_path,
          litert_lm.Backend.CPU(),
          prefill_tokens=10,
          decode_tokens=10,
          cache_dir=":nocache",
          speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
              mode=litert_lm.SpeculativeDecodingMode.OFF
          ),
      )
      benchmark.run()

      mock_set_config.assert_called_once()

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
      self.assertNotEmpty(eos_token_ids)
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
      self.assertLess(len(text), 15)

  def test_create_conversation_with_chat_template(self):
    tmpl = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    with (
        self._create_engine() as engine,
        engine.create_conversation(chat_template=tmpl) as conversation,
    ):
      self.assertEqual(conversation.chat_template, tmpl)

  def test_conversation_tools_without_system_message_gemma4_templates(self):
    def get_weather(location: str) -> str:
      """Get the current weather for a given location.

      Args:
        location: The city and state, e.g. San Francisco, CA.

      Returns:
        A string describing the weather.
      """
      return f"Sunny in {location}"

    base_dir = (
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm"
    )
    full_history_tmpl = (
        base_dir / "models/gemma4/chat_template_e2b_e4b.jinja"
    ).read_text(encoding="utf-8")
    multi_prefill_tmpl = (
        base_dir
        / "runtime/components/testdata/google-gemma-4-multi-prefill.jinja"
    ).read_text(encoding="utf-8")

    with self._create_engine(max_num_tokens=200) as engine:
      with engine.create_conversation(
          tools=[get_weather],
          chat_template=full_history_tmpl,
          automatic_tool_calling=False,
          max_output_tokens=1,
      ) as conversation:
        rendered = conversation.render_message_to_string("Hello world!")
        self.assertIn("get_weather", rendered)
        self.assertIn("Hello world!", rendered)
        message = conversation.send_message("Hello world!")
        self.assertEqual(message["role"], "assistant")

      with engine.create_conversation(
          tools=[get_weather],
          chat_template=multi_prefill_tmpl,
          automatic_tool_calling=False,
          max_output_tokens=1,
      ) as conversation:
        # With a single-turn/multi-prefill template (`is_appending_to_prefill`),
        # `RenderSingleTurnTemplate` first renders the preface alone with
        # `messages=[]` and `tools=[get_weather]` before rendering the user
        # turn.
        rendered = conversation.render_message_to_string("Hello world!")
        self.assertIn("get_weather", rendered)
        self.assertIn("Hello world!", rendered)
        message = conversation.send_message("Hello world!")
        self.assertEqual(message["role"], "assistant")

  def test_create_conversation_with_filter_channel_content_from_kv_cache(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation(
            filter_channel_content_from_kv_cache=True
        ) as conversation,
    ):
      self.assertIsNotNone(conversation)

  def _mock_speculative_decoding_set_mode(self):
    lib = litert_lm._ffi._get_lib()
    return mock.patch.object(
        lib,
        "litert_lm_speculative_decoding_config_set_mode",
        autospec=True,
        side_effect=lib.litert_lm_speculative_decoding_config_set_mode,
    )

  def _mock_session_config_set_speculative_decoding_config(self):
    lib = litert_lm._ffi._get_lib()
    return mock.patch.object(
        lib,
        "litert_lm_session_config_set_speculative_decoding_config",
        autospec=True,
        side_effect=lib.litert_lm_session_config_set_speculative_decoding_config,
    )

  def test_create_conversation_with_enable_speculative_decoding(self):
    with self._mock_speculative_decoding_set_mode() as mock_set_mode:
      with self._create_engine() as engine:
        for enable, mode in (
            (True, litert_lm.SpeculativeDecodingMode.ON),
            (False, litert_lm.SpeculativeDecodingMode.OFF),
        ):
          mock_set_mode.reset_mock()
          with self.assertWarnsRegex(
              DeprecationWarning, "enable_speculative_decoding is deprecated"
          ):
            conversation = engine.create_conversation(
                enable_speculative_decoding=enable
            )
          with conversation:
            self.assertIsNotNone(conversation)
          mock_set_mode.assert_called_once_with(mock.ANY, int(mode))

  def test_create_session_with_enable_speculative_decoding(self):
    with self._mock_speculative_decoding_set_mode() as mock_set_mode:
      with self._create_engine() as engine:
        for enable, mode in (
            (True, litert_lm.SpeculativeDecodingMode.ON),
            (False, litert_lm.SpeculativeDecodingMode.OFF),
        ):
          mock_set_mode.reset_mock()
          with self.assertWarnsRegex(
              DeprecationWarning, "enable_speculative_decoding is deprecated"
          ):
            session = engine.create_session(enable_speculative_decoding=enable)
          with session:
            self.assertIsNotNone(session)
          mock_set_mode.assert_called_once_with(mock.ANY, int(mode))

  def test_speculative_decoding_config_overrides_enable_speculative_decoding(
      self,
  ):
    with self._mock_speculative_decoding_set_mode() as mock_set_mode:
      with self._create_engine() as engine:
        with self.assertWarns(DeprecationWarning):
          session = engine.create_session(
              enable_speculative_decoding=False,
              speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
                  mode=litert_lm.SpeculativeDecodingMode.ADAPTIVE
              ),
          )
        with session:
          self.assertIsNotNone(session)
        mock_set_mode.assert_called_once_with(
            mock.ANY, int(litert_lm.SpeculativeDecodingMode.ADAPTIVE)
        )

  def test_create_conversation_with_speculative_decoding_config(self):
    with (
        self._mock_session_config_set_speculative_decoding_config() as mock_set,
        self._mock_speculative_decoding_set_mode() as mock_set_mode,
    ):
      with self._create_engine() as engine:
        with engine.create_conversation() as conversation:
          self.assertIsNotNone(conversation)
        mock_set.assert_not_called()

        with engine.create_conversation(
            speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
                mode=litert_lm.SpeculativeDecodingMode.ADAPTIVE
            )
        ) as conversation:
          self.assertIsNotNone(conversation)
        mock_set.assert_called_once()
        mock_set_mode.assert_called_once_with(
            mock.ANY, int(litert_lm.SpeculativeDecodingMode.ADAPTIVE)
        )

  def test_create_session_with_speculative_decoding_config(self):
    with (
        self._mock_session_config_set_speculative_decoding_config() as mock_set,
        self._mock_speculative_decoding_set_mode() as mock_set_mode,
    ):
      with self._create_engine() as engine:
        with engine.create_session() as session:
          self.assertIsNotNone(session)
        mock_set.assert_not_called()

        # A config without a mode still overrides (nothing) and sets no mode.
        with engine.create_session(
            speculative_decoding_config=litert_lm.SpeculativeDecodingConfig()
        ) as session:
          self.assertIsNotNone(session)
        mock_set.assert_called_once()
        mock_set_mode.assert_not_called()

        mock_set.reset_mock()
        with engine.create_session(
            speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
                mode=litert_lm.SpeculativeDecodingMode.OFF
            )
        ) as session:
          self.assertIsNotNone(session)
        mock_set.assert_called_once()
        mock_set_mode.assert_called_once_with(
            mock.ANY, int(litert_lm.SpeculativeDecodingMode.OFF)
        )

  def test_engine_with_speculative_decoding_config(self):
    lib = litert_lm._ffi._get_lib()
    with (
        mock.patch.object(
            lib,
            "litert_lm_engine_settings_set_speculative_decoding_config",
            autospec=True,
            side_effect=lib.litert_lm_engine_settings_set_speculative_decoding_config,
        ) as mock_set,
        self._mock_speculative_decoding_set_mode() as mock_set_mode,
    ):
      with self._create_engine() as engine:
        self.assertIsNone(engine.speculative_decoding_config)
      mock_set.assert_not_called()

      with self._create_engine(
          speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
              mode=litert_lm.SpeculativeDecodingMode.OFF
          )
      ) as engine:
        self.assertEqual(
            engine.speculative_decoding_config.mode,
            litert_lm.SpeculativeDecodingMode.OFF,
        )
      mock_set.assert_called_once()
      mock_set_mode.assert_called_once_with(
          mock.ANY, int(litert_lm.SpeculativeDecodingMode.OFF)
      )

  def test_engine_with_deprecated_enable_speculative_decoding(self):
    with self._mock_speculative_decoding_set_mode() as mock_set_mode:
      with self.assertWarnsRegex(
          DeprecationWarning, "enable_speculative_decoding is deprecated"
      ):
        engine = self._create_engine(enable_speculative_decoding=False)
      with engine:
        pass
      mock_set_mode.assert_called_once_with(
          mock.ANY, int(litert_lm.SpeculativeDecodingMode.OFF)
      )

  def test_speculative_decoding_config_lenience(self):
    lib = litert_lm._ffi._get_lib()
    with mock.patch.object(
        lib,
        "litert_lm_speculative_decoding_config_set_lenience",
        autospec=True,
        side_effect=lib.litert_lm_speculative_decoding_config_set_lenience,
    ) as mock_set_lenience:
      with self._create_engine(
          speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
              lenience=0.5
          )
      ) as engine:
        mock_set_lenience.assert_called_once_with(mock.ANY, 0.5)
        mock_set_lenience.reset_mock()
        with engine.create_session(
            speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(
                lenience=0.25
            )
        ) as session:
          self.assertIsNotNone(session)
        mock_set_lenience.assert_called_once_with(mock.ANY, 0.25)

  def test_speculative_decoding_config_rejects_invalid_lenience(self):
    for lenience in (-0.1, 1.1, float("nan")):
      with self.subTest(lenience=lenience):
        with self.assertRaises(ValueError):
          litert_lm.SpeculativeDecodingConfig(lenience=lenience)

  def test_speculative_decoding_config_rejects_invalid_mode(self):
    with self.assertRaises(ValueError):
      litert_lm.SpeculativeDecodingConfig(mode=1)  # pyrefly: ignore[bad-argument-type]

  def test_speculative_decoding_mode_values_match_c_api(self):
    self.assertEqual(int(litert_lm.SpeculativeDecodingMode.OFF), 0)
    self.assertEqual(int(litert_lm.SpeculativeDecodingMode.ON), 1)
    self.assertEqual(int(litert_lm.SpeculativeDecodingMode.ADAPTIVE), 2)

  def test_create_conversation_with_max_output_tokens_async(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation(max_output_tokens=1) as conversation,
    ):
      stream = conversation.send_message_async("Hello world!")
      text_pieces = self._extract_text(stream)
      self.assertLen(text_pieces, 1)
      self.assertLess(len("".join(text_pieces)), 15)

  def test_conversation_send_message_object(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = litert_lm.Message.user("Hello world!")
      message = conversation.send_message(user_message)
      self.assertEqual(message.role, litert_lm.Role.MODEL)

  def test_conversation_send_contents_object(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_contents = litert_lm.Contents.of("Hello world!")
      message = conversation.send_message(user_contents)
      self.assertEqual(message.role, litert_lm.Role.MODEL)

  def test_conversation_send_dict_message(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      user_message = {"role": "user", "content": "Hello world!"}
      message = conversation.send_message(user_message)
      self.assertEqual(message.role, litert_lm.Role.MODEL)

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

  def test_conversation_clone(self):
    def get_weather(location: str) -> str:
      """Gets weather for a location."""
      return f"Weather in {location} is sunny."

    messages = [litert_lm.Message.system("You are a helpful assistant.")]
    extra_context = {"key": "value"}
    thinking_config = litert_lm.ThinkingConfig(
        enable_thinking=True, thinking_token_budget=10
    )
    sampler_config = litert_lm.SamplerConfig(
        top_k=1, top_p=1.0, temperature=0.0, seed=42
    )
    constrained_config = litert_lm.ConstrainedDecodingConfig(enable=True)
    tmpl = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"

    engine = self._create_engine(max_num_tokens=64)
    with (
        engine,
        engine.create_conversation(
            messages=messages,
            tools=[get_weather],
            automatic_tool_calling=False,
            extra_context=extra_context,
            thinking_config=thinking_config,
            sampler_config=sampler_config,
            max_output_tokens=2,
            chat_template=tmpl,
            constrained_decoding_config=constrained_config,
            visual_token_budget=70,
        ) as conversation,
    ):
      first_response = conversation.send_message("Hello")
      self.assertIsNotNone(first_response)
      token_count_after_first = conversation.token_count
      self.assertGreater(token_count_after_first, 0)

      with conversation.clone() as cloned:
        self.assertIsInstance(cloned, litert_lm.Conversation)
        self.assertEqual(cloned.messages, conversation.messages)
        self.assertIsNot(cloned.messages, conversation.messages)
        self.assertEqual(cloned.tools, conversation.tools)
        self.assertIsNot(cloned.tools, conversation.tools)
        self.assertEqual(
            cloned.automatic_tool_calling, conversation.automatic_tool_calling
        )
        self.assertEqual(cloned.extra_context, conversation.extra_context)
        self.assertIsNot(cloned.extra_context, conversation.extra_context)
        self.assertEqual(cloned.thinking_config, conversation.thinking_config)
        self.assertEqual(cloned.sampler_config, conversation.sampler_config)
        self.assertEqual(
            cloned.max_output_tokens, conversation.max_output_tokens
        )
        self.assertEqual(cloned.chat_template, conversation.chat_template)
        self.assertEqual(
            cloned.constrained_decoding_config,
            conversation.constrained_decoding_config,
        )
        self.assertEqual(cloned.token_count, token_count_after_first)

        orig_second = conversation.send_message("world")
        cloned_second = cloned.send_message("world")
        self.assertEqual(orig_second, cloned_second)

  def test_conversation_clone_after_original_closed(self):
    with self._create_engine(max_num_tokens=20) as engine:
      conversation = engine.create_conversation(max_output_tokens=2)
      conversation.send_message("Hello")
      cloned = conversation.clone()
      conversation.close()

      with cloned:
        response = cloned.send_message("world")
        self.assertEqual(response.role, litert_lm.Role.MODEL)
        self.assertNotEmpty(str(response))

  def test_conversation_clone_when_closed_raises(self):
    with self._create_engine() as engine:
      conversation = engine.create_conversation()
      conversation.close()
      with self.assertRaisesRegex(RuntimeError, "Conversation is closed."):
        conversation.clone()

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

  def test_conversation_get_benchmark_info_without_benchmark_raises(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      with self.assertRaisesRegex(
          RuntimeError,
          r"litert_lm_conversation_get_benchmark_info failed with status",
      ):
        conversation.get_benchmark_info()

  def test_create_conversation_failure_raises_and_frees_config(self):
    engine = self._create_engine()
    lib = litert_lm._ffi._get_lib()
    orig_config_delete = lib.litert_lm_conversation_config_delete
    self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_conversation_create",
            autospec=True,
            return_value=litert_lm._ffi.StatusCode.INTERNAL,
        )
    )
    mock_config_delete = self.enter_context(
        mock.patch.object(
            lib,
            "litert_lm_conversation_config_delete",
            autospec=True,
            side_effect=orig_config_delete,
        )
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r"litert_lm_conversation_create failed with status INTERNAL \(13\)",
    ):
      engine.create_conversation()
    mock_config_delete.assert_called_once()

  def test_conversation_token_count_failure_raises(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      lib = litert_lm._ffi._get_lib()
      with mock.patch.object(
          lib,
          "litert_lm_conversation_get_token_count",
          autospec=True,
          return_value=litert_lm._ffi.StatusCode.INTERNAL,
      ):
        with self.assertRaisesRegex(
            RuntimeError,
            r"litert_lm_conversation_get_token_count failed with status"
            r" INTERNAL \(13\)",
        ):
          _ = conversation.token_count

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

  def test_session_clone(self):
    with (
        self._create_engine() as engine,
        engine.create_session() as session,
    ):
      session.run_prefill(["Hello", " world!"])
      with session.clone() as cloned:
        self.assertIsInstance(cloned, litert_lm.Session)
        orig_responses = session.run_decode()
        cloned_responses = cloned.run_decode()
        self.assertEqual(orig_responses.texts, [self._EXPECTED_RESPONSE])
        self.assertEqual(cloned_responses.texts, [self._EXPECTED_RESPONSE])

  def test_session_clone_after_original_closed(self):
    with self._create_engine() as engine:
      session = engine.create_session()
      session.run_prefill(["Hello", " world!"])
      cloned = session.clone()
      session.close()

      with cloned:
        cloned_responses = cloned.run_decode()
        self.assertEqual(cloned_responses.texts, [self._EXPECTED_RESPONSE])

  def test_session_clone_when_closed_raises(self):
    with self._create_engine() as engine:
      session = engine.create_session()
      session.close()
      with self.assertRaisesRegex(RuntimeError, "Session is closed."):
        session.clone()

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

  def test_session_api_scoring_matches_prefill(self):
    with self._create_engine(max_num_tokens=16) as engine:
      # Case A: Prefill "Hello", score " world again"
      with engine.create_session() as s1:
        s1.run_prefill(["Hello"])
        res1 = s1.run_text_scoring([" world again"], store_token_lengths=True)
        token_scores1 = res1.token_scores[0]
        score_again_case_a = token_scores1[-1]

      # Case B: Prefill "Hello world", score " again"
      with engine.create_session() as s2:
        s2.run_prefill(["Hello world"])
        res2 = s2.run_text_scoring([" again"], store_token_lengths=True)
        token_scores2 = res2.token_scores[0]
        score_again_case_b = token_scores2[-1]

      self.assertAlmostEqual(score_again_case_a, score_again_case_b, places=3)

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
      self.assertEqual(message.role, litert_lm.Role.MODEL)

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
      self.assertEqual(message.role, litert_lm.Role.MODEL)

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
      self.assertEqual(message.role, litert_lm.Role.MODEL)

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
      self.assertEqual(message.role, litert_lm.Role.MODEL)
      # Response should be shorter because of max_output_tokens=1
      text = str(message)
      self.assertLess(len(text), 15)

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
      self.assertLess(len("".join(text_pieces)), 15)

  @parameterized.parameters(True, False)
  def test_session_api_apply_prompt_template(self, apply_prompt_template):
    with self._create_engine() as engine:
      with engine.create_session(
          apply_prompt_template=apply_prompt_template
      ) as session:
        self.assertIsNotNone(session)

  def test_response_format_validation(self):
    with self.assertRaisesRegex(ValueError, "Invalid JSON schema string"):
      litert_lm.ResponseFormat.json("{invalid_json: true")

  def test_response_format_without_enabling(self):
    with (
        self._create_engine() as engine,
        engine.create_conversation() as conversation,
    ):
      with self.assertRaisesRegex(
          ValueError,
          "response_format cannot be used unless constrained_decoding_config",
      ):
        conversation.send_message(
            "What is the capital of France?",
            response_format=litert_lm.ResponseFormat.regex("[0-9]{3}"),
        )

      with self.assertRaisesRegex(
          ValueError,
          "response_format cannot be used unless constrained_decoding_config",
      ):
        # We must iterate the generator to trigger the exception
        next(
            conversation.send_message_async(
                "What is the capital of France?",
                response_format=litert_lm.ResponseFormat.regex("[0-9]{3}"),
            )
        )

  def test_create_conversation_enable_constrained_decoding(self):
    lib = litert_lm._ffi._get_lib()
    with (
        mock.patch.object(
            lib,
            "litert_lm_conversation_config_set_enable_constrained_decoding",
            wraps=lib.litert_lm_conversation_config_set_enable_constrained_decoding,
        ) as mock_set_enable_constrained,
        mock.patch.object(
            lib,
            "litert_lm_conversation_config_set_constraint_provider",
            wraps=lib.litert_lm_conversation_config_set_constraint_provider,
        ) as mock_set_constraint_provider,
    ):
      with (
          self._create_engine() as engine,
          engine.create_conversation(
              constrained_decoding_config=litert_lm.ConstrainedDecodingConfig(
                  enable=True
              )
          ) as conv,
      ):
        self.assertIsNotNone(conv)
        mock_set_enable_constrained.assert_called_once()
        args, _ = mock_set_enable_constrained.call_args
        self.assertTrue(args[1])

      mock_set_enable_constrained.reset_mock()
      mock_set_constraint_provider.reset_mock()

      with (
          self._create_engine() as engine,
          engine.create_conversation(
              constrained_decoding_config=litert_lm.ConstrainedDecodingConfig(
                  enable=True,
                  provider=litert_lm.LiteRtLmConstraintProviderType.LL_GUIDANCE,
              )
          ) as conv,
      ):
        self.assertIsNotNone(conv)
        mock_set_constraint_provider.assert_called_once()
        provider_args, _ = mock_set_constraint_provider.call_args
        self.assertEqual(
            provider_args[1],
            litert_lm.LiteRtLmConstraintProviderType.LL_GUIDANCE.value,
        )
        mock_set_enable_constrained.assert_called_once()
        args, _ = mock_set_enable_constrained.call_args
        self.assertTrue(args[1])


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
