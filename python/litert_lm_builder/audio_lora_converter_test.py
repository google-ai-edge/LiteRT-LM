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

"""Tests for Audio LoRA sidecar conversion utilities."""

from __future__ import annotations

import json
import pathlib
import struct
import tempfile
import unittest

from litert_lm_builder import audio_lora_converter


class _FakeArray:
  """Small array-like test double; keeps Bazel tests independent of NumPy."""

  def __init__(self, shape, bytes_per_element=2):
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    self._bytes_per_element = bytes_per_element

  @property
  def T(self):
    if self.ndim != 2:
      return self
    return _FakeArray((self.shape[1], self.shape[0]), self._bytes_per_element)

  @property
  def nbytes(self):
    count = 1
    for dim in self.shape:
      count *= dim
    return count * self._bytes_per_element

  def __mul__(self, _scale):
    return self

  def astype(self, dtype):
    dtype_name = getattr(dtype, "__name__", str(dtype)).lower()
    bytes_per_element = 4 if dtype_name in ("f32", "float32") else 2
    return _FakeArray(self.shape, bytes_per_element)

  def tobytes(self, order="C"):
    if order != "C":
      raise ValueError("test fake only supports C order")
    return b"\x00" * self.nbytes


def _zeros(shape):
  return _FakeArray(shape)


class AudioLoraConverterTest(unittest.TestCase):

  def test_peft_key_to_litert_lora_name(self):
    self.assertEqual(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "base_model.model.audio_tower.layers.7.self_attn.q_proj.lora_A.weight"
        ),
        "lora_audio_attn_q_a_weight_7",
    )
    self.assertIsNone(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "model.layers.12.attn.o_proj.lora_B.weight"
        )
    )
    self.assertEqual(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "model.audio_tower.layers.3.self_attn.post.linear.lora_A.weight"
        ),
        "lora_audio_attn_o_a_weight_3",
    )
    self.assertEqual(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "model.audio_tower.layers.2.feed_forward1.ffw_layer_1."
            "linear.lora_B.weight"
        ),
        "lora_audio_ff1_l1_b_weight_2",
    )
    self.assertEqual(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "model.audio_tower.layers.5.lconv1d.linear_end.linear."
            "lora_A.weight"
        ),
        "lora_audio_lconv_end_a_weight_5",
    )
    self.assertEqual(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "base_model.model.model.audio_tower.output_proj.lora_B.weight"
        ),
        "lora_audio_output_proj_b_weight_0",
    )
    self.assertIsNone(
        audio_lora_converter.peft_key_to_litert_lora_name(
            "model.layers.0.mlp.gate_proj.lora_A.weight"
        )
    )

  def test_build_lora_tflite_bytes(self):
    tensors = [
        audio_lora_converter.LoraTensor(
            name="lora_audio_attn_q_a_weight_0",
            data=b"\x00\x3c" * 8,
            shape=(2, 4),
        ),
        audio_lora_converter.LoraTensor(
            name="lora_audio_attn_q_b_weight_0",
            data=b"\x00\x40" * 4,
            shape=(4, 1),
        ),
        audio_lora_converter.LoraTensor(
            name="lora_audio_ff1_l1_a_weight_0",
            data=b"\x00\x00" * 4,
            shape=(2, 2),
        ),
    ]

    sidecar = audio_lora_converter.build_lora_tflite_bytes(
        tensors, lora_rank=8, payload_alignment=1024
    )
    model = audio_lora_converter._TfliteModelReader(sidecar)

    self.assertEqual(len(sidecar) % 65536, 0)
    self.assertEqual(model.version(), 3)
    self.assertEqual(model.metadata_buffer("lora_rank"), 8)
    self.assertGreater(model.buffers_length(), model.metadata_buffer("lora_rank"))
    self.assertEqual(model.subgraphs_length(), 1)

    subgraph = model.subgraph(0)
    self.assertEqual(subgraph.tensors_length(), 3)
    tensor0 = subgraph.tensor(0)
    self.assertEqual(tensor0.name(), tensors[0].name)
    self.assertEqual(tensor0.shape(), (2, 4))
    buffer0 = model.buffer(tensor0.buffer_index())
    self.assertEqual(buffer0.size(), len(tensors[0].data))
    self.assertEqual(
        sidecar[buffer0.offset() : buffer0.offset() + buffer0.size()],
        tensors[0].data,
    )

  def test_load_safetensors_without_numpy_converts_f32_to_f16(self):
    header = {
        "tensor": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
    safetensors_bytes = (
        struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      file_path = pathlib.Path(temp_dir) / "adapter_model.safetensors"
      file_path.write_bytes(safetensors_bytes)
      tensor = audio_lora_converter._load_safetensors_without_numpy(file_path)[
          "tensor"
      ]

    self.assertEqual(tensor.shape, (2, 2))
    self.assertEqual(
        (tensor * 2.0).astype("float16").tobytes(),
        struct.pack("<eeee", 2.0, 4.0, 6.0, 8.0),
    )
    self.assertEqual(
        tensor.T.astype("float16").tobytes(),
        struct.pack("<eeee", 1.0, 3.0, 2.0, 4.0),
    )

  def test_rank_inference_uses_peft_key_regex(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
        "base_model.model.language_model.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((3, 4)),
        "base_model.model.language_model.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 3)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {}
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        report = audio_lora_converter.compile_peft_audio_lora(
            peft_lora_dir=temp_path / "adapter",
            output_path=temp_path / "out.tflite",
            peft_key_regex="audio_tower",
        )

      self.assertEqual(report.lora_rank, 2)
      self.assertEqual(
          report.tensors_written,
          (
              "lora_audio_attn_q_a_weight_0",
              "lora_audio_attn_q_b_weight_0",
          ),
      )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config

  def test_target_dtype_controls_sidecar_payload_dtype(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    original_extract_lora_input_specs = audio_lora_converter.extract_lora_input_specs
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    audio_lora_converter.extract_lora_input_specs = lambda _: {
        "lora_audio_attn_q_a_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_a_weight_0",
            shape=(2, 4),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT32,
            byte_size=32,
        ),
        "lora_audio_attn_q_b_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_b_weight_0",
            shape=(4, 2),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT32,
            byte_size=32,
        ),
    }
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        output_path = temp_path / "out.tflite"
        audio_lora_converter.compile_peft_audio_lora(
            peft_lora_dir=temp_path / "adapter",
            output_path=output_path,
            target_tflite_path=temp_path / "target.tflite",
        )
        model = audio_lora_converter._TfliteModelReader(output_path.read_bytes())
        tensor0 = model.subgraph(0).tensor(0)
        buffer0 = model.buffer(tensor0.buffer_index())

      self.assertEqual(tensor0.tensor_type(), audio_lora_converter._TENSOR_TYPE_FLOAT32)
      self.assertEqual(buffer0.size(), 32)
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config
      audio_lora_converter.extract_lora_input_specs = original_extract_lora_input_specs

  def test_fail_on_unmatched_peft_keys(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
        "base_model.model.audio_tower.layers.0.mlp.gate_proj."
        "lora_A.weight": _zeros((2, 4)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with self.assertRaisesRegex(ValueError, "do not map"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              fail_on_unmatched_peft_keys=True,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config

  def test_fail_on_missing_target_inputs(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    original_extract_lora_input_specs = audio_lora_converter.extract_lora_input_specs
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    audio_lora_converter.extract_lora_input_specs = lambda _: {
        "lora_audio_attn_q_a_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_a_weight_0",
            shape=(2, 4),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
        ),
        "lora_audio_attn_q_b_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_b_weight_0",
            shape=(4, 2),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
        ),
        "lora_audio_attn_k_a_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_k_a_weight_0",
            shape=(2, 4),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
        ),
    }
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with self.assertRaisesRegex(ValueError, "no matching PEFT tensor"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              target_tflite_path=temp_path / "target.tflite",
              fail_on_missing_target_inputs=True,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config
      audio_lora_converter.extract_lora_input_specs = original_extract_lora_input_specs

  def test_fail_on_unused_peft_keys(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    original_extract_lora_input_specs = audio_lora_converter.extract_lora_input_specs
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
        "base_model.model.model.audio_tower.output_proj."
        "lora_A.weight": _zeros((2, 4)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    audio_lora_converter.extract_lora_input_specs = lambda _: {
        "lora_audio_attn_q_a_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_a_weight_0",
            shape=(2, 4),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
        ),
        "lora_audio_attn_q_b_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_b_weight_0",
            shape=(4, 2),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
        ),
    }
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with self.assertRaisesRegex(ValueError, "absent from the target graph"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              target_tflite_path=temp_path / "target.tflite",
              fail_on_unused_peft_keys=True,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config
      audio_lora_converter.extract_lora_input_specs = original_extract_lora_input_specs

  def test_fail_on_unconsumed_target_inputs(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    original_extract_lora_input_specs = audio_lora_converter.extract_lora_input_specs
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    audio_lora_converter.extract_lora_input_specs = lambda _: {
        "lora_audio_attn_q_a_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_a_weight_0",
            shape=(2, 4),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
            is_consumed=True,
        ),
        "lora_audio_attn_q_b_weight_0": audio_lora_converter.TargetTensorSpec(
            name="lora_audio_attn_q_b_weight_0",
            shape=(4, 2),
            tensor_type=audio_lora_converter._TENSOR_TYPE_FLOAT16,
            byte_size=16,
            is_consumed=False,
        ),
    }
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with self.assertRaisesRegex(ValueError, "not consumed"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              target_tflite_path=temp_path / "target.tflite",
              fail_on_unconsumed_target_inputs=True,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config
      audio_lora_converter.extract_lora_input_specs = original_extract_lora_input_specs

  def test_default_strict_mode_rejects_language_lora_in_audio_converter(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
        "base_model.model.language_model.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with self.assertRaisesRegex(ValueError, "do not map"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config

  def test_target_without_official_lora_inputs_fails(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        target_tflite = temp_path / "target_without_lora_inputs.tflite"
        target_tflite.write_bytes(
            audio_lora_converter._build_lora_model_header(
                [
                    audio_lora_converter.LoraTensor(
                        name="regular_audio_input", data=b"\x00\x00", shape=(1,)
                    )
                ],
                lora_rank=1,
                payload_offsets=[1024],
            )
        )

        with self.assertRaisesRegex(ValueError, "no LiteRT-LM LoRA input tensors"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              target_tflite_path=target_tflite,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config

  def test_lora_sidecar_cannot_be_used_as_target_graph(self):
    original_load_safetensors = audio_lora_converter._load_safetensors
    original_read_adapter_config = audio_lora_converter._read_adapter_config
    audio_lora_converter._load_safetensors = lambda _: {
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_A.weight": _zeros((2, 4)),
        "base_model.model.audio_tower.layers.0.self_attn.q_proj."
        "lora_B.weight": _zeros((4, 2)),
    }
    audio_lora_converter._read_adapter_config = lambda _: {"r": 2, "lora_alpha": 2}
    try:
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        sidecar_as_target = temp_path / "sidecar_as_target.tflite"
        sidecar_as_target.write_bytes(
            audio_lora_converter.build_lora_tflite_bytes(
                [
                    audio_lora_converter.LoraTensor(
                        name="lora_audio_attn_q_a_weight_0",
                        data=b"\x00\x00" * 8,
                        shape=(2, 4),
                    )
                ],
                lora_rank=2,
                payload_alignment=1024,
            )
        )

        with self.assertRaisesRegex(ValueError, "no LiteRT-LM LoRA input tensors"):
          audio_lora_converter.compile_peft_audio_lora(
              peft_lora_dir=temp_path / "adapter",
              output_path=temp_path / "out.tflite",
              target_tflite_path=sidecar_as_target,
          )
    finally:
      audio_lora_converter._load_safetensors = original_load_safetensors
      audio_lora_converter._read_adapter_config = original_read_adapter_config

  def test_invalid_litert_lora_name_fails(self):
    with self.assertRaises(ValueError):
      audio_lora_converter.build_lora_tflite_bytes(
          [
              audio_lora_converter.LoraTensor(
                  name="not_a_lora_tensor", data=b"\x00\x00", shape=(1,)
              )
          ],
          lora_rank=1,
      )


if __name__ == "__main__":
  unittest.main()
