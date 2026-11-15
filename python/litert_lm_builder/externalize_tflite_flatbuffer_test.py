# Copyright 2026 Google LLC.
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

from absl.testing import absltest
import flatbuffers
import numpy as np

from litert_lm_builder import externalize_tflite_flatbuffer
from litert_lm_builder import tflite_schema_py_generated as schema


def _make_float32_tensor(shape: list[int], buffer_index: int) -> schema.TensorT:
  """Returns a FLOAT32 tensor with `shape` backed by `buffer_index`."""
  tensor = schema.TensorT()
  tensor.shape = shape
  tensor.type = externalize_tflite_flatbuffer._enum_value(
      schema.TensorType, "FLOAT32"
  )
  tensor.buffer = buffer_index
  return tensor


class ExternalizeTfliteFlatbufferTest(absltest.TestCase):

  def test_depthwise_weights_require_unit_depth_multiplier(self):
    weight_data = b"\x00\x00\x80\x3f" * (3 * 3 * 128)
    bias_data = b"\x00" * (128 * 4)

    def make_tensor(shape, buffer_index):
      tensor = schema.TensorT()
      tensor.shape = shape
      tensor.type = schema.TensorType.FLOAT32
      tensor.buffer = buffer_index
      return tensor

    for multipliers in ((1,), (2,), (128,), (1, 128), (128, 1)):
      with self.subTest(multipliers=multipliers):
        root = pathlib.Path(self.create_tempdir().full_path)
        model = schema.ModelT()
        model.version = 3
        code = schema.OperatorCodeT()
        code.builtinCode = schema.BuiltinOperator.DEPTHWISE_CONV_2D
        code.deprecatedBuiltinCode = code.builtinCode
        model.operatorCodes = [code]
        model.buffers = [schema.BufferT() for _ in range(3)]
        model.buffers[1].data = weight_data
        model.buffers[2].data = bias_data

        subgraph = schema.SubGraphT()
        subgraph.tensors = [
            make_tensor([1, 3, 3, 128], 1),
            make_tensor([128], 2),
        ]
        subgraph.inputs = []
        subgraph.outputs = []
        subgraph.operators = []
        for multiplier in multipliers:
          input_index = len(subgraph.tensors)
          subgraph.tensors.extend([
              make_tensor([1, 3, 3, 128 // multiplier], 0),
              make_tensor([1, 1, 1, 128], 0),
          ])
          subgraph.inputs.append(input_index)
          subgraph.outputs.append(input_index + 1)
          op = schema.OperatorT()
          op.opcodeIndex = 0
          op.inputs = [input_index, 0, 1]
          op.outputs = [input_index + 1]
          op.builtinOptionsType = schema.BuiltinOptions.DepthwiseConv2DOptions
          options = schema.DepthwiseConv2DOptionsT()
          options.depthMultiplier = multiplier
          options.padding = schema.Padding.VALID
          options.strideH = options.strideW = 1
          options.dilationHFactor = options.dilationWFactor = 1
          op.builtinOptions = options
          subgraph.operators.append(op)
        model.subgraphs = [subgraph]
        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
        input_model = root / "input.tflite"
        input_model.write_bytes(bytes(builder.Output()))

        output_dir = root / "output"
        externalize_tflite_flatbuffer.externalize(
            input_model=input_model,
            output_dir=output_dir,
            group_name="tflite_weights",
            num_elements_threshold=256,
        )
        processed = schema.ModelT.InitFromPackedBuf(
            (output_dir / "model.tflite").read_bytes(), 0
        )
        weight = processed.subgraphs[0].tensors[0]
        if all(multiplier == 1 for multiplier in multipliers):
          self.assertNotEqual(weight.externalBuffer, 0)
          self.assertEqual(weight.buffer, 0)
          self.assertEqual(
              (output_dir / "tflite_weights").read_bytes(), weight_data
          )
        else:
          self.assertEqual(weight.externalBuffer, 0)
          self.assertEqual(
              bytes(processed.buffers[weight.buffer].data), weight_data
          )
          self.assertFalse(processed.externalBuffers)
          self.assertEqual((output_dir / "tflite_weights").read_bytes(), b"")
        self.assertEqual(bytes(processed.buffers[2].data), bias_data)

  def test_preserves_existing_weight_blob(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model = root / "input.tflite"
    existing_weights = root / "existing_weights"
    existing_weights.write_bytes(b"existing external weights")

    model = schema.ModelT()
    model.version = 3
    model.operatorCodes = []
    model.subgraphs = []
    model.buffers = []
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    input_model.write_bytes(bytes(builder.Output()))

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=256,
        existing_weights=existing_weights,
    )

    self.assertEqual(
        (output_dir / "model.tflite").read_bytes(), input_model.read_bytes()
    )
    self.assertEqual(
        (output_dir / "tflite_weights").read_bytes(),
        existing_weights.read_bytes(),
    )

  def _write_appended_buffer_model(
      self, root: pathlib.Path, empty_buffer: bool = False
  ) -> tuple[pathlib.Path, bytes, bytes]:
    """Writes a FullyConnected model whose buffers use appended data.

    Args:
      root: Directory to write `input.tflite` into.
      empty_buffer: If true, also adds a constant tensor whose buffer is a
        zero-length appended payload (`offset > 1`, `size == 0`).

    Returns:
      The model path, the weight bytes, and the bias bytes.
    """
    input_model = root / "input.tflite"
    weight_data = bytes(range(64))
    bias_data = b"bias data bytes!"

    def make_tensor(shape, buffer_index):
      tensor = schema.TensorT()
      tensor.shape = shape
      tensor.type = externalize_tflite_flatbuffer._enum_value(
          schema.TensorType, "FLOAT32"
      )
      tensor.buffer = buffer_index
      return tensor

    model = schema.ModelT()
    model.version = 3
    operator_code = schema.OperatorCodeT()
    operator_code.builtinCode = externalize_tflite_flatbuffer._enum_value(
        schema.BuiltinOperator, "FULLY_CONNECTED"
    )
    operator_code.deprecatedBuiltinCode = operator_code.builtinCode
    model.operatorCodes = [operator_code]

    operator = schema.OperatorT()
    operator.opcodeIndex = 0
    operator.inputs = [0, 1, 2]
    operator.outputs = [3]
    subgraph = schema.SubGraphT()
    subgraph.tensors = [
        make_tensor([1, 4], 0),
        make_tensor([4, 4], 1),
        make_tensor([4], 2),
        make_tensor([1, 4], 0),
    ]
    subgraph.inputs = [0]
    subgraph.outputs = [3]
    subgraph.operators = [operator]
    model.subgraphs = [subgraph]
    model.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    model.buffers[1].offset = 1
    model.buffers[1].size = len(weight_data)
    model.buffers[2].offset = 1
    model.buffers[2].size = len(bias_data)
    if empty_buffer:
      subgraph.tensors.append(make_tensor([0], 3))
      model.buffers.append(schema.BufferT())
      model.buffers[3].offset = 1
      model.buffers[3].size = 0

    def pack_model():
      builder = flatbuffers.Builder(1024)
      builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
      return bytes(builder.Output())

    placeholder = pack_model()
    weight_offset = (len(placeholder) + 15) // 16 * 16
    bias_offset = (weight_offset + len(weight_data) + 15) // 16 * 16
    model.buffers[1].offset = weight_offset
    model.buffers[2].offset = bias_offset
    if empty_buffer:
      model.buffers[3].offset = bias_offset + len(bias_data)
    packed_model = pack_model()
    self.assertEqual(len(packed_model), len(placeholder))
    input_bytes = bytearray(packed_model)
    input_bytes.extend(b"\0" * (weight_offset - len(input_bytes)))
    input_bytes.extend(weight_data)
    input_bytes.extend(b"\0" * (bias_offset - len(input_bytes)))
    input_bytes.extend(bias_data)
    input_model.write_bytes(input_bytes)
    return input_model, weight_data, bias_data

  def test_externalizes_appended_tflite_buffers_without_losing_bias(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model, weight_data, bias_data = self._write_appended_buffer_model(
        root
    )

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=4,
    )

    self.assertEqual((output_dir / "tflite_weights").read_bytes(), weight_data)
    processed_model = schema.ModelT.InitFromPackedBuf(
        (output_dir / "model.tflite").read_bytes(), 0
    )
    weight_tensor = processed_model.subgraphs[0].tensors[1]
    self.assertNotEqual(weight_tensor.externalBuffer, 0)
    self.assertEqual(weight_tensor.buffer, 0)
    processed_bias = processed_model.buffers[2]
    self.assertEqual(bytes(processed_bias.data), bias_data)
    self.assertEqual(processed_bias.offset, 0)
    self.assertEqual(processed_bias.size, 0)

  def test_reappends_large_unexternalized_buffers_when_exceeding_inline_limit(
      self,
  ):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model = root / "input.tflite"
    fc_weight_data = bytes(range(64))
    unexternalized_data = b"moe_expert_weights!" * 8

    model = schema.ModelT()
    model.version = 3
    fc_code = schema.OperatorCodeT()
    fc_code.builtinCode = externalize_tflite_flatbuffer._enum_value(
        schema.BuiltinOperator, "FULLY_CONNECTED"
    )
    fc_code.deprecatedBuiltinCode = fc_code.builtinCode
    custom_code = schema.OperatorCodeT()
    custom_code.builtinCode = externalize_tflite_flatbuffer._enum_value(
        schema.BuiltinOperator, "CUSTOM"
    )
    custom_code.deprecatedBuiltinCode = custom_code.builtinCode
    custom_code.customCode = "custom:moe"
    model.operatorCodes = [fc_code, custom_code]

    fc_op = schema.OperatorT()
    fc_op.opcodeIndex = 0
    fc_op.inputs = [0, 1]
    fc_op.outputs = [3]
    custom_op = schema.OperatorT()
    custom_op.opcodeIndex = 1
    custom_op.inputs = [3, 2]
    custom_op.outputs = [4]

    subgraph = schema.SubGraphT()
    subgraph.tensors = [
        _make_float32_tensor([1, 4], 0),
        _make_float32_tensor([4, 4], 1),
        _make_float32_tensor([4, 4], 2),
        _make_float32_tensor([1, 4], 0),
        _make_float32_tensor([1, 4], 0),
    ]
    subgraph.inputs = [0]
    subgraph.outputs = [4]
    subgraph.operators = [fc_op, custom_op]
    model.subgraphs = [subgraph]
    model.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    model.buffers[1].offset = 1
    model.buffers[1].size = len(fc_weight_data)
    model.buffers[2].offset = 1
    model.buffers[2].size = len(unexternalized_data)

    def pack_model():
      builder = flatbuffers.Builder(1024)
      builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
      return bytes(builder.Output())

    placeholder = pack_model()
    fc_offset = (len(placeholder) + 15) // 16 * 16
    moe_offset = (fc_offset + len(fc_weight_data) + 15) // 16 * 16
    model.buffers[1].offset = fc_offset
    model.buffers[2].offset = moe_offset
    packed_model = pack_model()
    input_bytes = bytearray(packed_model)
    input_bytes.extend(b"\0" * (fc_offset - len(input_bytes)))
    input_bytes.extend(fc_weight_data)
    input_bytes.extend(b"\0" * (moe_offset - len(input_bytes)))
    input_bytes.extend(unexternalized_data)
    input_model.write_bytes(input_bytes)

    output_dir = root / "output"
    with mock.patch.object(
        externalize_tflite_flatbuffer, "_MAX_INLINE_BUFFER_BYTES", 32
    ):
      externalize_tflite_flatbuffer.externalize(
          input_model=input_model,
          output_dir=output_dir,
          group_name="tflite_weights",
          num_elements_threshold=4,
      )

    self.assertEqual(
        (output_dir / "tflite_weights").read_bytes(), fc_weight_data
    )
    output_bytes = (output_dir / "model.tflite").read_bytes()
    processed_model = schema.ModelT.InitFromPackedBuf(output_bytes, 0)
    moe_buf = processed_model.buffers[2]
    self.assertIsNone(moe_buf.data)
    self.assertGreater(moe_buf.offset, 0)
    self.assertLen(unexternalized_data, moe_buf.size)
    self.assertEqual(moe_buf.offset % 16, 0)
    self.assertEqual(
        output_bytes[moe_buf.offset : moe_buf.offset + moe_buf.size],
        unexternalized_data,
    )

  def test_accepts_zero_length_appended_buffer(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model, weight_data, bias_data = self._write_appended_buffer_model(
        root, empty_buffer=True
    )

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=4,
    )

    self.assertEqual((output_dir / "tflite_weights").read_bytes(), weight_data)
    processed_model = schema.ModelT.InitFromPackedBuf(
        (output_dir / "model.tflite").read_bytes(), 0
    )
    self.assertEqual(bytes(processed_model.buffers[2].data), bias_data)
    self.assertEqual(processed_model.subgraphs[0].tensors[4].buffer, 3)
    empty = processed_model.buffers[3]
    # Still a present-but-empty vector, like TFLite's view of the input.
    self.assertIsNotNone(empty.data)
    self.assertEmpty(empty.data)
    self.assertEqual(empty.offset, 0)
    self.assertEqual(empty.size, 0)

  def test_rejects_sentinel_offset_with_appended_size(self):
    for offset in (0, 1):
      with self.subTest(offset=offset):
        model = schema.ModelT()
        model.buffers = [schema.BufferT(), schema.BufferT()]
        model.buffers[1].offset = offset
        model.buffers[1].size = 4
        with self.assertRaisesRegex(ValueError, "invalid external range"):
          externalize_tflite_flatbuffer._materialize_appended_buffer_data(
              model, bytes(16)
          )

  def test_copies_model_unchanged_when_nothing_is_externalized(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model, _, _ = self._write_appended_buffer_model(root)

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=1024,
    )

    self.assertEqual(
        (output_dir / "model.tflite").read_bytes(), input_model.read_bytes()
    )
    self.assertEqual((output_dir / "tflite_weights").read_bytes(), b"")

  def test_materialized_buffers_pack_like_generated_code(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model, weight_data, bias_data = self._write_appended_buffer_model(
        root
    )
    input_bytes = input_model.read_bytes()
    model = schema.ModelT.InitFromPackedBuf(input_bytes, 0)

    externalize_tflite_flatbuffer._materialize_appended_buffer_data(
        model, input_bytes
    )

    for buffer, expected in (
        (model.buffers[1], weight_data),
        (model.buffers[2], bias_data),
    ):
      # BufferT.Pack only takes its vectorized path for exact ndarrays.
      self.assertIs(type(buffer.data), np.ndarray)
      self.assertEqual(buffer.data.dtype, np.uint8)
      self.assertEqual(buffer.data.tobytes(), expected)
      self.assertEqual(buffer.offset, 0)
      self.assertEqual(buffer.size, 0)

    def pack(model_t):
      builder = flatbuffers.Builder(1024)
      builder.Finish(model_t.Pack(builder), file_identifier=b"TFL3")
      return bytes(builder.Output())

    fast_bytes = pack(model)
    # The generated BufferT.Pack serializes `bytes` data one byte at a time;
    # the numpy fast path must produce the same output.
    reference = schema.ModelT.InitFromPackedBuf(input_bytes, 0)
    for index, data in ((1, weight_data), (2, bias_data)):
      reference.buffers[index].data = data
      reference.buffers[index].offset = 0
      reference.buffers[index].size = 0
    self.assertEqual(fast_bytes, pack(reference))

  def test_materializes_appended_buffers_without_numpy(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model, weight_data, bias_data = self._write_appended_buffer_model(
        root, empty_buffer=True
    )

    output_dir = root / "output"
    with mock.patch.object(externalize_tflite_flatbuffer, "np", None):
      externalize_tflite_flatbuffer.externalize(
          input_model=input_model,
          output_dir=output_dir,
          group_name="tflite_weights",
          num_elements_threshold=4,
      )

    self.assertEqual((output_dir / "tflite_weights").read_bytes(), weight_data)
    processed_model = schema.ModelT.InitFromPackedBuf(
        (output_dir / "model.tflite").read_bytes(), 0
    )
    self.assertEqual(bytes(processed_model.buffers[2].data), bias_data)
    self.assertEmpty(processed_model.buffers[3].data)

  def _write_fc_model(self, path: pathlib.Path, weights: tuple[bytes, ...]):
    """Writes a FLOAT32 model chaining one FullyConnected op per [4,4] weight."""
    model = schema.ModelT()
    model.version = 3
    code = schema.OperatorCodeT()
    code.builtinCode = externalize_tflite_flatbuffer._enum_value(
        schema.BuiltinOperator, "FULLY_CONNECTED"
    )
    code.deprecatedBuiltinCode = code.builtinCode
    model.operatorCodes = [code]
    model.buffers = [schema.BufferT()]
    subgraph = schema.SubGraphT()
    subgraph.tensors = [_make_float32_tensor([1, 4], 0)]
    subgraph.inputs = [0]
    subgraph.operators = []
    activation = 0
    for weight in weights:
      buffer = schema.BufferT()
      buffer.data = weight
      model.buffers.append(buffer)
      subgraph.tensors.append(
          _make_float32_tensor([4, 4], len(model.buffers) - 1)
      )
      subgraph.tensors.append(_make_float32_tensor([1, 4], 0))
      op = schema.OperatorT()
      op.opcodeIndex = 0
      op.inputs = [activation, len(subgraph.tensors) - 2]
      op.outputs = [len(subgraph.tensors) - 1]
      subgraph.operators.append(op)
      activation = len(subgraph.tensors) - 1
    subgraph.outputs = [activation]
    model.subgraphs = [subgraph]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    path.write_bytes(bytes(builder.Output()))

  def _externalize(self, root: pathlib.Path, name: str, weights):
    source = root / f"{name}.tflite"
    self._write_fc_model(source, weights)
    output_dir = root / name
    externalize_tflite_flatbuffer.externalize(
        input_model=source,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=4,
    )
    return output_dir / "model.tflite", output_dir / "tflite_weights"

  def test_collect_external_weight_digests(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    first, second = bytes(range(64)), b"B" * 64
    model, weights = self._externalize(root, "m", (first, second, first))

    placements = externalize_tflite_flatbuffer.collect_external_weight_digests(
        model, weights, "tflite_weights"
    )

    digest = externalize_tflite_flatbuffer._weight_digest
    self.assertEqual(
        placements, {digest(first): (0, 64), digest(second): (64, 64)}
    )
    self.assertEmpty(
        externalize_tflite_flatbuffer.collect_external_weight_digests(
            model, weights, "other_group"
        )
    )

  def test_redirect_points_shared_payload_at_other_group_and_compacts(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    table, own = bytes(range(64)), b"E" * 64
    model, weights = self._externalize(root, "embedder", (table, own))
    owner_offset = 128  # Where the owner model stores the same payload.

    result = externalize_tflite_flatbuffer.redirect_external_weights(
        input_model=model,
        input_weights=weights,
        output_dir=root / "redirected",
        group_name="tflite_weights",
        redirects={
            externalize_tflite_flatbuffer._weight_digest(table): (
                "tflite_weights:tf_lite_prefill_decode",
                owner_offset,
                len(table),
            )
        },
    )

    self.assertEqual(result.redirected_buffers, 1)
    self.assertEqual(result.removed_bytes, 64)
    # Only the unshared payload stays in the model's own blob, at offset 0.
    self.assertEqual((root / "redirected" / "tflite_weights").read_bytes(), own)
    processed = schema.ModelT.InitFromPackedBuf(
        (root / "redirected" / "model.tflite").read_bytes(), 0
    )
    groups = [
        g.name.decode() if isinstance(g.name, bytes) else g.name
        for g in processed.externalBufferGroups
    ]
    by_id = {b.id: b for b in processed.externalBuffers}
    table_ref = by_id[processed.subgraphs[0].tensors[1].externalBuffer]
    own_ref = by_id[processed.subgraphs[0].tensors[3].externalBuffer]
    self.assertEqual(
        groups[table_ref.group], "tflite_weights:tf_lite_prefill_decode"
    )
    self.assertEqual((table_ref.offset, table_ref.length), (owner_offset, 64))
    self.assertEqual(groups[own_ref.group], "tflite_weights")
    self.assertEqual((own_ref.offset, own_ref.length), (0, 64))

  def test_redirect_without_match_copies_inputs(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    model, weights = self._externalize(root, "m", (b"A" * 64,))

    result = externalize_tflite_flatbuffer.redirect_external_weights(
        input_model=model,
        input_weights=weights,
        output_dir=root / "out",
        group_name="tflite_weights",
        redirects={
            externalize_tflite_flatbuffer._weight_digest(b"Z" * 64): (
                "tflite_weights:x",
                0,
                64,
            )
        },
    )

    self.assertEqual(result.redirected_buffers, 0)
    self.assertEqual(
        (root / "out" / "model.tflite").read_bytes(), model.read_bytes()
    )
    self.assertEqual(
        (root / "out" / "tflite_weights").read_bytes(), weights.read_bytes()
    )

  def test_redirect_rejects_own_group_target(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    payload = b"A" * 64
    model, weights = self._externalize(root, "m", (payload,))
    with self.assertRaisesRegex(ValueError, "must differ"):
      externalize_tflite_flatbuffer.redirect_external_weights(
          input_model=model,
          input_weights=weights,
          output_dir=root / "out",
          group_name="tflite_weights",
          redirects={
              externalize_tflite_flatbuffer._weight_digest(payload): (
                  "tflite_weights",
                  0,
                  64,
              )
          },
      )


if __name__ == "__main__":
  absltest.main()
