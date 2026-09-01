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

from absl.testing import absltest
from absl.testing import parameterized
import click
from click.testing import CliRunner

from litert_lm_cli import common


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero_bytes', 0, '0B'),
      ('half_kib', 512, '512B'),
      ('almost_kib', 1023, '1023B'),
      ('one_kib', 1024, '1.0KiB'),
      ('one_and_half_kib', 1536, '1.5KiB'),
      ('one_mib', 1048576, '1.0MiB'),
      ('one_gib', 1073741824, '1.0GiB'),
      ('one_tib', 1099511627776, '1.0TiB'),
      ('one_pib', 1125899906842624, '1.0PiB'),
      ('exceed_pib', 1125899906842624 * 1024, '1024.0PiB'),
  )
  def test_size_string_from_bytes(self, size_in_bytes, expected):
    # pylint: disable=protected-access
    self.assertEqual(common._size_string_from_bytes(size_in_bytes), expected)

  @parameterized.named_parameters(
      ('none_size', None, ''),
      ('valid_size', 1024, ' (1.0KiB)'),
  )
  def test_download_size_suffix(self, total_size, expected):
    self.assertEqual(common.download_size_suffix(total_size), expected)

  @parameterized.named_parameters(
      ('with_total_50_pct', 50, 100, '50%'),
      ('with_total_0_pct', 0, 100, '0%'),
      ('no_total_half_kib', 500, None, '0.5 KiB'),
      ('no_total_one_kib', 1024, None, '1.0 KiB'),
      ('no_total_one_mib_kib', 1048576, None, '1024.0 KiB'),
      ('no_total_one_mib', 1048577, None, '1.0 MiB'),
  )
  def test_format_download_progress(
      self, current_pos_bytes, total_size, expected
  ):
    self.assertEqual(
        common.format_download_progress(current_pos_bytes, total_size), expected
    )

  @parameterized.named_parameters(
      ('valid_int', '12345', 12345),
      ('none_input', None, None),
      ('invalid_str', 'invalid', None),
  )
  def test_parse_total_size(self, content_length, expected):
    self.assertEqual(common.parse_total_size(content_length), expected)

  @parameterized.named_parameters(
      ('none_cache', None, ''),
      ('disk_cache', 'disk', ''),
      ('memory_cache', 'memory', ':memory'),
      ('no_cache', 'no', ':nocache'),
  )
  def test_cache_dir_value_from_cache_mode(self, cache, expected):
    self.assertEqual(common.cache_dir_value_from_cache_mode(cache), expected)

  def test_cache_dir_value_from_cache_mode_invalid(self):
    with self.assertRaises(ValueError):
      common.cache_dir_value_from_cache_mode('invalid')


class CommonInferenceOptionsTest(parameterized.TestCase):

  def test_enable_speculative_decoding_case_insensitive(self):
    @click.command()
    @common.common_inference_options
    def dummy_cmd(**kwargs):
      click.echo(
          f"speculative_decoding: {kwargs.get('enable_speculative_decoding')}"
      )

    runner = CliRunner()

    # Test lowercase
    result = runner.invoke(dummy_cmd, ['--enable-speculative-decoding', 'true'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)
    self.assertIn(
        "Warning: '--enable-speculative-decoding' is deprecated",
        result.output,
    )

    # Test uppercase (verifies case_sensitive=False)
    result = runner.invoke(dummy_cmd, ['--enable-speculative-decoding', 'TRUE'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)
    self.assertIn(
        "Warning: '--enable-speculative-decoding' is deprecated",
        result.output,
    )

    # Test hidden from --help
    result = runner.invoke(dummy_cmd, ['--help'])
    self.assertEqual(result.exit_code, 0)
    self.assertNotIn('--enable-speculative-decoding', result.output)

    # Test invalid value
    result = runner.invoke(
        dummy_cmd, ['--enable-speculative-decoding', 'invalid']
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn(
        "Error: Invalid value for '--enable-speculative-decoding'",
        result.output,
    )

  def test_speculative_decoding_options(self):
    @click.command()
    @common.common_inference_options
    def dummy_cmd(**kwargs):
      click.echo(f"speculative_decoding: {kwargs.get('speculative_decoding')}")

    runner = CliRunner()

    # Flag mode (no value)
    result = runner.invoke(dummy_cmd, ['--speculative-decoding'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)

    # Choice true mode (lowercase)
    result = runner.invoke(dummy_cmd, ['--speculative-decoding', 'true'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)

    # Choice true mode (uppercase, case insensitive)
    result = runner.invoke(dummy_cmd, ['--speculative-decoding', 'TRUE'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)

    # Choice false mode
    result = runner.invoke(dummy_cmd, ['--speculative-decoding', 'false'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: False', result.output)

    # Not set mode (default is None)
    result = runner.invoke(dummy_cmd, [])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: None', result.output)

    # Invalid value
    result = runner.invoke(dummy_cmd, ['--speculative-decoding', 'invalid'])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn(
        "Error: Invalid value for '--speculative-decoding'",
        result.output,
    )

  def test_ringbuffers_local_attention_options(self):
    @click.command()
    @common.common_inference_options
    def dummy_cmd(**kwargs):
      click.echo(
          f'ringbuffers_local_attention:'
          f" {kwargs.get('ringbuffers_local_attention')}"
      )

    runner = CliRunner()

    # Flag mode (no value)
    result = runner.invoke(dummy_cmd, ['--ringbuffers-local-attention'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('ringbuffers_local_attention: True', result.output)

    # Choice true mode
    result = runner.invoke(dummy_cmd, ['--ringbuffers-local-attention', 'true'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('ringbuffers_local_attention: True', result.output)

    # Choice false mode
    result = runner.invoke(
        dummy_cmd, ['--ringbuffers-local-attention', 'false']
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn('ringbuffers_local_attention: False', result.output)

    # Not set mode (default is None)
    result = runner.invoke(dummy_cmd, [])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('ringbuffers_local_attention: None', result.output)

  def test_gpu_decode_steps_per_sync_options(self):
    @click.command()
    @common.common_inference_options
    def dummy_cmd(**kwargs):
      val = kwargs.get('gpu_decode_steps_per_sync')
      click.echo(f'gpu_decode_steps_per_sync: {val}')

    runner = CliRunner()

    # Valid int value
    result = runner.invoke(dummy_cmd, ['--gpu-decode-steps-per-sync', '4'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('gpu_decode_steps_per_sync: 4', result.output)

    # Not set mode (default is None)
    result = runner.invoke(dummy_cmd, [])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('gpu_decode_steps_per_sync: None', result.output)

    # Invalid value (< 1)
    result = runner.invoke(dummy_cmd, ['--gpu-decode-steps-per-sync', '0'])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn(
        "Error: Invalid value for '--gpu-decode-steps-per-sync'",
        result.output,
    )

  def test_enable_ynnpack_options(self):
    @click.command()
    @common.common_inference_options
    def dummy_cmd(**kwargs):
      click.echo(f"enable_ynnpack: {kwargs.get('enable_ynnpack')}")

    runner = CliRunner()

    # Flag mode (no value)
    result = runner.invoke(dummy_cmd, ['--enable-ynnpack'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('enable_ynnpack: True', result.output)

    # Explicit value: true
    result = runner.invoke(dummy_cmd, ['--enable-ynnpack', 'true'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('enable_ynnpack: True', result.output)

    # Explicit value: false
    result = runner.invoke(dummy_cmd, ['--enable-ynnpack', 'false'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('enable_ynnpack: False', result.output)

    # Not set mode (default is None)
    result = runner.invoke(dummy_cmd, [])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('enable_ynnpack: None', result.output)

    # Test hidden from --help
    result = runner.invoke(dummy_cmd, ['--help'])
    self.assertEqual(result.exit_code, 0)
    self.assertNotIn('--enable-ynnpack', result.output)


if __name__ == '__main__':
  absltest.main()
