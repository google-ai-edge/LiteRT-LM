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

    # Test uppercase (verifies case_sensitive=False)
    result = runner.invoke(dummy_cmd, ['--enable-speculative-decoding', 'TRUE'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn('speculative_decoding: True', result.output)

    # Test invalid value
    result = runner.invoke(
        dummy_cmd, ['--enable-speculative-decoding', 'invalid']
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn(
        "Error: Invalid value for '--enable-speculative-decoding'",
        result.output,
    )


if __name__ == '__main__':
  absltest.main()
