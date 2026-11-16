# Copyright 2025 The ODML Authors.
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

"""Python API for LiteRT-LM schema."""

import sys as _sys
import warnings as _warnings

from . import litertlm_builder
from . import litertlm_peek

# Python 3.10 reached upstream end-of-life on 2026-10-01 (PEP 619). Support
# for it is deprecated and will be removed in the next release. FutureWarning
# (rather than DeprecationWarning) so end users see it even when this package
# is imported transitively (PEP 565 hides DeprecationWarning outside __main__).
if _sys.version_info < (3, 11):
  _warnings.warn(
      "litert-lm-builder support for Python 3.10 is deprecated and will be"
      " removed in the next release, following Python 3.10's upstream"
      " end-of-life (https://peps.python.org/pep-0619/). Please upgrade to"
      " Python 3.11 or newer.",
      FutureWarning,
      stacklevel=2,
  )

LitertLmFileBuilder = litertlm_builder.LitertLmFileBuilder
Metadata = litertlm_builder.Metadata
DType = litertlm_builder.DType
Backend = litertlm_builder.Backend
peek_litertlm_file = litertlm_peek.peek_litertlm_file
unpack = litertlm_builder.unpack
unpack_litertlm_file = litertlm_builder.unpack_litertlm_file
pack = litertlm_builder.pack
pack_litertlm_file = litertlm_builder.pack_litertlm_file

__all__ = [
    "LitertLmFileBuilder",
    "Metadata",
    "DType",
    "Backend",
    "peek_litertlm_file",
    "unpack",
    "unpack_litertlm_file",
    "pack",
    "pack_litertlm_file",
]
