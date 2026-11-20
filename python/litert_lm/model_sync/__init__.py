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
"""Model synchronization and differential update API.

Updates a local `.litertlm` model to a newer remote version by downloading only
the sections that changed and reusing the unchanged ones from the local model.

Typical usage example:

  from litert_lm import model_sync

  local_model = "/path/to/model.litertlm"
  url = "https://example.com/model.litertlm"
  with model_sync.plan_sync(local_model, url) as plan:
    print(plan.summary())  # The sections to reuse and to download.
    if not plan.is_up_to_date:
      model_sync.update_model(
          local_path=local_model,
          output_path="/path/to/output_model.litertlm",
          remote=url,
      )

To update the model in place, pass the same path as `local_path` and
`output_path`. `cache_sync` keeps a model up to date inside a cache directory
instead, and returns the cached copy when the remote model can't be reached.
"""

# pylint: disable=g-importing-member

from ._model_sync import cache_sync
from ._model_sync import create_file_range_fetcher
from ._model_sync import create_http_range_fetcher
from ._model_sync import create_range_fetcher
from ._model_sync import plan_sync
from ._model_sync import RangeFetcher
from ._model_sync import SyncOptions
from ._model_sync import SyncPlan
from ._model_sync import update_model

__all__ = (
    "RangeFetcher",
    "SyncOptions",
    "SyncPlan",
    "cache_sync",
    "create_file_range_fetcher",
    "create_http_range_fetcher",
    "create_range_fetcher",
    "plan_sync",
    "update_model",
)
