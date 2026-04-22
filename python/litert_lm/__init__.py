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

"""LiteRT LM is a library for running GenAI models on devices."""

from .interfaces import AbstractBenchmark
from .interfaces import AbstractConversation
from .interfaces import AbstractEngine
from .interfaces import AbstractSession
from .interfaces import Backend
from .interfaces import BenchmarkInfo
from .interfaces import Responses
from .interfaces import SamplerConfig
from .interfaces import Tool
from .interfaces import ToolEventHandler
from .litert_lm_ext import _Benchmark  # pytype: disable=import-error
from .litert_lm_ext import _Engine  # pytype: disable=import-error
from .litert_lm_ext import Benchmark  # pytype: disable=import-error
from .litert_lm_ext import BenchmarkInfo as _BenchmarkInfo  # pytype: disable=import-error
from .litert_lm_ext import Conversation  # pytype: disable=import-error
from .litert_lm_ext import Engine  # pytype: disable=import-error
from .litert_lm_ext import LogSeverity  # pytype: disable=import-error
from .litert_lm_ext import Session  # pytype: disable=import-error
from .litert_lm_ext import set_min_log_severity  # pytype: disable=import-error
from .tools import tool_from_function

# Because the C++ class is created by nanobind and the Python
# interface is a standard ABC, they cannot easily share a formal
# inheritance tree across the C++/Python boundary. Instead, we use the
# register() method in the package's entry point to set the
# relationship.
AbstractEngine.register(_Engine)
AbstractConversation.register(Conversation)
AbstractBenchmark.register(_Benchmark)
BenchmarkInfo.register(_BenchmarkInfo)
AbstractSession.register(Session)

__all__ = (
    "AbstractBenchmark",
    "AbstractConversation",
    "AbstractEngine",
    "AbstractSession",
    "Backend",
    "Benchmark",
    "BenchmarkInfo",
    "Conversation",
    "Engine",
    "LogSeverity",
    "Responses",
    "SamplerConfig",
    "Session",
    "Tool",
    "ToolEventHandler",
    "set_min_log_severity",
    "tool_from_function",
)
