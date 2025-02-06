# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from docsassist.schema import ApplicationType, CoreSettings, RAGType

from .common.globals import (
    GlobalPredictionEnvironmentPlatforms,
    GlobalRuntimeEnvironment,
)
from .common.schema import (
    PredictionEnvironmentArgs,
    UseCaseArgs,
)
from .common.stack import get_stack

project_name = get_stack()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.absolute()

# Core settings are overridable by environment variables; env values take precedence
core = CoreSettings(
    rag_documents=str(
        PROJECT_ROOT / "assets" / "datarobot_english_documentation_docsassist.zip"
    ),
    rag_type=RAGType.DR,
    application_type=ApplicationType.DR,
)

runtime_environment_moderations = GlobalRuntimeEnvironment.PYTHON_311_MODERATIONS.value

default_prediction_server_id: Optional[str] = None

prediction_environment_args = PredictionEnvironmentArgs(
    resource_name=f"Guarded RAG Prediction Environment [{project_name}]",
    platform=GlobalPredictionEnvironmentPlatforms.DATAROBOT_SERVERLESS,
).model_dump(mode="json", exclude_none=True)

use_case_args = UseCaseArgs(
    resource_name=f"Guarded RAG Use Case [{project_name}]",
    description="Use case for Guarded RAG Assistant application",
).model_dump(exclude_none=True)
