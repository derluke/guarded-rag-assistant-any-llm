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

import datarobot as dr
from datarobotx.idp.custom_application_source_version import (
    _unsafe_get_or_create_custom_application_source_version_from_previous,
)
import pulumi

from docsassist.i18n import LanguageCode, LocaleSettings
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.schema import ApplicationSourceArgs
from infra.settings_main import PROJECT_ROOT, project_name


def ensure_app_settings(app_id: str) -> None:
    try:
        dr.client.get_client().patch(
            f"customApplications/{app_id}/",
            json={"allowAutoStopping": True},
            timeout=60,
        )
    except Exception:
        pulumi.warn("Could not enable autostopping for the Application")


_application_path = PROJECT_ROOT / "frontend"


source_files = [
    (str(f), str(f.relative_to(_application_path)))
    for f in _application_path.glob("**/*")
    if f.is_file()
]

docsassist_path = PROJECT_ROOT / "docsassist"

source_files.extend(
    [
        (str(docsassist_path / "__init__.py"), "docsassist/__init__.py"),
        (str(docsassist_path / "deployments.py"), "docsassist/deployments.py"),
        (str(docsassist_path / "predict.py"), "docsassist/predict.py"),
        (str(docsassist_path / "schema.py"), "docsassist/schema.py"),
        (str(docsassist_path / "i18n.py"), "docsassist/i18n.py"),
    ]
)

application_locale = LocaleSettings().app_locale

if application_locale != LanguageCode.EN:
    source_files.append(
        (
            str(
                docsassist_path
                / "locale"
                / application_locale
                / "LC_MESSAGES"
                / "base.mo"
            ),
            f"docsassist/locale/{application_locale}/LC_MESSAGES/base.mo",
        )
    )

app_source_args = ApplicationSourceArgs(
    resource_name=f"Guarded RAG App Source [{project_name}]",
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_STREAMLIT.value.id,
    files=source_files,
).model_dump(mode="json", exclude_none=True)

app_resource_name: str = f"Guarded RAG Application [{project_name}]"


def apply_feedback_score(
    application_id: str, custom_metric_id: str, rag_deployment_id: str
) -> str:
    client = dr.client.get_client()
    try:
        app_source_id = (
            client.get(f"customApplications/{application_id}/")
            .json()
            .get("customApplicationSourceId")
        )
        pulumi.info(f"app_source_id: {app_source_id}")
    except Exception as e:
        raise pulumi.RunError(
            f"Failed to get custom application source id for application {application_id}"
        ) from e
    app_source_version_id = (
        _unsafe_get_or_create_custom_application_source_version_from_previous(
            endpoint=client.endpoint,
            token=client.token,
            custom_application_source_id=app_source_id,
            runtime_parameter_values=[
                {
                    "field_name": "DEPLOYMENT_ID",
                    "value": rag_deployment_id,
                    "type": "string",
                },
                {
                    "field_name": "CUSTOM_METRIC_ID",
                    "value": custom_metric_id,
                    "type": "string",
                },
            ],
        )
    )
    return str(app_source_version_id)
