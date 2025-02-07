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

import os
import pathlib
import sys

import pulumi
import pulumi_datarobot as datarobot

sys.path.append("..")

from docsassist.deployments import (
    app_env_name,
    rag_deployment_env_name,
)
from docsassist.i18n import LocaleSettings
from docsassist.schema import (
    ApplicationType,
    RAGType,
)
from infra import (
    settings_app_infra,
    settings_generative,
    settings_main,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.globals import GlobalLLM
from infra.common.papermill import run_notebook
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_llm_credential import (
    get_credential_runtime_parameter_values,
    get_credentials,
)
from infra.components.proxy_llm_blueprint import ProxyLLMBlueprint
from infra.settings_global_guardrails import llm_metrics

# from infra.settings_global_guardrails import stay_on_topic_guardrail
from infra.settings_proxy_llm import TEXTGEN_DEPLOYMENT_PROMPT_COLUMN_NAME

DEPLOYMENT_ID = os.environ.get("TEXTGEN_DEPLOYMENT_ID")

if settings_generative.LLM == GlobalLLM.DEPLOYED_LLM:
    if DEPLOYMENT_ID is None:
        raise ValueError(
            "TEXTGEN_DEPLOYMENT_ID must be set when using a deployed LLM. Plese check your .env file"
        )

LocaleSettings().setup_locale()

check_feature_flags(pathlib.Path("feature_flag_requirements.yaml"))

if "DATAROBOT_DEFAULT_USE_CASE" in os.environ:
    use_case_id = os.environ["DATAROBOT_DEFAULT_USE_CASE"]
    pulumi.info(f"Using existing use case '{use_case_id}'")
    use_case = datarobot.UseCase.get(
        id=use_case_id,
        resource_name="Guarded RAG Use Case [PRE-EXISTING]",
    )
else:
    use_case = datarobot.UseCase(**settings_main.use_case_args)

if settings_main.default_prediction_server_id is None:
    prediction_environment = datarobot.PredictionEnvironment(
        **settings_main.prediction_environment_args,
    )
else:
    prediction_environment = datarobot.PredictionEnvironment.get(
        "Guarded RAG Prediction Environment [PRE-EXISTING]",
        settings_main.default_prediction_server_id,
    )

credentials = get_credentials(settings_generative.LLM)

credential_runtime_parameter_values = get_credential_runtime_parameter_values(
    credentials=credentials
)

guard_configurations = llm_metrics
# + [stay_on_topic_guardrail]

if settings_main.core.rag_type == RAGType.DR:
    dataset = datarobot.DatasetFromFile(
        use_case_ids=[use_case.id],
        **settings_generative.dataset_args.model_dump(),
    )
    vector_database = datarobot.VectorDatabase(
        dataset_id=dataset.id,
        use_case_id=use_case.id,
        **settings_generative.vector_database_args.model_dump(),
    )
    playground = datarobot.Playground(
        use_case_id=use_case.id,
        **settings_generative.playground_args.model_dump(),
    )

    if settings_generative.LLM == GlobalLLM.DEPLOYED_LLM:
        assert DEPLOYMENT_ID is not None, "TEXTGEN_DEPLOYMENT_ID must be set in .env"
        proxy_llm_deployment = datarobot.Deployment.get(
            resource_name="Existing LLM Deployment", id=DEPLOYMENT_ID
        )

        llm_blueprint = ProxyLLMBlueprint(
            resource_name=f"Proxy Model LLM Blueprint [{settings_main.project_name}]",
            llm_blueprint_args=settings_generative.llm_blueprint_args,
            use_case_id=use_case.id,
            playground_id=playground.id,
            proxy_llm_deployment_id=proxy_llm_deployment.id,
            vector_database_id=vector_database.id,
            prompt_column_name=TEXTGEN_DEPLOYMENT_PROMPT_COLUMN_NAME,
        )

    elif settings_generative.LLM.name != GlobalLLM.DEPLOYED_LLM.name:
        llm_blueprint = datarobot.LlmBlueprint(  # type: ignore[assignment]
            playground_id=playground.id,
            vector_database_id=vector_database.id,
            **settings_generative.llm_blueprint_args.model_dump(),
        )

    rag_custom_model = datarobot.CustomModel(
        **settings_generative.custom_model_args.model_dump(exclude_none=True),
        use_case_ids=[use_case.id],
        source_llm_blueprint_id=llm_blueprint.id,
        guard_configurations=guard_configurations,
        runtime_parameter_values=[]
        if settings_generative.LLM.name == GlobalLLM.DEPLOYED_LLM.name
        else credential_runtime_parameter_values,
    )

elif settings_main.core.rag_type == RAGType.DIY:
    if not all(
        [
            path.exists()
            for path in settings_generative.diy_rag_nb_output.model_dump().values()
        ]
    ):
        pulumi.info("Executing doc chunking + vdb building notebook...")
        run_notebook(settings_generative.diy_rag_nb)
    else:
        pulumi.info(
            f"Using existing outputs from build_rag.ipynb in '{settings_generative.diy_rag_deployment_path}'"
        )

    rag_custom_model = datarobot.CustomModel(
        files=settings_generative.get_diy_rag_files(
            runtime_parameter_values=credential_runtime_parameter_values,
        ),
        runtime_parameter_values=credential_runtime_parameter_values,
        guard_configurations=guard_configurations,
        use_case_ids=[use_case.id],
        **settings_generative.custom_model_args.model_dump(
            mode="json", exclude_none=True
        ),
    )
else:
    raise NotImplementedError(f"Unknown RAG type: {settings_main.core.rag_type}")

# rag_deployment = CustomModelDeployment(
#     resource_name=f"Guarded RAG Deploy [{settings_main.project_name}]",
#     custom_model_version_id=rag_custom_model.version_id,
#     registered_model_args=settings_generative.registered_model_args,
#     prediction_environment=prediction_environment,
#     deployment_args=settings_generative.deployment_args,
#     use_case_ids=[use_case.id],
# )

# app_runtime_parameters = [
#     datarobot.ApplicationSourceRuntimeParameterValueArgs(
#         key=rag_deployment_env_name, type="deployment", value=rag_deployment.id
#     ),
#     datarobot.ApplicationSourceRuntimeParameterValueArgs(
#         key="APP_LOCALE", type="string", value=LocaleSettings().app_locale
#     ),
# ]

# if settings_main.core.application_type == ApplicationType.DIY:
#     application_source = datarobot.ApplicationSource(
#         runtime_parameter_values=app_runtime_parameters,
#         **settings_app_infra.app_source_args,
#     )
#     qa_application = datarobot.CustomApplication(
#         resource_name=settings_app_infra.app_resource_name,
#         source_version_id=application_source.version_id,
#         use_case_ids=[use_case.id],
#     )
# elif settings_main.core.application_type == ApplicationType.DR:
#     qa_application = datarobot.QaApplication(  # type: ignore[assignment]
#         resource_name=settings_app_infra.app_resource_name,
#         name=f"Guarded RAG Assistant [{settings_main.project_name}]",
#         deployment_id=rag_deployment.deployment_id,
#         opts=pulumi.ResourceOptions(delete_before_replace=True),
#     )
# else:
#     raise NotImplementedError(
#         f"Unknown application type: {settings_main.core.application_type}"
#     )

# qa_application.id.apply(settings_app_infra.ensure_app_settings)


# pulumi.export(rag_deployment_env_name, rag_deployment.id)
# pulumi.export(app_env_name, qa_application.id)

# pulumi.export(
#     settings_generative.deployment_args.resource_name,
#     rag_deployment.id.apply(get_deployment_url),
# )
# pulumi.export(
#     settings_app_infra.app_resource_name,
#     qa_application.application_url,
# )
