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

from typing import Any, Optional, Union

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from datarobotx.idp.custom_model_llm_validation import (
    get_update_or_create_custom_model_llm_validation,
)
from datarobotx.idp.llm_blueprints import get_or_create_llm_blueprint

from infra.common.schema import LLMBlueprintArgs


class ProxyLLMBlueprint(pulumi.ComponentResource):
    @staticmethod
    def _get_custom_model_llm_validation(
        proxy_llm_deployment_id: str,
        use_case_id: str,
        prompt_column_name: str | None = None,
    ) -> str:
        dr_client = dr.client.get_client()
        try:
            deployment = dr.Deployment.get(deployment_id=proxy_llm_deployment_id)  # type: ignore[attr-defined]
        except Exception as e:
            raise ValueError("Couldn't find deployment ID") from e
        if deployment.model is None:
            raise ValueError("Deployment model is not set")

        target_column_name = deployment.model["target_name"]

        if prompt_column_name is None:
            if (
                "prompt" not in deployment.model
                or deployment.model.get("prompt") is None
            ):
                pulumi.warn(
                    "Couldn't infer prompt column name of the textgen deployment. Using default 'promptText'."
                )
            prompt_column_name = str(deployment.model.get("prompt", "promptText"))

        from datarobot.models.genai.custom_model_validation import CustomModelValidation  # noqa: I001
        from datarobot import Deployment, Model  # type: ignore

        CustomModelValidation._update = CustomModelValidation.update  # type: ignore

        def new_update(
            self: Any,
            name: Optional[str] = None,
            prompt_column_name: Optional[str] = None,
            target_column_name: Optional[str] = None,
            deployment: Optional[Union[Deployment, str]] = None,
            model: Optional[Union[Model, str]] = None,
            prediction_timeout: Optional[int] = None,
            **kwargs: Any,
        ) -> CustomModelValidation:
            return CustomModelValidation._update(  # type: ignore
                self,
                name=name,
                prompt_column_name=prompt_column_name,
                target_column_name=target_column_name,
                deployment=deployment,
                model=model,
                prediction_timeout=prediction_timeout,
            )

        CustomModelValidation.update = new_update  # type: ignore

        llm_validation_id = get_update_or_create_custom_model_llm_validation(
            endpoint=dr_client.endpoint,
            token=dr_client.token,
            deployment_id=proxy_llm_deployment_id,
            prompt_column_name=prompt_column_name,
            target_column_name=target_column_name,
            use_case=use_case_id,
        )
        return str(llm_validation_id)

    def __init__(
        self,
        resource_name: str,
        proxy_llm_deployment_id: pulumi.Output[str],
        use_case_id: pulumi.Output[str],
        playground_id: pulumi.Output[str],
        llm_blueprint_args: LLMBlueprintArgs,
        vector_database_id: pulumi.Output[str] | None = None,
        prompt_column_name: str | None = None,
        opts: Optional[pulumi.ResourceOptions] = None,
    ) -> None:
        super().__init__(
            "custom:datarobot:ProxyLLMBlueprint", resource_name, None, opts
        )

        self.llm_validation_id = pulumi.Output.all(
            proxy_llm_deployment_id=proxy_llm_deployment_id, use_case_id=use_case_id
        ).apply(
            lambda kwargs: self._get_custom_model_llm_validation(
                proxy_llm_deployment_id=kwargs["proxy_llm_deployment_id"],
                use_case_id=kwargs["use_case_id"],
                prompt_column_name=prompt_column_name,
            )
        )
        dr_client = dr.client.get_client()

        old_settings = llm_blueprint_args
        llm_blueprint_id = pulumi.Output.all(
            llm_validation_id=self.llm_validation_id,
            playground_id=playground_id,
            vector_database_id=vector_database_id
            if vector_database_id is not None
            else None,
        ).apply(
            lambda kwargs: get_or_create_llm_blueprint(
                endpoint=dr_client.endpoint,
                token=dr_client.token,
                llm_settings=dr.models.genai.llm_blueprint.LLMSettingsCustomModelDict(
                    system_prompt=old_settings.llm_settings.system_prompt,
                    validation_id=kwargs["llm_validation_id"],
                ),
                llm="custom-model",
                playground=kwargs["playground_id"],
                name=old_settings.resource_name,
                vector_database=kwargs["vector_database_id"],
                vector_database_settings=old_settings.vector_database_settings.model_dump(),
            )
        )
        self.llm_blueprint = datarobot.LlmBlueprint.get(
            id=llm_blueprint_id, resource_name="Custom Blueprint"
        )
        self.register_outputs(
            {
                "llm_validation_id": self.llm_validation_id,
                "id": self.llm_blueprint.id,
            }
        )

    @property
    @pulumi.getter(name="id")
    def id(self) -> pulumi.Output[str]:
        """
        The ID of the latest Custom Model version.
        """
        return self.llm_blueprint.id
