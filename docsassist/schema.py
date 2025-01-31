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

import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class RAGType(str, Enum):
    DIY = "diy"
    DR = "dr"


class ApplicationType(str, Enum):
    DIY = "diy"
    DR = "dr"


PROMPT_COLUMN_NAME: str = "promptText"
TARGET_COLUMN_NAME: str = "response"


class RAGInput(BaseModel):
    promptText: str = Field(serialization_alias=PROMPT_COLUMN_NAME)
    association_id: str
    messages: Optional[list[ChatCompletionMessageParam]] = []


class ReferenceMetadata(BaseModel):
    source: str
    page: Optional[int] = None


class Reference(BaseModel):
    page_content: str
    metadata: ReferenceMetadata


class DocumentModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}


class RAGModelSettings(BaseModel):
    embedding_model_name: str
    max_retries: int
    request_timeout: int
    stuff_prompt: str
    temperature: float

    @classmethod
    def filename(cls) -> str:
        return "rag_settings.yaml"


class RAGOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    completion: str = Field(validation_alias=TARGET_COLUMN_NAME)
    references: List[Reference]
    usage: Optional[Dict[str, Any]] = None
    question: Optional[str] = None

    @model_validator(mode="before")
    def parse_references(cls, values: Any) -> Any:
        if "references" in values:
            return values

        references: List[Dict[str, Any]] = []
        citation_pattern = re.compile(r"CITATION_(\w+)_(\d+)")
        target_name = values.pop("__target_name", None)

        for key, value in values.items():
            if isinstance(value, float) and math.isnan(value):
                continue
            match = citation_pattern.match(key)
            if match:
                citation_type, index = match.groups()
                index = int(index)  # Convert to 0-based index

                if len(references) <= index:
                    references.extend([{} for _ in range(index - len(references) + 1)])

                if citation_type == "CONTENT":
                    references[index]["page_content"] = value
                elif citation_type == "SOURCE":
                    if "metadata" not in references[index]:
                        references[index]["metadata"] = {}
                    references[index]["metadata"]["source"] = value
                elif citation_type == "PAGE":
                    if "metadata" not in references[index]:
                        references[index]["metadata"] = {}
                    try:
                        value = float(value)
                    except Exception:
                        value = 0
                    references[index]["metadata"]["page"] = value

        # Find the answer field
        if target_name:
            answer_field: str | None = f"{target_name}_PREDICTION"
        else:
            answer_field = next(
                (k for k in values.keys() if k.endswith("_PREDICTION")), None
            )

        if answer_field:
            values["answer"] = values.pop(answer_field)
        values["references"] = [Reference(**ref) for ref in references if ref]
        return values

    def to_dataframe(self) -> pd.DataFrame:
        input_data = {
            "question": [self.question] if self.question else [""],
            "answer": [self.completion],
        }

        for i, ref in enumerate(self.references, start=1):
            input_data[f"context{i}"] = [ref.page_content]
            input_data[f"source{i}"] = [ref.metadata.source]
            if ref.metadata.page:
                input_data[f"page{i}"] = [str(ref.metadata.page)]

        if self.usage:
            for k, v in self.usage.items():
                input_data[k] = [v]

        return pd.DataFrame(input_data)


class CoreSettings(BaseSettings):
    """Schema for core settings that can also be overridden by environment variables

    e.g. for running automated tests.
    """

    rag_documents: str = Field(
        description="Local path to zip file of pdf, txt, docx, md files to use with RAG",
    )
    rag_type: RAGType = Field(
        description="Whether to use DR RAG chunking, vectorization, retrieval, or user-provided (DIY)",
    )
    application_type: ApplicationType = Field(
        description="Whether to use the default DR QA frontend or a user-provided frontend (DIY)",
    )

    model_config = SettingsConfigDict(env_prefix="MAIN_", case_sensitive=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
        )
