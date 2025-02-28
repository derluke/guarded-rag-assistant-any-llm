import time
from typing import Any, Dict, List, Optional, Union

import pulumi
from datarobot.client import get_client
from datarobot.models.genai.vector_database import ChunkingParameters, VectorDatabase
from pulumi import Output
from pulumi.dynamic import (
    CreateResult,
    DiffResult,
    ReadResult,
    Resource,
    ResourceProvider,
    UpdateResult,
)
from pulumi_datarobot import (
    VectorDatabaseChunkingParametersArgs,
    VectorDatabaseChunkingParametersArgsDict,
)


class DataRobotVectorDatabaseProvider(ResourceProvider):
    def create(self, props: Dict[str, Any]) -> CreateResult:
        """
        Create a new vector database in DataRobot.
        """
        # Extract core properties (to be tracked in state)
        dataset_id = props.get("dataset_id")
        name = props.get("name")
        use_case_id = props.get("use_case_id")
        chunking_parameters = props.get("chunking_parameters", {})

        # Extract operational properties (not tracked in state)
        options = props.get("options", {})
        wait_for_completion = options.get("wait_for_completion", True)
        timeout_seconds = options.get("timeout_seconds", 600)

        # Handle chunking parameters
        chunking_params = None
        if chunking_parameters:
            chunking_params = ChunkingParameters(
                embedding_model=chunking_parameters.get("embedding_model"),
                chunking_method=chunking_parameters.get("chunking_method"),
                chunk_size=chunking_parameters.get("chunk_size"),
                chunk_overlap_percentage=chunking_parameters.get(
                    "chunk_overlap_percentage"
                ),
                separators=chunking_parameters.get("separators"),
                custom_chunking=chunking_parameters.get("custom_chunking", False),
            )

            # Handle embedding validation if provided
            if chunking_parameters.get("embedding_validation_id"):
                chunking_params.embedding_validation_id = chunking_parameters.get(
                    "embedding_validation_id"
                )

        # Initialize DataRobot client if needed
        self._ensure_client()

        # Create vector database
        vdb = VectorDatabase.create(
            dataset_id=dataset_id,  # type: ignore
            chunking_parameters=chunking_params,
            use_case=use_case_id,
            name=name,
        )

        # Wait for vector database to be ready if specified
        if wait_for_completion:
            start_time = time.time()
            while vdb.execution_status not in ["COMPLETED", "FAILED"]:
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Vector database creation timed out after {timeout_seconds} seconds"
                    )

                # Sleep to avoid API rate limiting
                time.sleep(5)

                # Refresh vector database status
                vdb = VectorDatabase.get(vdb.id)

                if vdb.execution_status == "FAILED":
                    raise Exception(
                        f"Vector database creation failed: {vdb.error_message}"
                    )

        # Convert vector database to dict for output
        # Use exactly the same format as provided in the props to match existing state
        vdb_dict = self._vdb_to_dict(vdb, props)

        return CreateResult(id_=vdb.id, outs=vdb_dict)

    def read(self, id: str, props: Dict[str, Any]) -> ReadResult:
        """
        Read an existing vector database from DataRobot.
        """
        self._ensure_client()

        # Get vector database by ID
        vdb = VectorDatabase.get(id)

        # Use the same format as provided in the props to match existing state
        vdb_dict = self._vdb_to_dict(vdb, props)

        return ReadResult(id_=vdb.id, outs=vdb_dict)

    def update(
        self, id: str, old_props: Dict[str, Any], new_props: Dict[str, Any]
    ) -> UpdateResult:
        """
        Update an existing vector database in DataRobot.
        Only supports updating the name.
        """
        self._ensure_client()

        # Get vector database by ID
        vdb = VectorDatabase.get(id)

        # Update vector database (only name can be updated)
        name = new_props.get("name")
        if name and name != vdb.name:
            vdb = vdb.update(name=name)

        # Use the same format as provided in the new_props to match existing state
        vdb_dict = self._vdb_to_dict(vdb, new_props)
        return UpdateResult(outs=vdb_dict)

    def delete(self, id: str, props: Dict[str, Any]) -> None:
        """
        Delete a vector database from DataRobot.
        """
        self._ensure_client()

        # Get vector database by ID
        vdb = VectorDatabase.get(id)

        # Delete vector database
        vdb.delete()

    def diff(
        self, id: str, old_props: Dict[str, Any], new_props: Dict[str, Any]
    ) -> DiffResult:
        """
        Determine if an update is needed and what kind.
        """
        # Check if name changed
        name_changed = old_props.get("name") != new_props.get("name")

        # Check if chunking parameters changed
        changes = name_changed
        replaces = set()

        # If we have chunking parameters in both old and new, compare them
        if "chunking_parameters" in new_props and "chunking_parameters" in old_props:
            old_cp = old_props.get("chunking_parameters", {})
            new_cp = new_props.get("chunking_parameters", {})

            # Handle custom_chunking specifically to avoid unnecessary updates
            # If old doesn't have it (or it's false) and new has it as false, consider them equal
            if (
                "custom_chunking" not in old_cp
                and new_cp.get("custom_chunking") is False
            ):
                new_cp_for_diff = new_cp.copy()
                new_cp_for_diff.pop("custom_chunking", None)
                cp_changed, changed_fields = self._chunking_params_diff(
                    old_cp, new_cp_for_diff
                )
            else:
                cp_changed, changed_fields = self._chunking_params_diff(old_cp, new_cp)

            if cp_changed:
                changes = True
                # Add specific fields to replaces
                for field in changed_fields:
                    replaces.add(f"chunking_parameters.{field}")

        # Handle migration from flat to nested format
        # If old doesn't have chunking_parameters but has embedding_model, it's in flat format
        elif "chunking_parameters" in new_props and "embedding_model" in old_props:
            # Extract old flat chunking parameters
            old_flat_cp = {}
            for field in [
                "embedding_model",
                "chunking_method",
                "chunk_size",
                "chunk_overlap_percentage",
                "separators",
                "custom_chunking",
                "embedding_validation_id",
            ]:
                if field in old_props:
                    old_flat_cp[field] = old_props.get(field)

            # Compare with new nested parameters
            new_cp = new_props.get("chunking_parameters", {})

            # Handle custom_chunking specifically to avoid unnecessary updates
            if (
                "custom_chunking" not in old_flat_cp
                and new_cp.get("custom_chunking") is False
            ):
                new_cp_for_diff = new_cp.copy()
                new_cp_for_diff.pop("custom_chunking", None)
                cp_changed, changed_fields = self._chunking_params_diff(
                    old_flat_cp, new_cp_for_diff
                )
            else:
                cp_changed, changed_fields = self._chunking_params_diff(
                    old_flat_cp, new_cp
                )

            if cp_changed:
                changes = True
                # Add specific fields to replaces
                for field in changed_fields:
                    replaces.add(f"chunking_parameters.{field}")

        # If all chunking parameters need to change, use a more efficient replacement
        if (
            len(replaces) > 3
        ):  # If more than 3 parameters change, replace the whole thing
            replaces = {"chunking_parameters"}

        # Set delete_before_replace to False to ensure we create the new resource first
        return DiffResult(
            changes=changes, replaces=list(replaces), delete_before_replace=False
        )

    def _chunking_params_diff(
        self, old_params: Dict[str, Any], new_params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Compare chunking parameters and return if they changed and what fields changed.

        Returns a tuple of (changed: bool, changed_fields: List[str])
        """
        changed = False
        changed_fields = []

        # Keys to compare - these are the ones that affect chunking
        keys_to_compare = [
            "embedding_model",
            "chunking_method",
            "chunk_size",
            "chunk_overlap_percentage",
            "separators",
            "custom_chunking",
            "embedding_validation_id",
        ]

        for key in keys_to_compare:
            old_value = old_params.get(key)
            new_value = new_params.get(key)

            # Skip if both are None or not present
            if old_value is None and new_value is None:
                continue

            # Special case for custom_chunking
            if key == "custom_chunking":
                # If old is not set (or False) and new is False, consider them equal
                if (old_value is None or old_value is False) and new_value is False:
                    continue

            # Special case for separators which is a list
            if key == "separators" and old_value is not None and new_value is not None:
                if sorted(old_value) != sorted(new_value):
                    changed = True
                    changed_fields.append(key)
            elif old_value != new_value:
                changed = True
                changed_fields.append(key)

        return changed, changed_fields

    def _ensure_client(self) -> None:
        """
        Ensure DataRobot client is initialized.
        This assumes configuration is handled through environment variables or config file.
        """
        # This will use the configured client or initialize a new one if needed
        get_client()

    def _vdb_to_dict(
        self, vdb: VectorDatabase, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a VectorDatabase object to a dictionary for serialization.
        Formats the output to match the existing state as closely as possible.
        """
        # Start with core properties that are always tracked
        result: dict[str, Any] = {
            "id": vdb.id,
            "name": vdb.name,
            "use_case_id": vdb.use_case_id,
            "dataset_id": vdb.dataset_id,
        }

        # Add chunking parameters matching the format in props
        if "chunking_parameters" in props:
            # Use the same chunking parameters structure as in props
            chunking_parameters: dict[str, Any] = {}

            # Copy the structure of chunking parameters from props
            for key in props.get("chunking_parameters", {}):
                if key == "custom_chunking" and not vdb.custom_chunking:
                    # Only include custom_chunking if it was in the original props
                    # AND it's true in the vector database
                    if vdb.custom_chunking:
                        chunking_parameters["custom_chunking"] = True
                else:
                    # For all other keys, get the value from vdb
                    if key == "embedding_model":
                        chunking_parameters[key] = vdb.embedding_model
                    elif key == "chunking_method":
                        chunking_parameters[key] = vdb.chunking_method
                    elif key == "chunk_size":
                        chunking_parameters[key] = vdb.chunk_size
                    elif key == "chunk_overlap_percentage":
                        chunking_parameters[key] = vdb.chunk_overlap_percentage
                    elif key == "separators":
                        chunking_parameters[key] = vdb.separators
                    elif key == "embedding_validation_id" and hasattr(
                        vdb, "embedding_validation_id"
                    ):
                        chunking_parameters[key] = vdb.embedding_validation_id

            result["chunking_parameters"] = chunking_parameters
        else:
            # If props doesn't have chunking_parameters, use flat properties
            # for backward compatibility
            result["embedding_model"] = vdb.embedding_model
            result["chunking_method"] = vdb.chunking_method
            result["chunk_size"] = vdb.chunk_size
            result["chunk_overlap_percentage"] = vdb.chunk_overlap_percentage
            result["separators"] = vdb.separators

            # Only include custom_chunking if it's true
            if vdb.custom_chunking:
                result["custom_chunking"] = True

            # Add embedding_validation_id if it exists
            if (
                hasattr(vdb, "embedding_validation_id")
                and vdb.embedding_validation_id is not None
            ):
                result["embedding_validation_id"] = vdb.embedding_validation_id

        # Include chunks_count and execution_status only if they were in props
        if "chunks_count" in props:
            result["chunks_count"] = vdb.chunks_count

        if "execution_status" in props:
            result["execution_status"] = vdb.execution_status

        return result


class DataRobotVectorDatabase(Resource):
    """
    A Pulumi resource for managing DataRobot Vector Databases.
    """

    def __init__(
        self,
        resource_name: str,
        opts: Optional[pulumi.ResourceOptions] = None,
        dataset_id: Optional[pulumi.Input[str]] = None,
        name: Optional[pulumi.Input[str]] = None,
        use_case_id: Optional[pulumi.Input[str]] = None,
        wait_for_completion: Optional[pulumi.Input[bool]] = True,
        timeout_seconds: Optional[pulumi.Input[int]] = 600,
        # Chunking parameters as a nested object
        chunking_parameters: Optional[
            pulumi.Input[
                Union[
                    "VectorDatabaseChunkingParametersArgs",
                    "VectorDatabaseChunkingParametersArgsDict",
                    Dict[str, Any],
                ]
            ]
        ] = None,
        __props__: Any = None,
    ):
        """
        Create a new DataRobot Vector Database resource.

        :param resource_name: The name of the resource in Pulumi.
        :param dataset_id: The ID of the dataset to use for creating the vector database.
        :param use_case_id: The ID of the use case to link the vector database to.
        :param name: Custom name for the vector database.
        :param wait_for_completion: Flag to wait for vector database creation to complete (default: True).
        :param timeout_seconds: Timeout in seconds when waiting for completion (default: 600).
        :param chunking_parameters: Dictionary or object defining how documents are split and embedded.
        :param opts: Optional resource options.
        """
        # Define core properties to track in state
        props: dict[str, Any] = {
            "dataset_id": dataset_id,
            "use_case_id": use_case_id,
            "name": name if name is not None else f"pulumi-vdb-{resource_name}",
        }

        # Process chunking parameters as a nested object
        if chunking_parameters is not None:
            # Check if it's already a Dict
            if isinstance(chunking_parameters, dict):
                chunking_dict: dict[str, Any] = chunking_parameters
            # Check if it has __dict__ attribute
            elif hasattr(chunking_parameters, "__dict__"):
                chunking_dict = {
                    k: v
                    for k, v in chunking_parameters.__dict__.items()
                    if not k.startswith("_")
                }
            # Otherwise try to convert to dict
            else:
                try:
                    chunking_dict = {
                        k: v
                        for k, v in vars(chunking_parameters).items()
                        if not k.startswith("_")
                    }
                except (TypeError, AttributeError):
                    # If all else fails, just use it as is
                    chunking_dict = chunking_parameters  # type: ignore

            # Only include custom_chunking if it's explicitly set to True
            if (
                "custom_chunking" in chunking_dict
                and not chunking_dict["custom_chunking"]
            ):
                chunking_dict = chunking_dict.copy()
                chunking_dict.pop("custom_chunking")

            props["chunking_parameters"] = chunking_dict

        # Initialize the resource
        super().__init__(DataRobotVectorDatabaseProvider(), resource_name, props, opts)

    # Add output properties for Pulumi to track
    @property
    def vector_database_id(self) -> Output[str]:
        return self.id

    @property
    def execution_status(self) -> Output[str]:
        return self.execution_status

    @property
    def chunks_count(self) -> Output[int]:
        return self.chunks_count


# Helper class for creating chunking parameters
class ChunkingParametersArgs:
    """
    Helper class for creating chunking parameters in a structured way.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        chunking_method: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap_percentage: Optional[int] = None,
        separators: Optional[List[str]] = None,
        custom_chunking: bool = False,
        embedding_validation_id: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap_percentage = chunk_overlap_percentage
        self.separators = separators

        # Only set custom_chunking if it's True
        if custom_chunking:
            self.custom_chunking = custom_chunking

        if embedding_validation_id:
            self.embedding_validation_id = embedding_validation_id


# Helper function to create chunking parameters dictionary
def create_chunking_parameters(
    embedding_model: Optional[str] = None,
    chunking_method: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap_percentage: Optional[int] = None,
    separators: Optional[List[str]] = None,
    custom_chunking: bool = False,
    embedding_validation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper function to create chunking parameters dictionary for DataRobotVectorDatabase.

    :param embedding_model: Optional name of the text embedding model.
    :param chunking_method: Optional name of the method to split dataset documents.
    :param chunk_size: Optional size of each text chunk in number of tokens.
    :param chunk_overlap_percentage: Optional overlap percentage between chunks.
    :param separators: Optional strings used to split documents into text chunks.
    :param custom_chunking: Whether the chunking is custom (default: False).
    :param embedding_validation_id: Optional ID for custom embedding validation.
    :return: Dictionary of chunking parameters.
    """
    params = {
        "embedding_model": embedding_model,
        "chunking_method": chunking_method,
        "chunk_size": chunk_size,
        "chunk_overlap_percentage": chunk_overlap_percentage,
        "separators": separators,
    }

    # Only include custom_chunking if True
    if custom_chunking:
        params["custom_chunking"] = custom_chunking

    # Only include embedding_validation_id if provided
    if embedding_validation_id:
        params["embedding_validation_id"] = embedding_validation_id

    # Filter out None values
    return {k: v for k, v in params.items() if v is not None}
