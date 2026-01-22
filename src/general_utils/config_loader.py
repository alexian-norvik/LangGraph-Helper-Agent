"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is empty or invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config


def validate_config(config_dict: dict[str, Any], schema_class: type[BaseModel]) -> Any:
    """Validate configuration dictionary against a Pydantic schema.

    Args:
        config_dict: Configuration dictionary to validate
        schema_class: Pydantic model class for validation

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If the configuration doesn't match the schema
    """
    try:
        return schema_class(**config_dict)
    except ValidationError as e:
        raise e
