"""Configuration loader for BranchSelector."""

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file (.yaml, .yml, or .json).

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If file format is unsupported.

    Example:
        >>> config = load_config("config.yaml")
        >>> selector = BranchSelector.from_config(config)

    Config format:
        beam:
          k: 5
        scorer:
          type: "composite"
          weights:
            confidence: 0.3
            relevance: 0.25
            novelty_parent: 0.2
            intent_alignment: 0.15
            historical_reward: 0.1
        novelty:
          type: "cosine"
          threshold: 0.85
        entropy:
          type: "shannon"
          min_entropy: 0.6
          k_max: 5
        budget:
          max_nodes: 32
          max_tokens: 30000
          max_ms: 12000
          mode: "strict"
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    # Override with environment variables if present
    config = _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to config.

    Environment variables use CHATROUTES_ prefix with double underscore separator.

    Examples:
        CHATROUTES_BEAM__K=10
        CHATROUTES_BUDGET__MAX_NODES=64
        CHATROUTES_BUDGET__MODE=soft

    Args:
        config: Base configuration from file.

    Returns:
        Configuration with env overrides applied.
    """
    prefix = "CHATROUTES_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix and split by double underscore
        path = key[len(prefix) :].lower().split("__")

        # Navigate to the nested dict
        current = config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value (try to convert to int/float if possible)
        final_key = path[-1]
        try:
            current[final_key] = int(value)
        except ValueError:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value

    return config


def create_default_config(output_path: str | Path) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path where config will be written (.yaml or .json).

    Example:
        >>> create_default_config("config.yaml")
    """
    default_config = {
        "beam": {"k": 5},
        "scorer": {
            "type": "composite",
            "weights": {
                "confidence": 0.3,
                "relevance": 0.25,
                "novelty_parent": 0.2,
                "intent_alignment": 0.15,
                "historical_reward": 0.1,
            },
        },
        "novelty": {
            "type": "cosine",
            "threshold": 0.85,
        },
        "entropy": {
            "type": "shannon",
            "min_entropy": 0.6,
            "k_max": 5,
        },
        "budget": {
            "max_nodes": 32,
            "max_tokens": 30000,
            "max_ms": 12000,
            "mode": "strict",
        },
    }

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)
    elif suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
