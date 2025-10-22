"""Tests for configuration loader."""

import json
import os
import tempfile
from pathlib import Path
import pytest
import yaml
from chatroutes_autobranch.config.loader import (
    load_config,
    create_default_config,
    _apply_env_overrides,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_yaml_config(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.safe_dump({"beam": {"k": 5}, "budget": {"max_nodes": 32}}, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["beam"]["k"] == 5
            assert config["budget"]["max_nodes"] == 32
        finally:
            os.unlink(config_path)

    def test_load_yml_config(self):
        """Test loading config from .yml file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump({"beam": {"k": 10}}, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["beam"]["k"] == 10
        finally:
            os.unlink(config_path)

    def test_load_json_config(self):
        """Test loading config from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"beam": {"k": 7}}, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["beam"]["k"] == 7
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("beam:\n  k: 5")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_load_with_path_object(self):
        """Test loading config using Path object."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.safe_dump({"beam": {"k": 5}}, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config["beam"]["k"] == 5
        finally:
            os.unlink(config_path)


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_apply_env_overrides(self):
        """Test applying environment variable overrides."""
        config = {"beam": {"k": 5}, "budget": {"max_nodes": 32}}

        # Set env vars
        os.environ["CHATROUTES_BEAM__K"] = "10"
        os.environ["CHATROUTES_BUDGET__MAX_NODES"] = "64"

        try:
            overridden = _apply_env_overrides(config)
            assert overridden["beam"]["k"] == 10
            assert overridden["budget"]["max_nodes"] == 64
        finally:
            del os.environ["CHATROUTES_BEAM__K"]
            del os.environ["CHATROUTES_BUDGET__MAX_NODES"]

    def test_env_overrides_create_nested(self):
        """Test that env vars create nested dicts if missing."""
        config = {}

        os.environ["CHATROUTES_BEAM__K"] = "5"

        try:
            overridden = _apply_env_overrides(config)
            assert "beam" in overridden
            assert overridden["beam"]["k"] == 5
        finally:
            del os.environ["CHATROUTES_BEAM__K"]

    def test_env_overrides_string_values(self):
        """Test that non-numeric env vars are kept as strings."""
        config = {"budget": {"mode": "strict"}}

        os.environ["CHATROUTES_BUDGET__MODE"] = "soft"

        try:
            overridden = _apply_env_overrides(config)
            assert overridden["budget"]["mode"] == "soft"
        finally:
            del os.environ["CHATROUTES_BUDGET__MODE"]

    def test_env_overrides_float_values(self):
        """Test that float env vars are converted."""
        config = {}

        os.environ["CHATROUTES_SCORER__THRESHOLD"] = "0.85"

        try:
            overridden = _apply_env_overrides(config)
            assert overridden["scorer"]["threshold"] == 0.85
            assert isinstance(overridden["scorer"]["threshold"], float)
        finally:
            del os.environ["CHATROUTES_SCORER__THRESHOLD"]

    def test_env_overrides_ignore_other_vars(self):
        """Test that non-CHATROUTES env vars are ignored."""
        config = {"beam": {"k": 5}}

        os.environ["OTHER_VAR"] = "100"

        try:
            overridden = _apply_env_overrides(config)
            assert overridden["beam"]["k"] == 5
            assert "other" not in overridden
        finally:
            del os.environ["OTHER_VAR"]


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_create_default_yaml(self):
        """Test creating default YAML config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            config_path = f.name

        try:
            create_default_config(config_path)
            config = load_config(config_path)

            assert "beam" in config
            assert "scorer" in config
            assert "novelty" in config
            assert "entropy" in config
            assert "budget" in config

            assert config["beam"]["k"] == 5
            assert config["budget"]["max_nodes"] == 32
        finally:
            os.unlink(config_path)

    def test_create_default_json(self):
        """Test creating default JSON config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            config_path = f.name

        try:
            create_default_config(config_path)
            config = load_config(config_path)

            assert "beam" in config
            assert config["beam"]["k"] == 5
        finally:
            os.unlink(config_path)

    def test_create_default_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported output format"):
                create_default_config(config_path)
        finally:
            os.unlink(config_path)
