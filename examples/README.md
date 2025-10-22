# Examples

This directory contains example scripts demonstrating how to use chatroutes-autobranch.

## Basic Usage

Run the basic usage example:

```bash
python examples/basic_usage.py
```

This demonstrates:
- Creating all components manually
- Running a single selection step
- Inspecting results
- State management

## Config-Based Usage

Run the config-based example:

```bash
python examples/config_based.py
```

This demonstrates:
- Creating a default configuration file
- Loading configuration
- Configuration structure (BranchSelector.from_config() is not yet implemented)

## Running Examples

Make sure you've installed the package first:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run examples
python examples/basic_usage.py
python examples/config_based.py
```

## Next Steps

For production use:
1. Replace `DummyEmbeddingProvider` with real embeddings (OpenAI, HuggingFace, etc.)
2. Replace `StaticScorer` with `CompositeScorer` (requires implementation)
3. Implement actual entropy computation in `ShannonEntropyStopper`
4. Implement MMR and cosine novelty filtering

See the main README.md for more details.
