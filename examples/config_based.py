"""
Configuration-based usage example.

This example demonstrates:
1. Creating a default config file
2. Loading configuration
3. Creating BranchSelector from config (placeholder - not yet implemented)
"""

from pathlib import Path
from chatroutes_autobranch.config.loader import create_default_config, load_config


def main():
    """Run config-based example."""
    print("ChatRoutes AutoBranch - Config-Based Usage Example\n")

    # 1. Create default config
    config_path = Path("example_config.yaml")
    print(f"Creating default config at {config_path}...")
    create_default_config(config_path)
    print("✓ Config created\n")

    # 2. Load config
    print("Loading config...")
    config = load_config(config_path)
    print("✓ Config loaded\n")

    # 3. Display config
    print("Configuration:")
    print(f"  Beam K: {config['beam']['k']}")
    print(f"  Scorer type: {config['scorer']['type']}")
    print(f"  Novelty type: {config['novelty']['type']}")
    print(f"  Novelty threshold: {config['novelty']['threshold']}")
    print(f"  Entropy type: {config['entropy']['type']}")
    print(f"  Entropy min: {config['entropy']['min_entropy']}")
    print(f"  Budget max_nodes: {config['budget']['max_nodes']}")
    print(f"  Budget mode: {config['budget']['mode']}")

    # 4. Create selector from config
    print("\nCreating BranchSelector from config...")
    from chatroutes_autobranch import BranchSelector

    # Add embedding config for the components that need it
    config["embedding"] = {"type": "dummy", "dimension": 128, "seed": 42}

    selector = BranchSelector.from_config(config)
    print("✓ BranchSelector created successfully")

    print(f"\nSelector components:")
    print(f"  - Beam selector: k={selector.beam_selector.k}")
    print(f"  - Novelty filter: {type(selector.novelty_filter).__name__ if selector.novelty_filter else 'None'}")
    print(f"  - Entropy stopper: {type(selector.entropy_stopper).__name__ if selector.entropy_stopper else 'None'}")
    print(f"  - Budget manager: {type(selector.budget_manager).__name__ if selector.budget_manager else 'None'}")

    # Cleanup
    if config_path.exists():
        config_path.unlink()
        print(f"\n✓ Cleaned up {config_path}")


if __name__ == "__main__":
    main()
