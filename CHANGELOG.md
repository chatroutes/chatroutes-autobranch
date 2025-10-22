# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-19

### Added

#### Core Components
- **Beam Search** with configurable beam width (K)
  - Composite scoring with weighted features: confidence, relevance, novelty_parent, intent_alignment, historical_reward
  - Deterministic tie-breaking via lexicographic ID ordering
  - Support for custom scorers via Protocol interface

- **Novelty Filtering** for diversity enforcement
  - Cosine similarity threshold method (default 0.85)
  - MMR (Maximal Marginal Relevance) implementation with lambda parameter
  - Deterministic pruning order (score-descending processing)

- **Entropy-Based Stopping** for convergence detection
  - Shannon entropy on K-means clusters of candidate embeddings
  - Delta-entropy tracking across generations
  - Stateful entropy stopper with checkpoint/resume support
  - Normalized entropy scale [0,1] with configurable thresholds

- **Budget Management** for cost control
  - Multi-dimensional budgets: max_nodes, max_tokens, max_ms
  - Strict and soft enforcement modes
  - Pre-admit budget checking and post-update usage tracking
  - Rolling averages for improved estimation

#### Integrations
- **Embedding Providers**
  - OpenAI embeddings (text-embedding-3-large)
  - HuggingFace Transformers (sentence-transformers)
  - Dummy provider for testing and offline development

- **FastAPI Service Layer**
  - HTTP `/select` endpoint for language-agnostic access
  - Thread-safe per-request selector instantiation
  - Health check and config introspection endpoints
  - TypeScript client example included

#### Developer Experience
- **Configuration System**
  - YAML/JSON config file support
  - Environment variable overrides with double-underscore nesting
  - Pydantic-based validation with clear error messages
  - Multiple config loading methods (file, dict, unified)

- **Observability**
  - Structured JSON logging with configurable levels
  - Rich metrics per step (kept/pruned counts, scores, entropy, budget)
  - OpenTelemetry span instrumentation (optional)
  - PII-safe logging (text logging disabled by default)

- **Checkpointing**
  - Serializable checkpoint format with schema versioning
  - State management for entropy history and budget tracking
  - Resume from checkpoint with full state restoration
  - Backward-compatible checkpoint loading

#### Testing & Quality
- **Comprehensive Test Suite**
  - Unit tests for all core components
  - Property-based tests using Hypothesis
  - Golden-file tests for reproducibility
  - Thread-safety tests for concurrent usage
  - Edge case coverage (empty sets, small sets, budget exhaustion)

- **Benchmarks**
  - Performance benchmarks for K scaling
  - Diversity analysis for novelty thresholds
  - Tree depth analysis for entropy cutoffs
  - Baseline performance metrics documented

#### Documentation
- **Complete Technical Specification** (chatroutes_autobranch_v1.0.md)
  - 20 sections covering philosophy, design, API, examples, tuning
  - Algorithm specifications with pseudocode
  - Configuration reference with all parameters
  - Troubleshooting guide with common failure patterns
  - Performance benchmarks and optimization tips

- **Developer-Friendly README**
  - Quick start guide (< 5 minutes to first result)
  - Multiple usage examples (basic, multi-generation, custom scorer)
  - Integration examples (LangChain, LlamaIndex, FastAPI)
  - FAQ and tuning guidance

- **Contributing Guidelines**
  - Development setup instructions
  - Code standards and tooling requirements
  - Pull request process
  - Community guidelines (Contributor Covenant)

### Design Decisions
- **Deterministic Core**: All modules use fixed tie-breaking and seeded randomization for reproducibility
- **Zero Heavy Dependencies**: Core library has minimal deps; extras enable optional features
- **Protocol-Based Design**: All components use Python Protocols for easy swapping and testing
- **Thread-Safe**: Per-request selector instantiation pattern for concurrent usage
- **Embedding Reuse**: Parent and candidate embeddings batched and cached per step() for efficiency

### Known Limitations
- No async/await support (planned for v0.3.0)
- K-means clustering only (HDBSCAN adapter planned)
- No built-in tree visualization (planned for v0.4.0)
- Local state only (no distributed state management)

### Breaking Changes
- N/A (initial release)

---

## Development Notes

**Version Numbering:**
- MAJOR: Breaking API changes, incompatible checkpoints
- MINOR: Backward-compatible features, new optional parameters
- PATCH: Bug fixes, documentation improvements

**Unreleased Section:**
- Features merged to `next` branch but not yet released
- Preview of upcoming changes in next version

**Migration Guides:**
- Will be added when MAJOR version changes occur
- Located in `migrations/` directory

---

[unreleased]: https://github.com/chatroutes/chatroutes-autobranch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chatroutes/chatroutes-autobranch/releases/tag/v0.1.0
