"""Budget management for cost control."""

from dataclasses import dataclass


class BudgetExceededError(Exception):
    """Raised when budget is exceeded in strict mode."""

    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


@dataclass
class Budget:
    """
    Budget limits for tree exploration.

    Attributes:
        max_nodes: Maximum total branches in tree.
        max_tokens: Maximum total tokens (input + output).
        max_ms: Maximum cumulative latency (milliseconds).

    Example:
        >>> budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
    """

    max_nodes: int
    max_tokens: int
    max_ms: int

    def __post_init__(self) -> None:
        """Validate budget values."""
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {self.max_nodes}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.max_ms < 1:
            raise ValueError(f"max_ms must be >= 1, got {self.max_ms}")


class BudgetManager:
    """
    Manages budget tracking and enforcement.

    Args:
        budget: Budget limits.
        mode: "strict" (raise on exceeded) or "soft" (return False).
              Default: "strict"

    Usage:
        1. PRE-GENERATION: Call admit() before generating children
        2. POST-SELECTION: Call update() after step() with actuals

    Example:
        >>> budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        >>> manager = BudgetManager(budget, mode="strict")
        >>>
        >>> # Before generation
        >>> if manager.admit(n_new=5, est_tokens=1000, est_ms=2000):
        ...     # Generate children
        ...     pass
        >>>
        >>> # After generation
        >>> manager.update(actual_tokens=1200, actual_ms=1800)
    """

    def __init__(self, budget: Budget, mode: str = "strict"):
        if mode not in ("strict", "soft"):
            raise ValueError(f"mode must be 'strict' or 'soft', got {mode}")

        self.budget = budget
        self.mode = mode

        # Usage tracking
        self._used_nodes = 0
        self._used_tokens = 0
        self._used_ms = 0

    def admit(self, n_new: int, est_tokens: int, est_ms: int) -> bool:
        """
        Check if budget allows admitting n_new nodes.

        Args:
            n_new: Number of new nodes to add.
            est_tokens: Estimated total tokens.
            est_ms: Estimated latency (milliseconds).

        Returns:
            True if budget allows; False if would exceed.

        Raises:
            BudgetExceededError: In strict mode when budget would be exceeded.
        """
        # Check if adding these would exceed budget
        would_exceed_nodes = (self._used_nodes + n_new) > self.budget.max_nodes
        would_exceed_tokens = (self._used_tokens + est_tokens) > self.budget.max_tokens
        would_exceed_ms = (self._used_ms + est_ms) > self.budget.max_ms

        if would_exceed_nodes or would_exceed_tokens or would_exceed_ms:
            details = {
                "max_nodes": self.budget.max_nodes,
                "used_nodes": self._used_nodes,
                "requested": n_new,
                "max_tokens": self.budget.max_tokens,
                "used_tokens": self._used_tokens,
                "est_tokens": est_tokens,
                "max_ms": self.budget.max_ms,
                "used_ms": self._used_ms,
                "est_ms": est_ms,
            }

            if self.mode == "strict":
                raise BudgetExceededError("Budget would be exceeded", details)
            else:
                return False

        return True

    def update(self, actual_tokens: int, actual_ms: int) -> None:
        """
        Record actual usage after generation.

        Args:
            actual_tokens: Actual tokens consumed.
            actual_ms: Actual latency (milliseconds).

        Note: Node count is tracked internally by BranchSelector.
              This is called by user code after step().
        """
        self._used_tokens += actual_tokens
        self._used_ms += actual_ms

    def record_nodes(self, n_nodes: int) -> None:
        """
        Record node count (called by BranchSelector internally).

        Args:
            n_nodes: Number of nodes to record.
        """
        self._used_nodes += n_nodes

    def get_state(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "used_nodes": self._used_nodes,
            "used_tokens": self._used_tokens,
            "used_ms": self._used_ms,
        }

    def set_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._used_nodes = state.get("used_nodes", 0)
        self._used_tokens = state.get("used_tokens", 0)
        self._used_ms = state.get("used_ms", 0)

    @property
    def usage(self) -> dict:
        """Get current usage statistics."""
        return {
            "used_nodes": self._used_nodes,
            "max_nodes": self.budget.max_nodes,
            "used_tokens": self._used_tokens,
            "max_tokens": self.budget.max_tokens,
            "used_ms": self._used_ms,
            "max_ms": self.budget.max_ms,
        }
