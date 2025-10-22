"""Tests for Budget and BudgetManager."""

import pytest
from chatroutes_autobranch.budget.manager import (
    Budget,
    BudgetManager,
    BudgetExceededError,
)


class TestBudget:
    """Tests for Budget dataclass."""

    def test_create_valid_budget(self):
        """Test creating a valid budget."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        assert budget.max_nodes == 32
        assert budget.max_tokens == 30000
        assert budget.max_ms == 12000

    def test_budget_validation_nodes(self):
        """Test that max_nodes must be >= 1."""
        with pytest.raises(ValueError, match="max_nodes must be >= 1"):
            Budget(max_nodes=0, max_tokens=1000, max_ms=1000)

    def test_budget_validation_tokens(self):
        """Test that max_tokens must be >= 1."""
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            Budget(max_nodes=10, max_tokens=-100, max_ms=1000)

    def test_budget_validation_ms(self):
        """Test that max_ms must be >= 1."""
        with pytest.raises(ValueError, match="max_ms must be >= 1"):
            Budget(max_nodes=10, max_tokens=1000, max_ms=0)


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_initialization_strict_mode(self):
        """Test BudgetManager initialization in strict mode."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        assert manager.mode == "strict"
        assert manager.budget == budget
        assert manager._used_nodes == 0
        assert manager._used_tokens == 0
        assert manager._used_ms == 0

    def test_initialization_soft_mode(self):
        """Test BudgetManager initialization in soft mode."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="soft")

        assert manager.mode == "soft"

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        with pytest.raises(ValueError, match="mode must be 'strict' or 'soft'"):
            BudgetManager(budget, mode="invalid")

    def test_admit_within_budget(self):
        """Test admitting nodes within budget."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        # Should succeed
        result = manager.admit(n_new=5, est_tokens=1000, est_ms=500)
        assert result is True

    def test_admit_exceeds_nodes_strict(self):
        """Test exceeding node budget in strict mode raises exception."""
        budget = Budget(max_nodes=10, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        # Use up most of the budget
        manager.record_nodes(8)

        # Try to exceed
        with pytest.raises(BudgetExceededError) as exc_info:
            manager.admit(n_new=5, est_tokens=1000, est_ms=500)

        assert "Budget would be exceeded" in str(exc_info.value)
        assert exc_info.value.details["max_nodes"] == 10
        assert exc_info.value.details["used_nodes"] == 8
        assert exc_info.value.details["requested"] == 5

    def test_admit_exceeds_tokens_strict(self):
        """Test exceeding token budget in strict mode."""
        budget = Budget(max_nodes=100, max_tokens=5000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        # Use up most of token budget
        manager.update(actual_tokens=4500, actual_ms=1000)

        # Try to exceed
        with pytest.raises(BudgetExceededError):
            manager.admit(n_new=5, est_tokens=1000, est_ms=500)

    def test_admit_exceeds_ms_strict(self):
        """Test exceeding latency budget in strict mode."""
        budget = Budget(max_nodes=100, max_tokens=30000, max_ms=5000)
        manager = BudgetManager(budget, mode="strict")

        # Use up most of latency budget
        manager.update(actual_tokens=1000, actual_ms=4500)

        # Try to exceed
        with pytest.raises(BudgetExceededError):
            manager.admit(n_new=5, est_tokens=1000, est_ms=1000)

    def test_admit_exceeds_soft_mode(self):
        """Test exceeding budget in soft mode returns False."""
        budget = Budget(max_nodes=10, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="soft")

        manager.record_nodes(8)

        # Should return False, not raise
        result = manager.admit(n_new=5, est_tokens=1000, est_ms=500)
        assert result is False

    def test_update_usage(self):
        """Test updating usage after generation."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        manager.update(actual_tokens=1200, actual_ms=800)

        assert manager._used_tokens == 1200
        assert manager._used_ms == 800

    def test_record_nodes(self):
        """Test recording node usage."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        manager.record_nodes(5)
        assert manager._used_nodes == 5

        manager.record_nodes(3)
        assert manager._used_nodes == 8

    def test_usage_property(self):
        """Test usage property returns current stats."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        manager.record_nodes(5)
        manager.update(actual_tokens=1000, actual_ms=500)

        usage = manager.usage
        assert usage["used_nodes"] == 5
        assert usage["max_nodes"] == 32
        assert usage["used_tokens"] == 1000
        assert usage["max_tokens"] == 30000
        assert usage["used_ms"] == 500
        assert usage["max_ms"] == 12000

    def test_get_set_state(self):
        """Test state serialization and restoration."""
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        manager = BudgetManager(budget, mode="strict")

        # Set some usage
        manager.record_nodes(5)
        manager.update(actual_tokens=1000, actual_ms=500)

        # Save state
        state = manager.get_state()
        assert state["used_nodes"] == 5
        assert state["used_tokens"] == 1000
        assert state["used_ms"] == 500

        # Create new manager and restore
        manager2 = BudgetManager(budget, mode="strict")
        manager2.set_state(state)

        assert manager2._used_nodes == 5
        assert manager2._used_tokens == 1000
        assert manager2._used_ms == 500
