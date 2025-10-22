"""Tests for BranchSelector pipeline orchestrator."""

import pytest
from chatroutes_autobranch.core.selector import BranchSelector, SelectionResult
from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.beam.selector import BeamSelector
from chatroutes_autobranch.core.scorer import StaticScorer
from chatroutes_autobranch.budget.manager import Budget, BudgetManager, BudgetExceededError


class TestBranchSelector:
    """Tests for BranchSelector."""

    def test_initialization_minimal(self):
        """Test BranchSelector with minimal config (beam only)."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        selector = BranchSelector(beam_selector=beam)

        assert selector.beam_selector == beam
        assert selector.novelty_filter is None
        assert selector.entropy_stopper is None
        assert selector.budget_manager is None

    def test_step_basic_pipeline(self, sample_parent, sample_candidates):
        """Test basic step() with beam selection only."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        selector = BranchSelector(beam_selector=beam)

        result = selector.step(sample_parent, sample_candidates)

        assert isinstance(result, SelectionResult)
        assert len(result.kept) <= 3
        assert result.should_continue is True
        assert len(result.scored) == len(sample_candidates)

    def test_step_with_budget_manager(self, sample_parent, sample_candidates):
        """Test step() with budget management."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        budget_mgr = BudgetManager(budget, mode="strict")

        selector = BranchSelector(beam_selector=beam, budget_manager=budget_mgr)
        result = selector.step(sample_parent, sample_candidates)

        assert "budget_usage" in result.details
        # Nodes should be recorded
        assert budget_mgr._used_nodes > 0

    def test_step_budget_exceeded_strict(self, sample_parent, sample_candidates):
        """Test that budget exceeded raises exception in strict mode."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        # Very small budget
        budget = Budget(max_nodes=1, max_tokens=100, max_ms=100)
        budget_mgr = BudgetManager(budget, mode="strict")

        # Use up budget
        budget_mgr.record_nodes(1)

        selector = BranchSelector(beam_selector=beam, budget_manager=budget_mgr)

        with pytest.raises(BudgetExceededError):
            selector.step(sample_parent, sample_candidates)

    def test_step_budget_exceeded_soft(self, sample_parent, sample_candidates):
        """Test that budget exceeded returns False in soft mode."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        budget = Budget(max_nodes=1, max_tokens=100, max_ms=100)
        budget_mgr = BudgetManager(budget, mode="soft")

        # Use up budget
        budget_mgr.record_nodes(1)

        selector = BranchSelector(beam_selector=beam, budget_manager=budget_mgr)
        result = selector.step(sample_parent, sample_candidates)

        # Should not raise, but should_continue should be False
        assert result.should_continue is False
        assert result.kept == []
        assert "budget_exceeded" in result.details

    def test_step_empty_candidates(self, sample_parent):
        """Test step() with empty candidate list."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        selector = BranchSelector(beam_selector=beam)

        result = selector.step(sample_parent, [])

        assert result.kept == []
        assert result.scored == []
        assert result.after_beam == []

    def test_reset_entropy_stopper(self):
        """Test that reset() calls entropy stopper reset."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)

        # Mock entropy stopper
        class MockEntropyStopper:
            def __init__(self):
                self.reset_called = False

            def reset(self):
                self.reset_called = True

            def should_continue(self, kept):
                from chatroutes_autobranch.core.protocols import EntropyDecision
                return EntropyDecision(
                    should_continue=True,
                    entropy=0.8,
                    delta_entropy=None,
                    details={},
                )

        entropy = MockEntropyStopper()
        selector = BranchSelector(beam_selector=beam, entropy_stopper=entropy)

        selector.reset()
        assert entropy.reset_called is True

    def test_get_set_state(self):
        """Test state serialization and restoration."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
        budget_mgr = BudgetManager(budget, mode="strict")

        # Use some budget
        budget_mgr.record_nodes(5)

        selector = BranchSelector(beam_selector=beam, budget_manager=budget_mgr)

        # Get state
        state = selector.get_state()
        assert "budget_manager" in state

        # Restore to new selector
        budget_mgr2 = BudgetManager(budget, mode="strict")
        selector2 = BranchSelector(beam_selector=beam, budget_manager=budget_mgr2)
        selector2.set_state(state)

        assert selector2.budget_manager._used_nodes == 5

    def test_selection_result_structure(self, sample_parent, sample_candidates):
        """Test SelectionResult contains all expected fields."""
        beam = BeamSelector(scorer=StaticScorer({}), k=3)
        selector = BranchSelector(beam_selector=beam)

        result = selector.step(sample_parent, sample_candidates)

        # Check all fields exist
        assert hasattr(result, "kept")
        assert hasattr(result, "scored")
        assert hasattr(result, "after_beam")
        assert hasattr(result, "after_novelty")
        assert hasattr(result, "entropy_decision")
        assert hasattr(result, "should_continue")
        assert hasattr(result, "details")

        # Check types
        assert isinstance(result.kept, list)
        assert isinstance(result.scored, list)
        assert isinstance(result.after_beam, list)
        assert isinstance(result.details, dict)
        assert isinstance(result.should_continue, bool)
