import pytest
from scholar_lens.core.token_tracker import TokenTracker, TokenUsage


class TestTokenUsage:
    def test_create(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
        assert u.total_tokens == 150

    def test_defaults(self):
        u = TokenUsage()
        assert u.total_tokens == 0


class TestTokenTracker:
    def test_record_and_total(self):
        tracker = TokenTracker()
        tracker.record("agent1", prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
        tracker.record("agent1", prompt_tokens=200, completion_tokens=100, model="gpt-4o-mini")
        total = tracker.get_total("agent1")
        assert total.prompt_tokens == 300
        assert total.completion_tokens == 150
        assert total.total_tokens == 450

    def test_multiple_agents(self):
        tracker = TokenTracker()
        tracker.record("explainer", prompt_tokens=100, completion_tokens=50, model="m1")
        tracker.record("tutor", prompt_tokens=200, completion_tokens=80, model="m2")
        assert tracker.get_total("explainer").total_tokens == 150
        assert tracker.get_total("tutor").total_tokens == 280
        assert tracker.get_grand_total().total_tokens == 430

    def test_unknown_agent_returns_zero(self):
        tracker = TokenTracker()
        total = tracker.get_total("nonexistent")
        assert total.total_tokens == 0

    def test_interaction_budget_check(self):
        tracker = TokenTracker()
        tracker.record("tutor", prompt_tokens=3000, completion_tokens=800, model="m")
        tracker.record("tutor", prompt_tokens=1000, completion_tokens=400, model="m")
        within = tracker.is_within_budget("tutor", budget=4600)
        assert within is False  # 5200 > 4600

    def test_summary(self):
        tracker = TokenTracker()
        tracker.record("a", prompt_tokens=100, completion_tokens=50, model="m")
        summary = tracker.summary()
        assert "a" in summary
        assert summary["a"]["total"] == 150
