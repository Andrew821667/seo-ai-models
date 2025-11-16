"""
Unit tests for AutoFix Engine.
"""

import pytest
from seo_ai_models.autofix.engine import AutoFixEngine, FixAction, FixComplexity, FixStatus
from seo_ai_models.autofix.fixers import MetaTagsFixer


class TestAutoFixEngine:
    """Tests for AutoFix Engine core functionality."""

    @pytest.fixture
    def engine(self):
        """Create AutoFix Engine instance."""
        return AutoFixEngine(auto_execute=True)

    @pytest.fixture
    def sample_analysis(self):
        """Sample analysis results with issues."""
        return {
            "missing_meta_tags": [
                {"page_id": "page1", "missing": ["title", "description"]},
                {"page_id": "page2", "missing": ["description"]}
            ],
            "missing_alt_tags": [
                {"src": "image1.jpg", "page": "page1"},
                {"src": "image2.jpg", "page": "page1"}
            ]
        }

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.auto_execute is True
        assert len(engine.actions_registry) == 0
        assert len(engine.executed_actions) == 0

    def test_register_action(self, engine):
        """Test registering fix actions."""
        fixer = MetaTagsFixer(llm_service=None)
        engine.register_fix_action("missing_meta_tags", fixer)

        assert "missing_meta_tags" in engine.actions_registry
        assert engine.actions_registry["missing_meta_tags"] == fixer

    def test_analyze_and_plan(self, engine, sample_analysis):
        """Test creating fix plan from analysis."""
        # Register fixer
        fixer = MetaTagsFixer(llm_service=None)
        engine.register_fix_action("missing_meta_tags", fixer)

        # Create plan
        plan = engine.analyze_and_plan(sample_analysis)

        assert len(plan) > 0
        assert all(isinstance(action, FixAction) for action in plan)
        assert all(action.status == FixStatus.PENDING for action in plan)

    def test_fix_complexity_levels(self):
        """Test fix complexity enum."""
        assert FixComplexity.TRIVIAL.value == "trivial"
        assert FixComplexity.SIMPLE.value == "simple"
        assert FixComplexity.MODERATE.value == "moderate"
        assert FixComplexity.COMPLEX.value == "complex"
        assert FixComplexity.CRITICAL.value == "critical"

    def test_fix_action_creation(self):
        """Test FixAction dataclass."""
        action = FixAction(
            action_type="test_fix",
            complexity=FixComplexity.SIMPLE,
            description="Test fix",
            metadata={"test": "data"},
            estimated_impact="high"
        )

        assert action.action_type == "test_fix"
        assert action.complexity == FixComplexity.SIMPLE
        assert action.status == FixStatus.PENDING
        assert action.metadata == {"test": "data"}

    def test_execute_plan_with_auto_execute(self, engine, sample_analysis):
        """Test executing fix plan with auto-execute enabled."""
        fixer = MetaTagsFixer(llm_service=None)
        engine.register_fix_action("missing_meta_tags", fixer)

        plan = engine.analyze_and_plan(sample_analysis)

        # Execute plan (simple fixes should auto-execute)
        result = engine.execute_plan(
            plan,
            require_approval_for=[FixComplexity.COMPLEX, FixComplexity.CRITICAL]
        )

        assert "executed" in result
        assert "pending_approval" in result
        assert "failed" in result

    def test_backup_and_rollback(self, engine):
        """Test backup mechanism."""
        # Test backup creation
        backup = engine._create_backup("test_page", {"content": "original"})

        assert "page_id" in backup
        assert "timestamp" in backup
        assert backup["original_data"] == {"content": "original"}

    def test_priority_calculation(self, engine):
        """Test priority score calculation."""
        # High impact, low complexity = high priority
        action1 = FixAction(
            action_type="test",
            complexity=FixComplexity.TRIVIAL,
            description="Test",
            estimated_impact="high"
        )

        # Low impact, high complexity = low priority
        action2 = FixAction(
            action_type="test",
            complexity=FixComplexity.COMPLEX,
            description="Test",
            estimated_impact="low"
        )

        score1 = engine._calculate_priority_score(action1)
        score2 = engine._calculate_priority_score(action2)

        assert score1 > score2
