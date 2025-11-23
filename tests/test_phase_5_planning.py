#!/usr/bin/env python3
"""
Test Phase 5: Planning
Tests the PlanningAgent's ability to create execution strategies.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_planning


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        pass


async def test_phase_5_planning():
    """
    Test Phase 5: Planning

    Tests:
    - Generate execution plan
    - Select models to try
    - Determine if feature engineering is needed
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 5: PLANNING - TEST")
    print("=" * 70)

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": "titanic",
        "data_path": "data/titanic",
        "problem_understanding": {
            "competition_type": "binary_classification",
            "evaluation_metric": "accuracy"
        },
        "data_analysis": {
            "data_modality": "tabular",
            "target_column": "Survived"
        }
    }

    try:
        print("\n🎯 Running Phase 5: Planning...")
        context = await run_planning(orchestrator, context)

        # Validate outputs
        assert "execution_plan" in context, "Missing execution_plan in context"
        assert "models_to_try" in context["execution_plan"], "Missing models_to_try in plan"
        assert "needs_feature_engineering" in context, "Missing needs_feature_engineering flag"
        assert len(context["execution_plan"]["models_to_try"]) > 0, "No models in execution plan"

        print("\n✅ Phase 5 PASSED")
        print(f"   Strategy: {context['execution_plan'].get('strategy_summary', 'N/A')}")
        print(f"   Models: {[m.get('model') for m in context['execution_plan']['models_to_try']]}")
        print(f"   Feature Engineering: {context.get('needs_feature_engineering')}")

        return True

    except Exception as e:
        print(f"\n❌ Phase 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 5 test"""
    success = await test_phase_5_planning()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())