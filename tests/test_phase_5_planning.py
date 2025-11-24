#!/usr/bin/env python3
"""
Test Phase 5: Planning
Tests the PlanningAgent's ability to create execution strategies.
"""
import asyncio
import sys
import json
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_planning


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        pass

@pytest.mark.asyncio
async def test_phase_5_planning(competition_name: str = "titanic"):
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

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase5_cache.json"

    # Check if we already have cached data
    if cache_file.exists():
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {
            "competition_name": competition_name,
            "data_path": f"data/{competition_name}",
        }
        context.update(cached)

        # Validate cached outputs
        assert "execution_plan" in context, "Missing execution_plan in cache"
        assert "models_to_try" in context["execution_plan"], "Missing models_to_try in cache"
        assert "needs_feature_engineering" in context, "Missing needs_feature_engineering in cache"

        print("\n✅ Phase 5 PASSED (cached)")
        print(f"   Strategy: {context['execution_plan'].get('strategy_summary', 'N/A')}")
        print(f"   Models: {[m.get('model') for m in context['execution_plan']['models_to_try']]}")
        print(f"   Feature Engineering: {context.get('needs_feature_engineering')}")
        return True

    # No cache - need Phase 2 & 3 data first
    # Load from their caches or use mock data
    phase2_cache = test_cache_dir / "test_phase2_cache.json"
    phase3_cache = test_cache_dir / "test_phase3_cache.json"

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": competition_name,
        "data_path": f"data/{competition_name}",
    }

    # Load Phase 2 data if available
    if phase2_cache.exists():
        with open(phase2_cache, 'r') as f:
            context.update(json.load(f))
    else:
        # Use mock data
        context["problem_understanding"] = {
            "competition_type": "binary_classification",
            "evaluation_metric": "accuracy"
        }

    # Load Phase 3 data if available
    if phase3_cache.exists():
        with open(phase3_cache, 'r') as f:
            context.update(json.load(f))
    else:
        # Use mock data
        context["data_analysis"] = {
            "data_modality": "tabular",
            "target_column": "Survived"
        }

    try:
        print("\n🎯 Running Phase 5: Planning...")
        context = await run_planning(orchestrator, context)

        # Validate outputs
        assert "execution_plan" in context, "Missing execution_plan in context"
        assert "models_to_try" in context["execution_plan"], "Missing models_to_try in plan"
        assert "needs_feature_engineering" in context, "Missing needs_feature_engineering flag"
        assert len(context["execution_plan"]["models_to_try"]) > 0, "No models in execution plan"

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "execution_plan": context["execution_plan"],
                "needs_feature_engineering": context["needs_feature_engineering"]
            }, f, indent=2)

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
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_5_planning(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())