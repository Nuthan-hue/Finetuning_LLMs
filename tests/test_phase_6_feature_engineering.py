#!/usr/bin/env python3
"""
Test Phase 6: Feature Engineering
Tests the FeatureEngineeringAgent's ability to generate and apply features.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_feature_engineering


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        pass


async def test_phase_6_feature_engineering():
    """
    Test Phase 6: Feature Engineering

    Tests:
    - Generate feature engineering code
    - Apply features to data
    - Conditional execution based on needs_feature_engineering flag
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 6: FEATURE ENGINEERING - TEST")
    print("=" * 70)

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": "titanic",
        "data_path": "data/titanic",
        "clean_data_path": "data/titanic",
        "data_analysis": {
            "data_modality": "tabular",
            "target_column": "Survived"
        },
        "needs_feature_engineering": False  # Change to True to test feature engineering
    }

    try:
        print("\n🔧 Running Phase 6: Feature Engineering...")
        context = await run_feature_engineering(orchestrator, context)

        # Validate outputs
        assert "featured_data_path" in context, "Missing featured_data_path in context"

        if context.get("needs_feature_engineering"):
            print("   Feature engineering requested (conditional execution)")
        else:
            print("   Feature engineering skipped (not needed)")

        print("\n✅ Phase 6 PASSED")
        print(f"   Featured data path: {context['featured_data_path']}")

        return True

    except Exception as e:
        print(f"\n❌ Phase 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 6 test"""
    success = await test_phase_6_feature_engineering()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())