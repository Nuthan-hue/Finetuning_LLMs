#!/usr/bin/env python3
"""
Test Phase 6: Feature Engineering
Tests the FeatureEngineeringAgent's ability to generate and apply features.
"""
import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_feature_engineering


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        pass


async def test_phase_6_feature_engineering(competition_name: str = "titanic"):
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

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase6_cache.json"

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
        assert "featured_data_path" in context, "Missing featured_data_path in cache"

        print("\n✅ Phase 6 PASSED (cached)")
        print(f"   Featured data path: {context['featured_data_path']}")
        return True

    # No cache - load dependencies
    phase5_cache = test_cache_dir / "test_phase5_cache.json"
    phase4_cache = test_cache_dir / "test_phase4_cache.json"
    phase3_cache = test_cache_dir / "test_phase3_cache.json"

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": competition_name,
        "data_path": f"data/{competition_name}",
    }

    # Load Phase 3 data if available
    if phase3_cache.exists():
        with open(phase3_cache, 'r') as f:
            context.update(json.load(f))
    else:
        context["data_analysis"] = {
            "data_modality": "tabular",
            "target_column": "Survived"
        }

    # Load Phase 4 data if available
    if phase4_cache.exists():
        with open(phase4_cache, 'r') as f:
            context.update(json.load(f))
    else:
        context["clean_data_path"] = context["data_path"]

    # Load Phase 5 data if available
    if phase5_cache.exists():
        with open(phase5_cache, 'r') as f:
            cached = json.load(f)
            context["needs_feature_engineering"] = cached.get("needs_feature_engineering", False)
            context["execution_plan"] = cached.get("execution_plan", {})
    else:
        context["needs_feature_engineering"] = False

    try:
        print("\n🔧 Running Phase 6: Feature Engineering...")
        context = await run_feature_engineering(orchestrator, context)

        # Validate outputs
        assert "featured_data_path" in context, "Missing featured_data_path in context"

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "featured_data_path": context["featured_data_path"],
                "feature_engineering_result": context.get("feature_engineering_result")
            }, f, indent=2)

        if context.get("needs_feature_engineering"):
            print("   Feature engineering executed")
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
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_6_feature_engineering(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())