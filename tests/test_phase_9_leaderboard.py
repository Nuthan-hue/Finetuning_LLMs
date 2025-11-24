#!/usr/bin/env python3
"""
Test Phase 9: Evaluation/Leaderboard
Tests the LeaderboardMonitor's ability to check competition status.
"""
import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_evaluation


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self, target_percentile=0.20):
        from src.agents.leaderboard import LeaderboardMonitor
        self.leaderboard_monitor = LeaderboardMonitor()
        self.target_percentile = target_percentile
        self.iteration = 1


async def test_phase_9_evaluation(competition_name: str = "titanic"):
    """
    Test Phase 9: Evaluation/Leaderboard

    Tests:
    - Fetch leaderboard data
    - Check current rank and percentile
    - Determine if target is met
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 9: EVALUATION - TEST")
    print("=" * 70)

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase9_cache.json"

    # Check cache
    if cache_file.exists():
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {"competition_name": competition_name}
        context.update(cached)

        print("\n✅ Phase 9 PASSED (cached)")
        print(f"   Current rank: {context.get('current_rank', 'N/A')}")
        print(f"   Current percentile: {context.get('current_percentile', 'N/A')}")
        print(f"   Meets target: {context.get('meets_target', False)}")
        return True

    # Load dependencies from previous phases
    phase8_cache = test_cache_dir / "test_phase8_cache.json"

    orchestrator = MockOrchestrator(target_percentile=0.20)
    context = {
        "competition_name": competition_name,
    }

    # Load Phase 8 data (submission info)
    if phase8_cache.exists():
        with open(phase8_cache, 'r') as f:
            cached_phase8 = json.load(f)
            context.update(cached_phase8)

    try:
        print("\n📊 Running Phase 9: Evaluation...")
        print("   (Checking leaderboard...)")

        context = await asyncio.wait_for(
            run_evaluation(orchestrator, context),
            timeout=120
        )

        assert "current_rank" in context, "Missing current_rank"
        assert "current_percentile" in context, "Missing current_percentile"
        assert "meets_target" in context, "Missing meets_target"

        # Save to cache (only JSON-serializable data)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "current_rank": context.get("current_rank"),
                "current_percentile": float(context.get("current_percentile")) if context.get("current_percentile") is not None else None,
                "meets_target": context.get("meets_target"),
                "needs_improvement": context.get("needs_improvement"),
                "leaderboard_results": context.get("leaderboard_results")
            }, f, indent=2)

        print("\n✅ Phase 9 PASSED")
        print(f"   Current rank: {context.get('current_rank', 'N/A')}")
        print(f"   Current percentile: {context.get('current_percentile', 'N/A')}")
        print(f"   Meets target: {context.get('meets_target', False)}")
        return True

    except asyncio.TimeoutError:
        print("\n❌ Phase 9 TIMED OUT")
        return False
    except Exception as e:
        print(f"\n❌ Phase 9 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 9 test"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_9_evaluation(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())