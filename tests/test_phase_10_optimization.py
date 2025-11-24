#!/usr/bin/env python3
"""
Test Phase 10: Optimization
Tests the StrategyAgent's ability to decide next optimization steps.
"""
import asyncio
import sys
import json
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_optimization


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self, target_percentile=0.20):
        self.target_percentile = target_percentile
        self.iteration = 1
        self.tried_models = ["lightgbm"]
        self.workflow_history = []

@pytest.mark.asyncio
async def test_phase_10_optimization(competition_name: str = "titanic"):
    """
    Test Phase 10: Optimization

    Tests:
    - AI decides optimization strategy
    - Determines where to loop back
    - Or skips if target achieved
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 10: OPTIMIZATION - TEST")
    print("=" * 70)

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase10_cache.json"

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 1: CHECK IF CACHE EXISTS
    # ═══════════════════════════════════════════════════════════════════
    if cache_file.exists():
        # ─────────────────────────────────────────────────────────────
        # CASE 1A: Cache exists - Load and return early
        # ─────────────────────────────────────────────────────────────
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {"competition_name": competition_name}
        context.update(cached)

        # ─────────────────────────────────────────────────────────────
        # NESTED BLOCK: Check if optimization was needed or skipped
        # ─────────────────────────────────────────────────────────────
        if context.get("optimization_strategy"):
            # CASE 1A-1: Strategy exists (needs improvement)
            print("\n✅ Phase 10 PASSED (cached)")
            print(f"   Strategy: {context.get('optimization_strategy', {}).get('action', 'N/A')}")
            print(f"   Loop back to: {context.get('optimization_strategy', {}).get('loop_back_to', 'N/A')}")
        else:
            # CASE 1A-2: Strategy is None (target achieved, phase was skipped)
            print("\n✅ Phase 10 PASSED (cached - skipped)")
            print("   Target achieved, no optimization needed")

        return True  # Early exit - don't run phase again

    else:
        # ─────────────────────────────────────────────────────────────
        # CASE 1B: Cache does not exist - Run phase from scratch
        # ─────────────────────────────────────────────────────────────
        print("💾 No cache found, running fresh...")

        # Load dependencies from previous phases
        phase3_cache = test_cache_dir / "test_phase3_cache.json"
        phase7_cache = test_cache_dir / "test_phase7_cache.json"
        phase9_cache = test_cache_dir / "test_phase9_cache.json"

        orchestrator = MockOrchestrator(target_percentile=0.20)
        context = {
            "competition_name": competition_name,
        }

        # ═════════════════════════════════════════════════════════════
        # BLOCK 2: LOAD PHASE 3 DATA (data modality)
        # ═════════════════════════════════════════════════════════════
        if phase3_cache.exists():
            # CASE 2A: Phase 3 cache exists - Load data modality
            with open(phase3_cache, 'r') as f:
                cached_phase3 = json.load(f)
                data_analysis = cached_phase3.get("data_analysis", {})
                context["data_modality"] = data_analysis.get("data_modality", "tabular")
            print("   Loaded Phase 3: data_modality")
        else:
            # CASE 2B: Phase 3 cache missing - Skip (AI will handle missing data)
            print("   ⚠️  Phase 3 cache not found, skipping data_modality")

        # ═════════════════════════════════════════════════════════════
        # BLOCK 3: LOAD PHASE 7 DATA (model type)
        # ═════════════════════════════════════════════════════════════
        if phase7_cache.exists():
            # CASE 3A: Phase 7 cache exists - Load model type
            with open(phase7_cache, 'r') as f:
                cached_phase7 = json.load(f)
                context["model_type"] = cached_phase7.get("model_type", "lightgbm")
            print("   Loaded Phase 7: model_type")
        else:
            # CASE 3B: Phase 7 cache missing - Skip
            print("   ⚠️  Phase 7 cache not found, skipping model_type")

        # ═════════════════════════════════════════════════════════════
        # BLOCK 4: LOAD PHASE 9 DATA (evaluation results) - CRITICAL!
        # ═════════════════════════════════════════════════════════════
        if phase9_cache.exists():
            # CASE 4A: Phase 9 cache exists - Load all evaluation data
            with open(phase9_cache, 'r') as f:
                cached_phase9 = json.load(f)
                context.update(cached_phase9)
            print("   Loaded Phase 9: evaluation results (needs_improvement, percentile, etc.)")
        else:
            # CASE 4B: Phase 9 cache missing - Use mock fallback data
            print("   ⚠️  Phase 9 cache not found, using mock evaluation data")
            context["needs_improvement"] = True
            context["current_percentile"] = 1.0
            context["leaderboard_results"] = {"recommendation": "improve"}

        # ═════════════════════════════════════════════════════════════
        # BLOCK 5: RUN PHASE 10 OPTIMIZATION
        # ═════════════════════════════════════════════════════════════
        try:
            # CASE 5A: Try to run phase successfully
            print("\n🔄 Running Phase 10: Optimization...")
            print("   (AI deciding strategy...)")

            context = await asyncio.wait_for(
                run_optimization(orchestrator, context),
                timeout=120
            )

            assert "optimization_strategy" in context, "Missing optimization_strategy"

            # ─────────────────────────────────────────────────────────
            # Save to cache (only Phase 10's output)
            # ─────────────────────────────────────────────────────────
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    "optimization_strategy": context.get("optimization_strategy")
                }, f, indent=2)

            # ─────────────────────────────────────────────────────────
            # NESTED BLOCK: Display results based on strategy
            # ─────────────────────────────────────────────────────────
            if context.get("optimization_strategy"):
                # CASE 5A-1: Strategy exists (needs improvement)
                print("\n✅ Phase 10 PASSED")
                print(f"   Strategy: {context.get('optimization_strategy', {}).get('action', 'N/A')}")
                print(f"   Loop back to: {context.get('optimization_strategy', {}).get('loop_back_to', 'N/A')}")
            else:
                # CASE 5A-2: Strategy is None (target achieved, phase skipped)
                print("\n✅ Phase 10 PASSED (skipped)")
                print("   Target achieved, no optimization needed")

            return True

        except asyncio.TimeoutError:
            # CASE 5B: Phase timed out after 120 seconds
            print("\n❌ Phase 10 TIMED OUT")
            print("   The AI took too long to decide optimization strategy")
            return False

        except Exception as e:
            # CASE 5C: Phase failed with error
            print(f"\n❌ Phase 10 FAILED: {e}")
            print("   An unexpected error occurred during optimization")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run Phase 10 test"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_10_optimization(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())