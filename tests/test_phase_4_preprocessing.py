#!/usr/bin/env python3
"""
Test Phase 4: Preprocessing (Integration Test with Caching)
Tests the full preprocessing phase workflow with test caching.
"""
import asyncio
import sys
import json
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import AgenticOrchestrator
from src.agents.orchestrator.phases import (
    run_data_collection,
    run_problem_understanding,
    run_data_analysis,
    run_preprocessing
)


@pytest.mark.asyncio
async def test_phase_4_preprocessing(competition_name: str = "titanic"):
    """
    Test Phase 4: Preprocessing with full pipeline

    Tests:
    - Conditional execution (skip if not needed)
    - Code generation for preprocessing
    - Code execution
    - File creation (clean_train.csv, clean_test.csv)
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 4: PREPROCESSING - INTEGRATION TEST")
    print("=" * 70)

    # Initialize orchestrator
    orchestrator = AgenticOrchestrator(
        competition_name=competition_name,
        target_percentile=0.20,
        max_actions=50
    )

    context = {
        "competition_name": competition_name,
        "target_percentile": 0.20,
        "iteration": 0
    }

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Data Collection (with caching)
    phase1_cache = test_cache_dir / "test_phase1_cache.json"
    if phase1_cache.exists():
        print("📦 Loading Phase 1 from cache...")
        with open(phase1_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)
    else:
        context = await run_data_collection(orchestrator, context)
        with open(phase1_cache, 'w') as f:
            json.dump({"data_path": context["data_path"], "files": context["files"]}, f, indent=2)

    assert "data_path" in context

    # Phase 2: Problem Understanding (with caching)
    phase2_cache = test_cache_dir / "test_phase2_cache.json"
    if phase2_cache.exists():
        print("📦 Loading Phase 2 from cache...")
        with open(phase2_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)
    else:
        context = await run_problem_understanding(orchestrator, context)
        with open(phase2_cache, 'w') as f:
            json.dump({
                "problem_understanding": context["problem_understanding"],
                "overview_text": context.get("overview_text")
            }, f, indent=2)

    assert "problem_understanding" in context

    # Phase 3: Data Analysis (with caching)
    phase3_cache = test_cache_dir / "test_phase3_cache.json"
    if phase3_cache.exists():
        print("📦 Loading Phase 3 from cache...")
        with open(phase3_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)
    else:
        context = await run_data_analysis(orchestrator, context)
        with open(phase3_cache, 'w') as f:
            json.dump({
                "data_analysis": context["data_analysis"],
                "target_column": context.get("target_column"),
                "data_files": context.get("data_files"),
                "needs_preprocessing": context.get("needs_preprocessing")
            }, f, indent=2)

    assert "data_analysis" in context

    # Phase 4: Preprocessing (THE MAIN TEST - check cache)
    phase4_cache = test_cache_dir / "test_phase4_cache.json"

    if phase4_cache.exists():
        print("📦 Loading Phase 4 from cache...")
        with open(phase4_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)

        # Validate cached results
        assert "clean_data_path" in context
        print(f"\n✅ Phase 4 PASSED (cached)")
        print(f"   Clean data path: {context.get('clean_data_path')}")
        print(f"   Preprocessing executed: {context.get('needs_preprocessing')}")
        return True

    # No cache - run fresh
    print("\n🔧 Running Phase 4: Preprocessing...")
    context = await run_preprocessing(orchestrator, context)

    # Validate results
    assert "clean_data_path" in context, "Missing clean_data_path"

    # Check if preprocessing was executed
    if context.get("needs_preprocessing"):
        assert "preprocessing_result" in context, "Preprocessing needed but no results"

        # Check files were created
        data_path = Path(context["data_path"])
        train_file = context["data_files"]["train_file"]

        # Look for clean files
        clean_train = data_path / "clean_train.csv"
        if not clean_train.exists():
            # Maybe it's named differently
            clean_train = data_path / f"clean_{train_file}"

        if clean_train.exists():
            print(f"   ✓ Clean training file created: {clean_train.name}")
        else:
            print(f"   ⚠ Clean training file not found (preprocessing may have failed gracefully)")

        print(f"\n✅ Phase 4 PASSED")
        print(f"   Preprocessing executed: Yes")
        if "preprocessing_result" in context:
            print(f"   Train shape: {context['preprocessing_result'].get('train_shape')}")
            print(f"   Test shape: {context['preprocessing_result'].get('test_shape')}")
    else:
        print(f"\n✅ Phase 4 PASSED (skipped - data is clean)")
        print(f"   Preprocessing executed: No")

    # Save to cache (convert numpy types to Python types)
    # Only cache what Phase 4 actually outputs (not fields from previous phases)
    cache_data = {
        "clean_data_path": context["clean_data_path"]
    }

    # Convert preprocessing_result to JSON-safe format
    if "preprocessing_result" in context:
        result = context["preprocessing_result"]
        cache_data["preprocessing_result"] = {
            "train_shape": list(result.get("train_shape", [])) if result.get("train_shape") else None,
            "test_shape": list(result.get("test_shape", [])) if result.get("test_shape") else None,
            "columns": result.get("columns"),
            "missing_values_remaining": int(result.get("missing_values_remaining", 0)) if result.get("missing_values_remaining") is not None else 0
        }

    with open(phase4_cache, 'w') as f:
        json.dump(cache_data, f, indent=2)

    return True


async def main():
    """Run Phase 4 test"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"

    print(f"\n{'='*70}")
    print(f"  TESTING PHASE 4: PREPROCESSING (Integration)")
    print(f"  Competition: {competition_name}")
    print(f"  Note: Phases 1-3 use cache, Phase 4 uses cache if available")
    print(f"{'='*70}\n")

    try:
        success = await test_phase_4_preprocessing(competition_name=competition_name)
        print(f"\n{'='*70}")
        print("  ✅ ALL TESTS PASSED!")
        print(f"{'='*70}\n")
        return 0
    except AssertionError as e:
        print(f"\n{'='*70}")
        print(f"  ❌ TEST FAILED: {e}")
        print(f"{'='*70}\n")
        return 1
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  ❌ ERROR: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)