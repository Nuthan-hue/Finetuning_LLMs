#!/usr/bin/env python3
"""
Test Phase 3: Data Analysis

Tests the DataAnalysisAgent's ability to:
- Identify train/test/submission files from CSV structure
- Detect data modality (tabular, NLP, vision, etc.)
- Identify target column
- Analyze feature types
- Recommend preprocessing steps
- Generate actionable insights

Usage:
    pytest tests/test_phase_3_data_analysis.py -v
    python tests/test_phase_3_data_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import pytest
from src.agents import AgenticOrchestrator
from src.agents.orchestrator.phases import run_data_collection, run_problem_understanding, run_data_analysis


async def _run_with_cache(phase_func, cache_file, *args, **kwargs):
    """Helper to run a phase with caching (TEST ONLY)"""
    # Check cache first
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Run phase
    result = await phase_func(*args, **kwargs)

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


@pytest.mark.asyncio
async def test_data_analysis_phase(competition_name: str = "titanic"):
    """Test Phase 3: Data Analysis with comprehensive validation"""

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

    # Setup cache directory (TEST ONLY - orchestrator doesn't use cache)
    cache_dir = Path("data") / competition_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Data Collection (with test-level caching)
    phase1_cache = cache_dir / "test_phase1_cache.json"
    if phase1_cache.exists():
        with open(phase1_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)
    else:
        context = await run_data_collection(orchestrator, context)
        with open(phase1_cache, 'w') as f:
            json.dump({"data_path": context["data_path"], "files": context["files"]}, f, indent=2)

    assert "data_path" in context
    assert "files" in context

    # Phase 2: Problem Understanding (with test-level caching)
    phase2_cache = cache_dir / "test_phase2_cache.json"
    if phase2_cache.exists():
        with open(phase2_cache, 'r') as f:
            cached = json.load(f)
            context.update(cached)
    else:
        context = await run_problem_understanding(orchestrator, context)
        with open(phase2_cache, 'w') as f:
            json.dump({
                "problem_understanding": context["problem_understanding"],
                "overview_text": context["overview_text"]
            }, f, indent=2)

    assert "problem_understanding" in context

    # Phase 3: Data Analysis (THE MAIN TEST - NO CACHE, always fresh)
    context = await run_data_analysis(orchestrator, context)

    # Validate results
    data_analysis = context.get("data_analysis")
    assert data_analysis is not None, "No data_analysis in context"

    # Validate file identification
    data_files = data_analysis.get("data_files", {})
    assert data_files.get("train_file"), "Train file not identified"
    assert data_files.get("test_file"), "Test file not identified"

    # Validate data modality detection
    data_modality = data_analysis.get("data_modality")
    assert data_modality in ["tabular", "nlp", "vision", "timeseries", "audio", "mixed"], \
        f"Invalid modality: {data_modality}"

    # Validate target column identification
    target_column = data_analysis.get("target_column")
    assert target_column, "Target column not identified"

    # Validate feature types analysis
    feature_types = data_analysis.get("feature_types", {})
    assert feature_types, "Feature types not analyzed"
    assert "id_columns" in feature_types

    # Validate data quality assessment
    data_quality = data_analysis.get("data_quality", {})
    assert "missing_values" in data_quality or "duplicates" in data_quality, \
        "Data quality not assessed"

    # Validate preprocessing recommendations
    preprocessing_required = data_analysis.get("preprocessing_required", False)
    if preprocessing_required:
        preprocessing = data_analysis.get("preprocessing", {})
        assert preprocessing, "Preprocessing required but no recommendations"

    # Validate context propagation
    assert context.get("target_column") == target_column, \
        "Target column not propagated to context"
    assert context.get("data_files") == data_files, \
        "File mapping not propagated to context"
    assert context.get("needs_preprocessing") == preprocessing_required, \
        "Preprocessing flag not propagated correctly"

    print(f"\n✅ Phase 3 test passed for {competition_name}")
    print(f"   Modality: {data_modality}, Target: {target_column}")
    print(f"   Files: {data_files.get('train_file')}, {data_files.get('test_file')}")

    return True


async def main():
    """Run the test as a standalone script"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"

    print(f"\n{'='*70}")
    print(f"  TESTING PHASE 3: DATA ANALYSIS")
    print(f"  Competition: {competition_name}")
    print(f"  Note: Phases 1-2 use test cache, Phase 3 runs fresh")
    print(f"{'='*70}\n")

    try:
        success = await test_data_analysis_phase(competition_name=competition_name)
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