#!/usr/bin/env python3
"""
Test Phase 3: Data Analysis

This test specifically targets the Data Analysis phase with comprehensive coverage.
It tests the DataAnalysisAgent's ability to:
- Identify train/test/submission files from CSV structure
- Detect data modality (tabular, NLP, vision, etc.)
- Identify target column
- Analyze feature types
- Recommend preprocessing steps
- Generate actionable insights

Usage:
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


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")


@pytest.mark.asyncio
async def test_data_analysis_phase(competition_name: str = "google-tunix-hackathon"):
    """Comprehensive test for Phase 3: Data Analysis

    Args:
        competition_name: Name of the Kaggle competition to test (default: "titanic")
    """

    print_header("PHASE 3: DATA ANALYSIS - COMPREHENSIVE TEST")
    print_info(f"Testing competition: {competition_name}")

    # Initialize orchestrator
    print_info("Initializing orchestrator...")
    orchestrator = AgenticOrchestrator(
        competition_name=competition_name,
        target_percentile=0.20,
        max_actions=50
    )
    print_success("Orchestrator initialized")

    # Build context progressively
    context = {
        "competition_name": competition_name,
        "target_percentile": 0.20,
        "iteration": 0
    }

    # Check if Phase 1 data already exists locally
    phase1_cache = Path("data") / competition_name / "phase1_data_collection.json"
    phase1_exists = phase1_cache.exists()

    # Phase 1: Data Collection (required for Phase 3)
    # Smart: Only runs if data doesn't exist
    print_header("PREREQUISITE: Phase 1 - Data Collection")
    if phase1_exists:
        print_info("Phase 1 cache found - will load from disk")
    else:
        print_info("Phase 1 cache not found - will download data")

    try:
        context = await run_data_collection(orchestrator, context)
        if phase1_exists:
            print_success("Phase 1 loaded from cache (0 downloads, 0 API calls)")
        else:
            print_success("Phase 1 completed - data downloaded")
        print_info(f"Data path: {context['data_path']}")
        print_info(f"Files found: {len(context.get('files', []))}")
    except Exception as e:
        print_error(f"Data collection failed: {e}")
        return False

    # Check if Phase 2 data already exists locally
    phase2_cache = Path("data") / competition_name / "phase2_problem_understanding.json"
    phase2_exists = phase2_cache.exists()

    # Phase 2: Problem Understanding (required for Phase 3)
    # Smart: Only runs if understanding doesn't exist
    print_header("PREREQUISITE: Phase 2 - Problem Understanding")
    if phase2_exists:
        print_info("Phase 2 cache found - will load from disk")
    else:
        print_info("Phase 2 cache not found - will analyze problem (1 API call)")

    try:
        context = await run_problem_understanding(orchestrator, context)
        if phase2_exists:
            print_success("Phase 2 loaded from cache (0 API calls)")
        else:
            print_success("Phase 2 completed - problem analyzed (1 API call)")
        print_info(f"Task type: {context['problem_understanding'].get('competition_type', 'N/A')}")
        print_info(f"Metric: {context['problem_understanding'].get('evaluation_metric', 'N/A')}")
    except Exception as e:
        print_error(f"Problem understanding failed: {e}")
        return False

    # PHASE 3: DATA ANALYSIS (THE MAIN TEST)
    print_header("MAIN TEST: Phase 3 - Data Analysis")

    try:
        # Execute data analysis phase
        print_info("Running data analysis agent...")
        context = await run_data_analysis(orchestrator, context)

        # Validate results
        print_header("VALIDATION: Checking Data Analysis Results")

        data_analysis = context.get("data_analysis")
        if not data_analysis:
            print_error("No data_analysis in context!")
            return False

        # 1. Check file identification
        print_info("Validating file identification...")
        data_files = data_analysis.get("data_files", {})
        train_file = data_files.get("train_file")
        test_file = data_files.get("test_file")
        submission_file = data_files.get("submission_file")

        if train_file:
            print_success(f"Train file identified: {train_file}")
        else:
            print_error("Train file not identified!")
            return False

        if test_file:
            print_success(f"Test file identified: {test_file}")
        else:
            print_error("Test file not identified!")
            return False

        if submission_file:
            print_success(f"Submission file identified: {submission_file}")
        else:
            print_info("No submission file identified (may not exist)")

        # 2. Check data modality detection
        print_info("Validating data modality detection...")
        data_modality = data_analysis.get("data_modality")
        if data_modality:
            print_success(f"Data modality detected: {data_modality}")
            if data_modality not in ["tabular", "nlp", "vision", "timeseries", "audio", "mixed"]:
                print_error(f"Invalid modality: {data_modality}")
                return False
        else:
            print_error("Data modality not detected!")
            return False

        # 3. Check target column identification
        print_info("Validating target column identification...")
        target_column = data_analysis.get("target_column")
        if target_column:
            print_success(f"Target column identified: {target_column}")
        else:
            print_error("Target column not identified!")
            return False

        # 4. Check feature types analysis
        print_info("Validating feature types analysis...")
        feature_types = data_analysis.get("feature_types", {})
        if feature_types:
            print_success(f"Feature types analyzed:")
            print_info(f"  - Numerical: {feature_types.get('numerical_columns', [])}")
            print_info(f"  - Categorical: {feature_types.get('categorical_columns', [])}")
            print_info(f"  - ID columns: {feature_types.get('id_columns', [])}")
            print_info(f"  - Text columns: {feature_types.get('text_columns', [])}")
        else:
            print_error("Feature types not analyzed!")
            return False

        # 5. Check data quality assessment
        print_info("Validating data quality assessment...")
        data_quality = data_analysis.get("data_quality", {})
        if data_quality:
            print_success("Data quality assessed:")
            print_info(f"  - Missing values: {data_quality.get('missing_values', {})}")
            print_info(f"  - Duplicates: {data_quality.get('duplicates', 0)}")
        else:
            print_info("Data quality not fully assessed")

        # 6. Check preprocessing recommendations
        print_info("Validating preprocessing recommendations...")
        preprocessing_required = data_analysis.get("preprocessing_required", False)
        print_info(f"Preprocessing required: {preprocessing_required}")

        if preprocessing_required:
            preprocessing = data_analysis.get("preprocessing", {})
            if preprocessing:
                print_success("Preprocessing recommendations provided:")
                print_info(f"  - Recommendations: {preprocessing}")
            else:
                print_error("Preprocessing required but no recommendations!")
                return False

        # 7. Check key insights
        print_info("Validating key insights...")
        key_insights = data_analysis.get("key_insights", [])
        if key_insights:
            print_success(f"Key insights generated ({len(key_insights)} insights):")
            for i, insight in enumerate(key_insights[:5], 1):
                print_info(f"  {i}. {insight}")
        else:
            print_info("No key insights generated")

        # 8. Verify context propagation
        print_header("VALIDATION: Context Propagation")

        if context.get("target_column") == target_column:
            print_success(f"Target column propagated to context: {target_column}")
        else:
            print_error("Target column not propagated correctly!")
            return False

        if context.get("data_files") == data_files:
            print_success("File mapping propagated to context")
        else:
            print_error("File mapping not propagated correctly!")
            return False

        if context.get("needs_preprocessing") == preprocessing_required:
            print_success(f"Preprocessing flag propagated: {preprocessing_required}")
        else:
            print_error("Preprocessing flag not propagated correctly!")
            return False

        # 9. Check saved artifacts
        print_header("VALIDATION: Saved Artifacts")

        analysis_file = Path(context["data_path"]) / "data_analysis.json"
        if analysis_file.exists():
            print_success(f"Data analysis saved to: {analysis_file}")

            # Verify file content
            with open(analysis_file, 'r') as f:
                saved_analysis = json.load(f)

            if saved_analysis == data_analysis:
                print_success("Saved analysis matches in-memory analysis")
            else:
                print_error("Saved analysis doesn't match in-memory analysis!")
                return False
        else:
            print_error(f"Data analysis file not created: {analysis_file}")
            return False

        # 10. Final summary
        print_header("TEST SUMMARY")
        print_success("Phase 3: Data Analysis - ALL CHECKS PASSED!")

        print_info("\nKey Metrics:")
        print_info(f"  - Train file: {train_file}")
        print_info(f"  - Test file: {test_file}")
        print_info(f"  - Data modality: {data_modality}")
        print_info(f"  - Target column: {target_column}")
        print_info(f"  - Numerical features: {len(feature_types.get('numerical_columns', []))}")
        print_info(f"  - Categorical features: {len(feature_types.get('categorical_columns', []))}")
        print_info(f"  - Preprocessing required: {preprocessing_required}")
        print_info(f"  - Key insights: {len(key_insights)}")

        return True

    except Exception as e:
        print_error(f"Data analysis phase failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    print_header("DATA ANALYSIS PHASE - COMPREHENSIVE TEST")

    # Check if competition name provided as command-line argument
    if len(sys.argv) > 1:
        competition_name = sys.argv[1]
        print_info(f"Testing competition from command line: {competition_name}")
    else:
        competition_name = "titanic"
        print_info(f"Testing default competition: {competition_name}")

    print_info("Testing Phase 3 with full validation")
    print()

    # Run test with the specified competition
    success = await test_data_analysis_phase(competition_name=competition_name)

    print_header("FINAL RESULT")
    if success:
        print_success("✨ ALL TESTS PASSED! ✨")
        print_info(f"Phase 3 (Data Analysis) is working correctly for {competition_name}")
        return 0
    else:
        print_error("❌ TESTS FAILED!")
        print_info(f"Phase 3 (Data Analysis) has issues for {competition_name}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)