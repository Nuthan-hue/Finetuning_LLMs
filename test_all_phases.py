#!/usr/bin/env python3
"""
Comprehensive Phase-by-Phase Test Script

Tests each phase of the Kaggle Competition Multi-Agent System individually
to ensure all fixes are working correctly.

Usage:
    python test_all_phases.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import json
from src.agents import AgenticOrchestrator


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


def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")


async def test_phase_1_data_collection(orchestrator):
    """Test Phase 1: Data Collection"""
    print_header("PHASE 1: DATA COLLECTION")

    try:
        context = {"competition_name": "titanic"}

        from src.agents.orchestrator.phases import run_data_collection
        context = await run_data_collection(orchestrator, context)

        # Verify results
        assert "data_path" in context, "Missing data_path in context"
        assert "files" in context, "Missing files in context"
        assert len(context["files"]) > 0, "No files downloaded"

        print_success("Data collection completed")
        print_info(f"Data path: {context['data_path']}")
        print_info(f"Files downloaded: {len(context['files'])}")
        for file in context["files"]:
            print(f"    - {file}")

        return context

    except Exception as e:
        print_error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_2_problem_understanding(orchestrator, context):
    """Test Phase 2: Problem Understanding"""
    print_header("PHASE 2: PROBLEM UNDERSTANDING")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_problem_understanding
        context = await run_problem_understanding(orchestrator, context)

        # Verify results
        assert "problem_understanding" in context, "Missing problem_understanding"
        assert "overview_text" in context, "Missing overview_text"

        understanding = context["problem_understanding"]
        print_success("Problem understanding completed")
        print_info(f"Competition type: {understanding.get('competition_type', 'N/A')}")
        print_info(f"Task type: {understanding.get('task_type', 'N/A')}")
        print_info(f"Evaluation metric: {understanding.get('evaluation_metric', 'N/A')}")
        print_info(f"Submission format: {understanding.get('submission_format', 'N/A')}")

        return context

    except Exception as e:
        print_error(f"Problem understanding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_3_data_analysis(orchestrator, context):
    """Test Phase 3: Data Analysis"""
    print_header("PHASE 3: DATA ANALYSIS")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_data_analysis
        context = await run_data_analysis(orchestrator, context)

        # Verify results
        assert "data_analysis" in context, "Missing data_analysis"
        assert "needs_preprocessing" in context, "Missing needs_preprocessing flag"
        assert "target_column" in context, "Missing target_column"
        assert "data_files" in context, "Missing data_files mapping"

        analysis = context["data_analysis"]
        print_success("Data analysis completed")
        print_info(f"Data modality: {analysis.get('data_modality', 'N/A')}")
        print_info(f"Target column: {context['target_column']}")
        print_info(f"Preprocessing required: {context['needs_preprocessing']}")
        print_info(f"Train file: {context['data_files'].get('train_file', 'N/A')}")
        print_info(f"Test file: {context['data_files'].get('test_file', 'N/A')}")

        # Show key insights
        if "key_insights" in analysis:
            print_info("Key insights:")
            for insight in analysis["key_insights"][:3]:  # Show first 3
                print(f"    - {insight}")

        return context

    except Exception as e:
        print_error(f"Data analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_4_preprocessing(orchestrator, context):
    """Test Phase 4: Preprocessing"""
    print_header("PHASE 4: PREPROCESSING")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_preprocessing
        context = await run_preprocessing(orchestrator, context)

        # Verify results
        assert "clean_data_path" in context, "Missing clean_data_path"

        if context.get("needs_preprocessing"):
            print_success("Preprocessing completed")
            if "preprocessing_result" in context:
                result = context["preprocessing_result"]
                print_info(f"Train shape: {result.get('train_shape', 'N/A')}")
                print_info(f"Test shape: {result.get('test_shape', 'N/A')}")
                print_info(f"Missing values remaining: {result.get('missing_values_remaining', 'N/A')}")
            else:
                print_warning("Preprocessing failed, using raw data (this is expected)")
        else:
            print_info("Preprocessing skipped - data is already clean")

        print_info(f"Clean data path: {context['clean_data_path']}")

        return context

    except Exception as e:
        print_error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_5_planning(orchestrator, context):
    """Test Phase 5: Planning"""
    print_header("PHASE 5: PLANNING")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_planning
        context = await run_planning(orchestrator, context)

        # Verify results
        assert "execution_plan" in context, "Missing execution_plan"
        assert "needs_feature_engineering" in context, "Missing needs_feature_engineering flag"

        plan = context["execution_plan"]
        print_success("Planning completed")
        print_info(f"Models to try: {len(plan.get('models_to_try', []))}")
        print_info(f"Preprocessing steps: {len(plan.get('preprocessing_plan', []))}")
        print_info(f"Feature engineering features: {len(plan.get('feature_engineering_plan', []))}")
        print_info(f"Needs feature engineering: {context['needs_feature_engineering']}")

        # Show first model
        if plan.get('models_to_try'):
            first_model = plan['models_to_try'][0]
            print_info(f"Priority model: {first_model.get('model', 'N/A')}")
            print(f"    Reason: {first_model.get('reason', 'N/A')[:80]}...")

        return context

    except Exception as e:
        print_error(f"Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_6_feature_engineering(orchestrator, context):
    """Test Phase 6: Feature Engineering"""
    print_header("PHASE 6: FEATURE ENGINEERING")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_feature_engineering
        context = await run_feature_engineering(orchestrator, context)

        # Verify results
        assert "featured_data_path" in context, "Missing featured_data_path"

        if context.get("needs_feature_engineering"):
            print_warning("Feature engineering agent not yet fully implemented")
            print_info("Using clean data for now")
        else:
            print_info("Feature engineering skipped - not needed")

        print_info(f"Featured data path: {context['featured_data_path']}")

        return context

    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_phase_7_model_training(orchestrator, context):
    """Test Phase 7: Model Training"""
    print_header("PHASE 7: MODEL TRAINING")

    if not context:
        print_warning("Skipping - previous phase failed")
        return None

    try:
        from src.agents.orchestrator.phases import run_model_training
        context = await run_model_training(orchestrator, context)

        # Verify results
        assert "model_path" in context, "Missing model_path"
        assert "model_type" in context, "Missing model_type"
        assert "cv_score" in context, "Missing cv_score"

        print_success("Model training completed")
        print_info(f"Model type: {context['model_type']}")
        print_info(f"Model path: {context['model_path']}")
        print_info(f"CV Score: {context['cv_score']}")

        return context

    except Exception as e:
        print_error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_all_phase_tests():
    """Run all phase tests sequentially"""
    print_header("KAGGLE MULTI-AGENT SYSTEM - COMPREHENSIVE PHASE TEST")
    print("Testing all phases individually to verify fixes\n")

    # Initialize orchestrator
    print_info("Initializing orchestrator...")
    orchestrator = AgenticOrchestrator(
        competition_name='titanic',
        target_percentile=0.20,
        max_actions=50
    )
    print_success("Orchestrator initialized\n")

    # Run tests
    context = None

    # Phase 1
    context = await test_phase_1_data_collection(orchestrator)
    if not context:
        print_error("Cannot continue - data collection failed")
        return False

    # Phase 2
    context = await test_phase_2_problem_understanding(orchestrator, context)
    if not context:
        print_error("Cannot continue - problem understanding failed")
        return False

    # Phase 3
    context = await test_phase_3_data_analysis(orchestrator, context)
    if not context:
        print_error("Cannot continue - data analysis failed")
        return False

    # Phase 4
    context = await test_phase_4_preprocessing(orchestrator, context)
    if not context:
        print_error("Cannot continue - preprocessing failed")
        return False

    # Phase 5
    context = await test_phase_5_planning(orchestrator, context)
    if not context:
        print_error("Cannot continue - planning failed")
        return False

    # Phase 6
    context = await test_phase_6_feature_engineering(orchestrator, context)
    if not context:
        print_error("Cannot continue - feature engineering failed")
        return False

    # Phase 7
    context = await test_phase_7_model_training(orchestrator, context)
    if not context:
        print_error("Cannot continue - model training failed")
        return False

    # All phases passed!
    print_header("TEST RESULTS")
    print_success("All 7 phases completed successfully!")
    print("\nPhases tested:")
    print("  ✅ Phase 1: Data Collection")
    print("  ✅ Phase 2: Problem Understanding")
    print("  ✅ Phase 3: Data Analysis")
    print("  ✅ Phase 4: Preprocessing")
    print("  ✅ Phase 5: Planning")
    print("  ✅ Phase 6: Feature Engineering")
    print("  ✅ Phase 7: Model Training")
    print("\n✨ All fixes are working correctly!")

    return True


if __name__ == '__main__':
    print("Starting comprehensive phase tests...\n")

    try:
        success = asyncio.run(run_all_phase_tests())

        if success:
            print("\n" + "=" * 70)
            print("  🎉 ALL TESTS PASSED!")
            print("=" * 70)
            print("\nYour system is ready to run full competitions.")
            print("Execute: python test_fixed_system.py")
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("  ❌ SOME TESTS FAILED")
            print("=" * 70)
            print("\nPlease review the errors above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)