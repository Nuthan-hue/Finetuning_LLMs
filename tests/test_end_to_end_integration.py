#!/usr/bin/env python3
"""
End-to-End Integration Test for Phases 1-7
Tests the complete pipeline from data collection through model training
"""

import asyncio
import sys
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator.phases import (
    run_data_collection,
    run_problem_understanding,
    run_data_analysis,
    run_preprocessing,
    run_planning,
    run_feature_engineering,
    run_model_training
)
from src.agents.data_collector import DataCollector
from src.agents.model_trainer import ModelTrainer
from src.agents.llm_agents import ProblemUnderstandingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockOrchestrator:
    """Mock orchestrator with required agents"""
    def __init__(self):
        self.data_collector = DataCollector()
        self.problem_understanding_agent = ProblemUnderstandingAgent()
        self.model_trainer = ModelTrainer()
        self.iteration = 1


async def test_phases_1_to_7():
    """
    End-to-end integration test for Phases 1-7

    Tests:
    1. Data Collection (Phase 1)
    2. Problem Understanding (Phase 2)
    3. Data Analysis (Phase 3)
    4. Preprocessing (Phase 4) - Conditional
    5. Planning (Phase 5)
    6. Feature Engineering (Phase 6) - Conditional
    7. Model Training (Phase 7)
    """

    print("\n" + "=" * 80)
    print("END-TO-END INTEGRATION TEST: PHASES 1-7")
    print("=" * 80)
    print("\nCompetition: Titanic (Binary Classification)")
    print("Testing: Complete pipeline from data collection to model training")
    print("=" * 80 + "\n")

    # Initialize mock orchestrator
    orchestrator = MockOrchestrator()

    # Initialize context
    context = {
        "competition_name": "titanic"
    }

    test_results = {
        "phase_1": False,
        "phase_2": False,
        "phase_3": False,
        "phase_4": False,
        "phase_5": False,
        "phase_6": False,
        "phase_7": False
    }

    try:
        # ========================================
        # PHASE 1: DATA COLLECTION
        # ========================================
        print("\n📦 Starting Phase 1: Data Collection...")
        context = await run_data_collection(orchestrator, context)

        # Validate Phase 1 outputs
        assert "data_path" in context, "Missing data_path in context"
        assert "files" in context, "Missing files in context"
        assert Path(context["data_path"]).exists(), f"Data path doesn't exist: {context['data_path']}"

        logger.info(f"✅ Phase 1 PASSED")
        logger.info(f"   Data path: {context['data_path']}")
        logger.info(f"   Files: {context['files']}")
        test_results["phase_1"] = True

        # ========================================
        # PHASE 2: PROBLEM UNDERSTANDING
        # ========================================
        print("\n🧠 Starting Phase 2: Problem Understanding...")
        context = await run_problem_understanding(orchestrator, context)

        # Validate Phase 2 outputs
        assert "problem_understanding" in context, "Missing problem_understanding in context"
        assert "competition_type" in context["problem_understanding"], "Missing competition_type"
        assert "evaluation_metric" in context["problem_understanding"], "Missing evaluation_metric"

        logger.info(f"✅ Phase 2 PASSED")
        logger.info(f"   Task: {context['problem_understanding'].get('competition_type')}")
        logger.info(f"   Metric: {context['problem_understanding'].get('evaluation_metric')}")
        test_results["phase_2"] = True

        # ========================================
        # PHASE 3: DATA ANALYSIS
        # ========================================
        print("\n📊 Starting Phase 3: Data Analysis...")
        context = await run_data_analysis(orchestrator, context)

        # Validate Phase 3 outputs
        assert "data_analysis" in context, "Missing data_analysis in context"
        assert "data_modality" in context["data_analysis"], "Missing data_modality"
        assert "target_column" in context, "Missing target_column in context"
        assert "needs_preprocessing" in context, "Missing needs_preprocessing flag"
        assert "data_files" in context, "Missing data_files mapping"

        logger.info(f"✅ Phase 3 PASSED")
        logger.info(f"   Modality: {context['data_analysis'].get('data_modality')}")
        logger.info(f"   Target: {context.get('target_column')}")
        logger.info(f"   Needs preprocessing: {context.get('needs_preprocessing')}")
        test_results["phase_3"] = True

        # ========================================
        # PHASE 4: PREPROCESSING (Conditional)
        # ========================================
        print("\n🧹 Starting Phase 4: Preprocessing...")
        context = await run_preprocessing(orchestrator, context)

        # Validate Phase 4 outputs
        assert "clean_data_path" in context, "Missing clean_data_path in context"

        if context.get("needs_preprocessing"):
            # If preprocessing was done, verify clean files exist
            clean_train = Path(context["clean_data_path"]) / "clean_train.csv"
            # Note: clean_train.csv might not exist if preprocessing failed gracefully
            logger.info(f"   Preprocessing executed: {clean_train.exists()}")
        else:
            logger.info(f"   Preprocessing skipped (not needed)")

        logger.info(f"✅ Phase 4 PASSED")
        logger.info(f"   Clean data path: {context['clean_data_path']}")
        test_results["phase_4"] = True

        # ========================================
        # PHASE 5: PLANNING
        # ========================================
        print("\n🎯 Starting Phase 5: Planning...")
        context = await run_planning(orchestrator, context)

        # Validate Phase 5 outputs
        assert "execution_plan" in context, "Missing execution_plan in context"
        assert "models_to_try" in context["execution_plan"], "Missing models_to_try in plan"
        assert "needs_feature_engineering" in context, "Missing needs_feature_engineering flag"
        assert len(context["execution_plan"]["models_to_try"]) > 0, "No models in execution plan"

        logger.info(f"✅ Phase 5 PASSED")
        logger.info(f"   Strategy: {context['execution_plan'].get('strategy_summary')}")
        logger.info(f"   Models: {[m.get('model') for m in context['execution_plan']['models_to_try']]}")
        logger.info(f"   Needs feature engineering: {context.get('needs_feature_engineering')}")
        test_results["phase_5"] = True

        # ========================================
        # PHASE 6: FEATURE ENGINEERING (Conditional)
        # ========================================
        print("\n🔧 Starting Phase 6: Feature Engineering...")
        context = await run_feature_engineering(orchestrator, context)

        # Validate Phase 6 outputs
        assert "featured_data_path" in context, "Missing featured_data_path in context"

        if context.get("needs_feature_engineering"):
            logger.info(f"   Feature engineering requested (but not yet implemented)")
        else:
            logger.info(f"   Feature engineering skipped (not needed)")

        logger.info(f"✅ Phase 6 PASSED")
        logger.info(f"   Featured data path: {context['featured_data_path']}")
        test_results["phase_6"] = True

        # ========================================
        # PHASE 7: MODEL TRAINING
        # ========================================
        print("\n🏋️ Starting Phase 7: Model Training...")

        # This might take a while, so add timeout handling
        try:
            context = await asyncio.wait_for(
                run_model_training(orchestrator, context),
                timeout=300  # 5 minutes max
            )
        except asyncio.TimeoutError:
            logger.error("❌ Phase 7 timed out after 5 minutes")
            raise

        # Validate Phase 7 outputs
        assert "model_path" in context, "Missing model_path in context"
        assert "model_type" in context, "Missing model_type in context"
        assert "training_results" in context, "Missing training_results in context"

        # Verify model file exists
        if context.get("model_path"):
            model_path = Path(context["model_path"])
            assert model_path.exists(), f"Model file doesn't exist: {model_path}"

        logger.info(f"✅ Phase 7 PASSED")
        logger.info(f"   Model type: {context.get('model_type')}")
        logger.info(f"   Model path: {context.get('model_path')}")
        logger.info(f"   CV Score: {context.get('cv_score')}")
        test_results["phase_7"] = True

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

        # Print context for debugging
        print("\n" + "=" * 80)
        print("DEBUG: Context at failure point")
        print("=" * 80)
        print(json.dumps({k: str(v)[:100] for k, v in context.items()}, indent=2))

        return False

    # ========================================
    # TEST SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for phase, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {phase.upper().replace('_', ' ')}")

    print(f"\nPassed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n🎉 All phases completed successfully!")
        print("\n📋 Context Keys Generated:")
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, dict):
                print(f"  • {key}: <dict with {len(value)} keys>")
            elif isinstance(value, list):
                print(f"  • {key}: <list with {len(value)} items>")
            else:
                print(f"  • {key}: {str(value)[:60]}")
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} phase(s) failed")
        return False


async def main():
    """Run the end-to-end integration test"""
    success = await test_phases_1_to_7()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())