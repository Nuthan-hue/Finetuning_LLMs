"""
Orchestrator Phase Execution
Handles individual phases of the competition workflow following the 10-phase architecture.
"""
import importlib
import logging
import json
from typing import Any, Dict
from pathlib import Path
import sys

from ..base import AgentState
from ..llm_agents import ProblemUnderstandingAgent, DataAnalysisAgent, PlanningAgent, PreprocessingAgent

logger = logging.getLogger(__name__)


async def run_data_collection(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 1: DATA COLLECTION (Worker - No LLM)
    Downloads competition files and performs basic analysis.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with data collection results
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("=" * 70)

    competition_name = context["competition_name"]

    logger.info(f"📥 Downloading data for competition: {competition_name}")

    collection_context = {
        "competition_name": competition_name,
        "analyze": False
    }

    results = await orchestrator.data_collector.run(collection_context)

    # Check for errors
    if orchestrator.data_collector.state == AgentState.ERROR:
        raise RuntimeError(f"Data collection failed: {orchestrator.data_collector.error}")

    # Add to context
    context.update({
        "data_path": results["data_path"],
        "files": results.get("files", []),
    })

    logger.info(f"✅ Data collected: {len(results.get('files', []))} files")
    logger.info(f"   Data path: {results['data_path']}")

    return context


async def run_problem_understanding(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 2: PROBLEM UNDERSTANDING (LLM Agent)
    AI reads competition description AND verifies against downloaded data.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict (now includes data)

    Returns:
        Updated context with problem understanding
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PROBLEM UNDERSTANDING")
    logger.info("=" * 70)

    competition_name = context["competition_name"]

    # Initialize Problem Understanding Agent
    problem_agent = orchestrator.problem_understanding_agent

    # Understand the competition (with access to downloaded data from Phase 1)
    understanding, overview_text = await problem_agent.understand_competition(
        competition_name=competition_name,
        data_path=context.get("data_path")  # From Phase 1
    )

    # Display summary
    summary = problem_agent.get_problem_summary(understanding)
    print(summary)

    # Add to context
    context["problem_understanding"] = understanding
    context["overview_text"] = overview_text

    logger.info("✅ Problem understanding completed")
    logger.info(f"   Task: {understanding.get('competition_type', 'N/A')}")
    logger.info(f"   Metric: {understanding.get('evaluation_metric', 'N/A')}")

    return context

async def run_data_analysis(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 3: DATA ANALYSIS (LLM Agent)
    Deep data analysis with modality detection and preprocessing recommendations.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict (includes problem_understanding, data)

    Returns:
        Updated context with AI data analysis
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: DATA ANALYSIS")
    logger.info("=" * 70)

    competition_name = context["competition_name"]

    # Initialize Data Analysis Agent
    data_agent = DataAnalysisAgent()

    # Analyze data with problem context and get JSON recommendations
    data_analysis = await data_agent.analyze_and_suggest(
        data_path=context["data_path"],
        competition_name=competition_name,
        problem_understanding=context["problem_understanding"]
    )

    # Log key findings
    logger.info(f"✅ Data modality: {data_analysis.get('data_modality')}")
    logger.info(f"✅ Target column: {data_analysis.get('target_column')}")
    logger.info(f"✅ Preprocessing required: {data_analysis.get('preprocessing_required')}")

    # Add to context
    context["data_analysis"] = data_analysis
    context["needs_preprocessing"] = data_analysis.get("preprocessing_required", False)

    # Extract target column for downstream phases
    context["target_column"] = data_analysis.get("target_column")

    # Extract file mapping (NO HARDCODED NAMES!)
    context["data_files"] = data_analysis.get("data_files", {})
    logger.info(f"📁 File mapping: train={context['data_files'].get('train_file')}, test={context['data_files'].get('test_file')}")

    # Log key insights
    if "key_insights" in data_analysis:
        logger.info("\n📊 Key Insights:")
        for insight in data_analysis["key_insights"]:
            logger.info(f"  • {insight}")

    return context


async def run_preprocessing(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 4: PREPROCESSING (Conditional - LLM + Worker)
    Generates and executes preprocessing code if needed.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with clean_data_path (or skipped if not needed)
    """
    if not context.get("needs_preprocessing", False):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: PREPROCESSING - SKIPPED")
        logger.info("=" * 70)
        logger.info("⏭️  Data is clean, no preprocessing needed")
        context["clean_data_path"] = context["data_path"]
        return context

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: PREPROCESSING")
    logger.info("=" * 70)

    # Step 1: Generate preprocessing code with AI
    logger.info("🤖 Generating preprocessing code with AI...")
    preprocessing_agent = PreprocessingAgent()

    try:
        preprocessing_code = await preprocessing_agent.generate_preprocessing_code(
            data_analysis=context["data_analysis"],
            data_path=context["data_path"]
        )
    except Exception as e:
        logger.error(f"❌ Preprocessing code generation failed: {e}")
        logger.warning("⚠️  Skipping preprocessing, using raw data")
        context["clean_data_path"] = context["data_path"]
        return context

    logger.info(f"✅ Generated {len(preprocessing_code)} chars of preprocessing code")

    # Save preprocessing code to raw folder
    data_path = Path(context["data_path"])
    preprocessing_file = data_path / "preprocessing.py"
    preprocessing_file.write_text(preprocessing_code)
    logger.info(f"💾 Saved preprocessing code to: {preprocessing_file}")

    # Step 2: Execute the generated code
    logger.info("⚙️  Executing preprocessing code...")

    try:
        # Create a namespace for execution with necessary variables
        namespace = {
            "data_path": context["data_path"],
            "data_files": context.get("data_files", {}),
            "train_file": context.get("data_files", {}).get("train_file"),
            "test_file": context.get("data_files", {}).get("test_file"),
            "target_column": context.get("target_column")
        }

        # Execute the code
        exec(preprocessing_code, namespace)

        # Call the preprocess_data function
        if "preprocess_data" not in namespace:
            raise RuntimeError("Generated code missing 'preprocess_data' function")

        preprocess_func = namespace["preprocess_data"]
        result = preprocess_func(context["data_path"])

        logger.info(f"✅ Preprocessing completed")
        logger.info(f"   Train shape: {result.get('train_shape')}")
        logger.info(f"   Test shape: {result.get('test_shape')}")
        logger.info(f"   Columns: {len(result.get('columns', []))}")
        logger.info(f"   Missing values remaining: {result.get('missing_values_remaining', 0)}")

        # Update context - clean files saved in same raw/ folder
        clean_train_path = data_path / "clean_train.csv"
        if clean_train_path.exists():
            context["clean_data_path"] = context["data_path"]
            context["preprocessing_result"] = result
        else:
            raise RuntimeError("Preprocessing did not create clean_train.csv")

    except Exception as e:
        logger.error(f"❌ Preprocessing execution failed: {e}")
        logger.warning("⚠️  Falling back to raw data")
        context["clean_data_path"] = context["data_path"]
        import traceback
        traceback.print_exc()

    return context


async def run_planning(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 5: PLANNING (LLM Agent)
    AI creates comprehensive execution plan with model selection and strategies.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with execution_plan
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: PLANNING")
    logger.info("=" * 70)

    # Initialize Planning Agent
    planning_agent = PlanningAgent()

    # Create comprehensive plan
    execution_plan = await planning_agent.create_plan(
        problem_understanding=context["problem_understanding"],
        data_analysis=context["data_analysis"],
        competition_name=context["competition_name"]
    )

    # Add to context
    context["execution_plan"] = execution_plan

    # Extract key flags
    context["needs_feature_engineering"] = execution_plan.get("feature_engineering_required", False)

    # Display plan summary
    summary = planning_agent.get_plan_summary(execution_plan)
    print(summary)

    logger.info("✅ Planning completed")
    logger.info(f"   Strategy: {execution_plan.get('strategy_summary', 'N/A')}")
    logger.info(f"   Models: {len(execution_plan.get('models_to_try', []))}")
    logger.info(f"   Needs feature engineering: {context['needs_feature_engineering']}")

    return context


async def run_feature_engineering(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 6: FEATURE ENGINEERING (Conditional - LLM + Worker)
    Generates and executes feature engineering code if needed.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with featured_data_path (or skipped if not needed)
    """
    if not context.get("needs_feature_engineering", False):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 6: FEATURE ENGINEERING - SKIPPED")
        logger.info("=" * 70)
        logger.info("⏭️  No feature engineering needed")
        context["featured_data_path"] = context.get("clean_data_path", context["data_path"])
        return context

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: FEATURE ENGINEERING")
    logger.info("=" * 70)

    # Import FeatureEngineeringAgent
    from ..llm_agents.feature_engineering_agent import FeatureEngineeringAgent

    # Step 1: Generate feature engineering code with AI
    logger.info("🤖 Generating feature engineering code with AI...")
    feature_agent = FeatureEngineeringAgent()

    # Get feature engineering plan from execution plan
    execution_plan = context.get("execution_plan", {})
    feature_plan = execution_plan.get("feature_engineering_plan", [])

    if not feature_plan:
        logger.warning("⚠️  No feature engineering plan found in execution plan")
        logger.info("   Using clean data")
        context["featured_data_path"] = context.get("clean_data_path", context["data_path"])
        return context

    feature_code = await feature_agent.generate_feature_engineering_code(
        feature_engineering_plan=feature_plan,
        data_analysis=context["data_analysis"],
        clean_data_path=context.get("clean_data_path", context["data_path"])
    )

    logger.info(f"✅ Generated {len(feature_code)} chars of feature engineering code")

    # Setup featured data directory: data/{competition}/featured/
    competition_name = context["competition_name"]
    data_base = Path("data") / competition_name
    featured_dir = data_base / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = data_base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save the generated code to metadata folder
    feature_file = metadata_dir / "feature_engineering.py"
    feature_file.write_text(feature_code)
    logger.info(f"💾 Saved feature engineering code to: {feature_file}")

    # Step 2: Execute the generated code
    logger.info("⚙️  Executing feature engineering code...")

    try:
        # Create a namespace for execution
        namespace = {}

        # Execute the code
        exec(feature_code, namespace)

        # Call the engineer_features function
        if "engineer_features" not in namespace:
            raise RuntimeError("Generated code missing 'engineer_features' function")

        engineer_func = namespace["engineer_features"]
        clean_data_path = context.get("clean_data_path", context["data_path"])
        result = engineer_func(clean_data_path, str(featured_dir))

        logger.info(f"✅ Feature engineering completed")
        logger.info(f"   Train shape: {result.get('train_shape')}")
        logger.info(f"   Test shape: {result.get('test_shape')}")
        logger.info(f"   Original features: {result.get('original_feature_count')}")
        logger.info(f"   New features: {result.get('new_feature_count')}")
        logger.info(f"   Features added: {result.get('features_added', 0)}")

        # Update context with featured data path
        featured_train = featured_dir / "featured_train.csv"
        if featured_train.exists():
            context["featured_data_path"] = str(featured_dir)
            context["feature_engineering_result"] = result
        else:
            raise RuntimeError("Feature engineering did not create featured_train.csv")

    except Exception as e:
        logger.error(f"❌ Feature engineering execution failed: {e}")
        logger.warning("⚠️  Falling back to clean data")
        context["featured_data_path"] = context.get("clean_data_path", context["data_path"])
        import traceback
        traceback.print_exc()

    return context


async def run_model_training(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 7: MODEL TRAINING (Worker - No LLM)
    Executes training based on execution plan from PlanningAgent.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict (includes execution_plan, data paths)

    Returns:
        Updated context with training results
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: MODEL TRAINING")
    logger.info("=" * 70)

    # Get execution plan (created by PlanningAgent)
    execution_plan = context.get("execution_plan")
    if not execution_plan:
        raise RuntimeError(
            "❌ No execution plan from PlanningAgent! "
            "This is a pure agentic system - requires AI decisions. "
            "Check GEMINI_API_KEY."
        )

    # Get training file from AI-identified file mapping (NO HARDCODED NAMES!)
    data_files = context.get("data_files", {})
    train_file = data_files.get("train_file")

    if not train_file:
        raise RuntimeError(
            "❌ No training file identified by DataAnalysisAgent! "
            "File mapping is missing from context."
        )

    # Determine which data directory to use (priority: featured > clean > raw)
    data_dir = Path(context.get("featured_data_path",
                               context.get("clean_data_path",
                                          context["data_path"])))

    # Construct full path to training file
    # Priority: featured > clean > raw
    if context.get("featured_data_path"):
        # Try featured version first
        featured_train = data_dir / "featured_train.csv"
        if featured_train.exists():
            data_path = featured_train
            logger.info(f"📁 Using featured training file: featured_train.csv")
        elif context.get("clean_data_path"):
            # Fallback to clean
            clean_train_file = f"clean_{train_file}"
            clean_path = data_dir / clean_train_file
            if clean_path.exists():
                data_path = clean_path
                logger.info(f"📁 Using cleaned training file: {clean_train_file}")
            else:
                data_path = data_dir / train_file
                logger.info(f"📁 Using original training file: {train_file}")
        else:
            data_path = data_dir / train_file
            logger.info(f"📁 Using original training file: {train_file}")
    elif context.get("clean_data_path"):
        # Try clean version
        clean_train_file = f"clean_{train_file}"
        clean_path = data_dir / clean_train_file
        if clean_path.exists():
            data_path = clean_path
            logger.info(f"📁 Using cleaned training file: {clean_train_file}")
        else:
            # Fallback to original
            data_path = data_dir / train_file
            logger.info(f"📁 Using original training file: {train_file}")
    else:
        data_path = data_dir / train_file
        logger.info(f"📁 Using training file: {train_file}")

    logger.info(f"📂 Data directory: {data_dir}")
    logger.info(f"🎯 Target column: {context['target_column']}")
    logger.info(f"🤖 Following AI execution plan")

    # Prepare training context with full accumulated context
    training_context = {
        "data_path": str(data_path),
        "target_column": context["target_column"],
        "execution_plan": execution_plan,  # ← PASS EXECUTION PLAN!
        "data_analysis": context["data_analysis"],
        "ai_analysis": context["data_analysis"],  # Model trainer expects this key
        "competition_name": context["competition_name"],
    }

    # Execute training
    results = await orchestrator.model_trainer.run(training_context)

    if orchestrator.model_trainer.state == AgentState.ERROR:
        raise RuntimeError(f"Model training failed: {orchestrator.model_trainer.error}")

    # Add to context
    context.update({
        "model_path": results["model_path"],
        "model_type": results["model_type"],
        "cv_score": results.get("best_score"),
        "training_results": results
    })

    logger.info(f"✅ Training completed")
    logger.info(f"   Model: {results['model_type']}")
    logger.info(f"   CV Score: {results.get('best_score', 'N/A')}")

    return context


async def run_submission(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 8: SUBMISSION (Worker - No LLM)
    Generates predictions and submits to Kaggle.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with submission results
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8: SUBMISSION")
    logger.info("=" * 70)

    # Get test file from AI-identified file mapping (NO HARDCODED NAMES!)
    data_files = context.get("data_files", {})
    test_file_name = data_files.get("test_file")

    if not test_file_name:
        logger.warning("⚠️  No test file identified by AI. Checking if we need to generate predictions from train...")
        # Some competitions don't have explicit test files
        # They generate future predictions or use train for cross-validation
        raise FileNotFoundError(
            "No test file identified. This competition may not have a separate test set. "
            "Consider implementing prediction generation from training data."
        )

    # Determine which data directory to use
    data_dir = Path(context.get("clean_data_path", context["data_path"]))

    # Construct full path to test file
    # If preprocessing was done, look for clean_ prefixed version
    if context.get("clean_data_path"):
        # Try clean version first
        clean_test_file = f"clean_{test_file_name}"
        clean_test_path = data_dir / clean_test_file
        if clean_test_path.exists():
            test_file = clean_test_path
            logger.info(f"📁 Using cleaned test file: {clean_test_file}")
        else:
            # Fallback to original
            test_file = data_dir / test_file_name
            logger.info(f"📁 Using original test file: {test_file_name}")
    else:
        test_file = data_dir / test_file_name
        logger.info(f"📁 Using test file: {test_file_name}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found at {test_file}")

    logger.info(f"📂 Test data directory: {data_dir}")

    # Prepare submission context
    submission_context = {
        "model_path": context["model_path"],
        "test_data_path": str(test_file),
        "competition_name": context["competition_name"],
        "model_type": context["model_type"],
        "submission_message": f"Iteration {orchestrator.iteration} - Automated submission",
        "auto_submit": True,
        "interactive": True,
    }

    # Add NLP-specific context if needed
    if context.get("model_type") == "transformer":
        submission_context["text_column"] = context.get("text_column")

    # Execute submission
    results = await orchestrator.submission_agent.run(submission_context)

    if orchestrator.submission_agent.state == AgentState.ERROR:
        raise RuntimeError(f"Submission failed: {orchestrator.submission_agent.error}")

    # Add to context
    context.update({
        "submission_file": results.get("submission_path"),  # Submitter returns "submission_path"
        "submission_id": results.get("submission_id"),
        "leaderboard_score": results.get("leaderboard_score")
    })

    logger.info(f"✅ Submission completed")
    logger.info(f"   Score: {results.get('leaderboard_score', 'Pending')}")

    return context


async def run_evaluation(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 9: EVALUATION (LLM Agent)
    Diagnoses model performance and decides if improvement is needed.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with evaluation results and needs_improvement flag
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9: EVALUATION")
    logger.info("=" * 70)

    # Wait for leaderboard to update
    import asyncio
    logger.info("⏳ Waiting for leaderboard to update...")
    await asyncio.sleep(60)

    # Check leaderboard
    leaderboard_context = {
        "competition_name": context["competition_name"],
        "target_percentile": orchestrator.target_percentile
    }

    leaderboard_results = await orchestrator.leaderboard_monitor.run(leaderboard_context)

    if orchestrator.leaderboard_monitor.state == AgentState.ERROR:
        logger.warning(f"Leaderboard check failed: {orchestrator.leaderboard_monitor.error}")
        # Continue anyway
        leaderboard_results = {
            "current_rank": "Unknown",
            "current_percentile": 1.0,
            "meets_target": False
        }

    current_rank = leaderboard_results.get("current_rank", "N/A")
    current_percentile = leaderboard_results.get("current_percentile", 1.0)
    meets_target = leaderboard_results.get("meets_target", False)

    # Add to context
    context.update({
        "current_rank": current_rank,
        "current_percentile": current_percentile,
        "meets_target": meets_target,
        "leaderboard_results": leaderboard_results
    })

    logger.info(f"✅ Evaluation completed")
    logger.info(f"   Current rank: {current_rank}")
    logger.info(f"   Current percentile: {current_percentile * 100:.1f}%")
    logger.info(f"   Target percentile: {orchestrator.target_percentile * 100:.1f}%")
    logger.info(f"   Meets target: {meets_target}")

    # Decide if improvement needed
    context["needs_improvement"] = not meets_target

    return context


async def run_optimization(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 10: OPTIMIZATION (Conditional - LLM Agent)
    AI decides next strategy and where to loop back.

    Args:
        orchestrator: Orchestrator instance
        context: Accumulated context dict

    Returns:
        Updated context with optimization strategy (or None if done)
    """
    if not context.get("needs_improvement"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 10: OPTIMIZATION - SKIPPED")
        logger.info("=" * 70)
        logger.info("🎉 Target achieved! No optimization needed.")
        context["optimization_strategy"] = None
        return context

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10: OPTIMIZATION")
    logger.info("=" * 70)

    from ..llm_agents import StrategyAgent

    strategy_agent = StrategyAgent()

    # Get AI strategy for next iteration
    strategy = await strategy_agent.select_optimization_strategy(
        recommendation=context.get("leaderboard_results", {}).get("recommendation", "improve"),
        current_model=context.get("model_type", "lightgbm"),
        tried_models=orchestrator.tried_models,
        current_percentile=context.get("current_percentile", 1.0),
        target_percentile=orchestrator.target_percentile,
        iteration=orchestrator.iteration,
        competition_type=context.get("data_modality", "tabular"),
        performance_history=orchestrator.workflow_history
    )

    # Add to context
    context["optimization_strategy"] = strategy

    logger.info(f"✅ Optimization strategy created")
    logger.info(f"   AI Strategy: {strategy.get('action', 'N/A')}")
    logger.info(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")
    logger.info(f"   Loop back to: {strategy.get('loop_back_to', 'planning')}")

    return context
