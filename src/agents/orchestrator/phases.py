"""
Orchestrator Phase Execution
Handles individual phases of the competition workflow following the 10-phase architecture.
"""
import importlib
import logging
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

    collection_context = {
        "competition_name": context["competition_name"],
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

    # Initialize Problem Understanding Agent
    #problem_agent = ProblemUnderstandingAgent()
    problem_agent = orchestrator.problem_understanding_agent

    # Understand the competition (with access to downloaded data from Phase 1)
    understanding, overview_text = await problem_agent.understand_competition(
        competition_name=context["competition_name"],
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

    # Initialize Data Analysis Agent
    data_agent = DataAnalysisAgent()

    # Analyze data with problem context
    EDA_Code = await data_agent.analyze_and_suggest(
        dataset = context.get("data_path"),
        competition_name=context["competition_name"],
        overview_text=context["overview_text"]
    )
    eda_file = Path(context["data_path"]) / "EDA.py"
    eda_file.write_text(EDA_Code)
    logger.info(f"💾 Saved preprocessing code to: {eda_file}")

    # Step 2: Execute the generated code
    logger.info("⚙️  Executing EDA code...")

    try:

        # Create a namespace for execution
        namespace = {}

        # Execute the code
        exec(EDA_Code, namespace)

    except Exception as e:
        logger.error(f"❌  execution failed: {e}")
        logger.warning("⚠️  Falling back to raw data")
        context["clean_data_path"] = context["data_path"]
        import traceback
        traceback.print_exc()

    return context

    # Add to context
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

    preprocessing_code = await preprocessing_agent.generate_preprocessing_code(
        data_analysis=context["data_analysis"],
        data_path=context["data_path"]
    )

    logger.info(f"✅ Generated {len(preprocessing_code)} chars of preprocessing code")

    # Save the generated code to file
    preprocessing_file = Path(context["data_path"]) / "preprocessing.py"
    preprocessing_file.write_text(preprocessing_code)
    logger.info(f"💾 Saved preprocessing code to: {preprocessing_file}")

    # Step 2: Execute the generated code
    logger.info("⚙️  Executing preprocessing code...")

    try:

        #Create a namespace for execution
        namespace = {}

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

        # Update context with clean data path
        clean_data_path = Path(context["data_path"]) / "clean_train.csv"
        if clean_data_path.exists():
            context["clean_data_path"] = str(Path(context["data_path"]))
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

    # TODO: Implement FeatureEngineeringAgent when ready
    # For now, use clean data
    logger.info("⚠️  FeatureEngineeringAgent not yet implemented")
    logger.info("   Using clean data for now")
    context["featured_data_path"] = context.get("clean_data_path", context["data_path"])

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

    # Find training file
    datasets = context.get("basic_stats", {}).get("datasets", {})
    train_file = None
    for filename in datasets.keys():
        if "train" in filename.lower():
            train_file = filename
            break

    if not train_file:
        train_file = list(datasets.keys())[0]

    # Use featured data if available, otherwise clean data, otherwise raw data
    data_dir = Path(context.get("featured_data_path",
                               context.get("clean_data_path",
                                          context["data_path"])))
    data_path = data_dir / train_file if data_dir.is_dir() else data_dir

    logger.info(f"📁 Using data: {data_path}")
    logger.info(f"🎯 Target: {context['target_column']}")
    logger.info(f"🤖 Following execution plan from AI")

    # Prepare training context with full accumulated context
    training_context = {
        "data_path": str(data_path),
        "target_column": context["target_column"],
        "execution_plan": execution_plan,  # ← PASS EXECUTION PLAN!
        "data_analysis": context["data_analysis"],
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

    # Find test file
    data_path = Path(context["data_path"])
    test_files = list(data_path.glob("*test*.csv"))

    if not test_files:
        logger.warning("Test file not found, looking for sample_submission...")
        test_files = list(data_path.glob("*sample*.csv"))

    if not test_files:
        raise FileNotFoundError("No test file found in data directory")

    test_file = test_files[0]
    logger.info(f"📄 Using test file: {test_file.name}")

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
        "submission_file": results.get("submission_file"),
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
