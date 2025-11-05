"""
Orchestrator Phase Execution
Handles individual phases of the competition workflow.
"""
import logging
from typing import Any, Dict
from pathlib import Path

from ..base import AgentState
from ..llm_agents import ProblemUnderstandingAgent, DataAnalysisAgent, PlanningAgent

logger = logging.getLogger(__name__)


async def run_problem_understanding(
    orchestrator,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute problem understanding phase.

    This is the FIRST phase - AI reads and understands the competition problem
    BEFORE looking at any data.

    Args:
        orchestrator: Orchestrator instance
        context: Execution context

    Returns:
        Dictionary containing problem understanding
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: PROBLEM UNDERSTANDING")
    logger.info("=" * 70)

    # Initialize Problem Understanding Agent
    problem_agent = ProblemUnderstandingAgent()

    # Understand the competition
    understanding = await problem_agent.understand_competition(
        orchestrator.competition_name
    )

    # Display summary
    summary = problem_agent.get_problem_summary(understanding)
    print(summary)
    logger.info("Problem understanding completed")

    return {
        "problem_understanding": understanding,
        "competition_name": orchestrator.competition_name
    }


async def run_data_collection(orchestrator, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data collection phase."""
    logger.info("Initiating data collection...")

    collection_context = {
        "competition_name": orchestrator.competition_name,
        "external_sources": context.get("external_sources", []),
        "analyze": True
    }

    results = await orchestrator.data_collector.run(collection_context)

    # Check for errors
    if orchestrator.data_collector.state == AgentState.ERROR:
        raise RuntimeError(f"Data collection failed: {orchestrator.data_collector.error}")

    return results


async def run_data_analysis(
    orchestrator,
    data_results: Dict[str, Any],
    problem_understanding: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute data analysis phase with problem context.

    Args:
        orchestrator: Orchestrator instance
        data_results: Results from data collection
        problem_understanding: Results from problem understanding

    Returns:
        Dictionary containing AI data analysis
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: AI DATA ANALYSIS")
    logger.info("=" * 70)

    # Initialize Data Analysis Agent
    data_agent = DataAnalysisAgent()

    # Analyze data with problem context
    ai_analysis = await data_agent.analyze_and_suggest(
        dataset_info=data_results.get("analysis_report", {}),
        competition_name=orchestrator.competition_name
    )

    logger.info(f"✅ AI Analysis completed")
    logger.info(f"   Target: {ai_analysis.get('target_column')}")
    logger.info(f"   Task: {ai_analysis.get('task_type')}")
    logger.info(f"   Features suggested: {len(ai_analysis.get('feature_engineering', []))}")

    return {
        "ai_analysis": ai_analysis,
        "competition_name": orchestrator.competition_name
    }


async def run_planning(
    orchestrator,
    problem_understanding: Dict[str, Any],
    data_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute planning phase - AI creates comprehensive execution plan.

    Args:
        orchestrator: Orchestrator instance
        problem_understanding: Results from problem understanding
        data_analysis: Results from data analysis

    Returns:
        Dictionary containing execution plan
    """
    logger.info("=" * 70)
    logger.info("PHASE 4: AI PLANNING")
    logger.info("=" * 70)

    # Initialize Planning Agent
    planning_agent = PlanningAgent()

    # Create comprehensive plan
    execution_plan = await planning_agent.create_plan(
        problem_understanding=problem_understanding,
        data_analysis=data_analysis,
        competition_name=orchestrator.competition_name
    )

    # Display plan summary
    summary = planning_agent.get_plan_summary(execution_plan)
    print(summary)
    logger.info("Planning completed")

    return {
        "execution_plan": execution_plan,
        "competition_name": orchestrator.competition_name
    }


async def run_initial_training(
    orchestrator,
    data_results: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute initial model training."""
    logger.info("Starting initial model training...")

    # Determine target column from analysis
    analysis = data_results.get("analysis_report", {})
    datasets = analysis.get("datasets", {})

    target_column = None
    train_file = None
    ai_analysis = None  # Initialize to avoid reference errors

    # Find training file and target column
    for filename, dataset_info in datasets.items():
        if "train" in filename.lower():
            train_file = filename
            target_column = dataset_info.get("target_column")
            break

    if not train_file:
        # Use first dataset
        train_file = list(datasets.keys())[0]

    # ALWAYS use AI agent for complete analysis
    if not target_column or not ai_analysis:
        logger.info("🤖 Using AI Agent to identify target column...")

        from ..llm_agents import DataAnalysisAgent

        data_agent = DataAnalysisAgent()

        # Ask AI to analyze the dataset
        ai_analysis = await data_agent.analyze_and_suggest(
            dataset_info=data_results.get("analysis_report", {}),
            competition_name=orchestrator.competition_name
        )

        target_column = ai_analysis.get("target_column")
        confidence = ai_analysis.get("target_confidence", "unknown")

        if not target_column:
            raise RuntimeError(
                "❌ AI Agent failed to identify target column. "
                "This is a pure agentic AI system - no hardcoded fallbacks! "
                "Make sure GEMINI_API_KEY is set in your environment."
            )

        logger.info(f"🤖 AI identified target: {target_column} (confidence: {confidence})")

        # Log AI suggestions
        if "preprocessing" in ai_analysis:
            logger.info(f"📋 AI preprocessing suggestions: {ai_analysis['preprocessing']}")
        if "feature_engineering" in ai_analysis:
            logger.info(f"💡 AI feature ideas: {ai_analysis['feature_engineering'][:3]}")

    logger.info(f"Using training file: {train_file}")
    logger.info(f"Target column: {target_column}")

    # Prepare training context WITH AI ANALYSIS
    data_path = Path(data_results["data_path"]) / train_file

    training_context = {
        "data_path": str(data_path),
        "target_column": target_column,
        "config": training_config,
        "ai_analysis": ai_analysis,  # 🤖 Pass AI recommendations to trainer!
        "competition_name": orchestrator.competition_name
    }

    results = await orchestrator.model_trainer.run(training_context)

    if orchestrator.model_trainer.state == AgentState.ERROR:
        raise RuntimeError(f"Model training failed: {orchestrator.model_trainer.error}")

    return results


async def run_submission(
    orchestrator,
    training_results: Dict[str, Any],
    data_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute submission phase."""
    logger.info("Preparing and submitting predictions...")

    # Find test file
    data_path = Path(data_results["data_path"])
    test_files = list(data_path.glob("*test*.csv"))

    if not test_files:
        logger.warning("Test file not found, looking for sample_submission...")
        test_files = list(data_path.glob("*sample*.csv"))

    if not test_files:
        raise FileNotFoundError("No test file found in data directory")

    test_file = test_files[0]
    logger.info(f"Using test file: {test_file.name}")

    # Prepare submission context
    submission_context = {
        "model_path": training_results["model_path"],
        "test_data_path": str(test_file),
        "competition_name": orchestrator.competition_name,
        "model_type": training_results["model_type"],
        "submission_message": f"Iteration {orchestrator.iteration} - Automated submission",
        "auto_submit": True,
        "interactive": True,  # Ask for confirmation before submitting
        # Format spec is auto-detected by SubmissionAgent
    }

    # Add NLP-specific context if needed
    if training_results["model_type"] == "transformer":
        submission_context["text_column"] = training_results.get("text_column")

    results = await orchestrator.submission_agent.run(submission_context)

    if orchestrator.submission_agent.state == AgentState.ERROR:
        raise RuntimeError(f"Submission failed: {orchestrator.submission_agent.error}")

    return results


def log_phase_results(phase_name: str, results: Dict[str, Any]) -> None:
    """Log phase completion and key results."""
    logger.info(f"✓ {phase_name} completed")

    # Log key metrics
    for key, value in results.items():
        if key not in ["analysis_report", "leaderboard_data"]:
            logger.info(f"  {key}: {value}")
