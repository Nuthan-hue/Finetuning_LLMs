"""
Orchestrator Phase Execution
Handles individual phases of the competition workflow.
"""
import logging
from typing import Any, Dict
from pathlib import Path

from ..base import AgentState

logger = logging.getLogger(__name__)


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

    # Find training file and target column
    for filename, dataset_info in datasets.items():
        if "train" in filename.lower():
            train_file = filename
            target_column = dataset_info.get("target_column")
            break

    if not train_file:
        # Use first dataset
        train_file = list(datasets.keys())[0]

    if not target_column:
        # Try to infer target column from common names
        columns = datasets[train_file]["columns"]
        potential_targets = ['Survived', 'target', 'label', 'class', 'y', 'outcome']

        for col in columns:
            if col in potential_targets or col.lower() in [t.lower() for t in potential_targets]:
                target_column = col
                logger.info(f"Auto-detected target column: {target_column}")
                break

        if not target_column:
            # Fallback: use second column if first is ID-like, otherwise last
            if columns[0].lower() in ['id', 'passengerid', 'customerid']:
                target_column = columns[1]
                logger.warning(f"Using second column as target: {target_column}")
            else:
                target_column = columns[-1]
                logger.warning(f"Using last column as target: {target_column}")

    logger.info(f"Using training file: {train_file}")
    logger.info(f"Target column: {target_column}")

    # Prepare training context
    data_path = Path(data_results["data_path"]) / train_file

    training_context = {
        "data_path": str(data_path),
        "target_column": target_column,
        "config": training_config
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
