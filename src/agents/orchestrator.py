"""
Orchestrator Agent
Central control unit that coordinates all specialized agents and manages
the competition lifecycle.
"""
import logging
import asyncio
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

from .base import BaseAgent, AgentState
from .data_collector import DataCollectorAgent
from .model_trainer import ModelTrainerAgent
from .submission import SubmissionAgent
from .leaderboard import LeaderboardMonitorAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Central orchestrator that manages the entire competition workflow."""

    def __init__(
        self,
        name: str = "Orchestrator",
        competition_name: str = None,
        target_percentile: float = 0.20,
        max_iterations: int = 5,
        data_dir: str = "data",
        models_dir: str = "models",
        submissions_dir: str = "submissions"
    ):
        super().__init__(name)
        self.competition_name = competition_name
        self.target_percentile = target_percentile
        self.max_iterations = max_iterations

        # Initialize specialized agents
        self.data_collector = DataCollectorAgent(data_dir=data_dir)
        self.model_trainer = ModelTrainerAgent(models_dir=models_dir)
        self.submission_agent = SubmissionAgent(submissions_dir=submissions_dir)
        self.leaderboard_monitor = LeaderboardMonitorAgent(
            target_percentile=target_percentile
        )

        self.iteration = 0
        self.workflow_history = []

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Main execution method for orchestration.

        Args:
            context: Dictionary containing:
                - competition_name: str - Kaggle competition name (optional if set in __init__)
                - target_percentile: float - Target ranking (optional)
                - external_sources: List[str] - External data sources (optional)
                - training_config: Dict - Model training configuration (optional)

        Returns:
            Dictionary containing:
                - final_rank: Final leaderboard position
                - final_score: Final score
                - iterations: Number of iterations performed
                - workflow_history: Complete workflow history
        """
        self.state = AgentState.RUNNING

        try:
            # Get competition name
            competition_name = context.get("competition_name") or self.competition_name
            if not competition_name:
                raise ValueError("competition_name must be provided")

            self.competition_name = competition_name
            logger.info(f"Starting orchestration for competition: {competition_name}")
            logger.info(f"Target: Top {self.target_percentile * 100}%")

            # Phase 1: Initial Data Collection and Analysis
            logger.info("\n=== PHASE 1: DATA COLLECTION ===")
            data_results = await self._run_data_collection(context)
            self._log_phase_results("Data Collection", data_results)

            # Phase 2: Initial Model Training
            logger.info("\n=== PHASE 2: INITIAL MODEL TRAINING ===")
            training_results = await self._run_initial_training(
                data_results,
                context.get("training_config", {})
            )
            self._log_phase_results("Initial Training", training_results)

            # Phase 3: First Submission
            logger.info("\n=== PHASE 3: FIRST SUBMISSION ===")
            submission_results = await self._run_submission(
                training_results,
                data_results
            )
            self._log_phase_results("Submission", submission_results)

            # Phase 4: Monitoring and Iteration Loop
            logger.info("\n=== PHASE 4: MONITORING & OPTIMIZATION ===")
            final_results = await self._optimization_loop(
                data_results,
                training_results,
                context
            )

            self.results.update(final_results)
            self.results["workflow_history"] = self.workflow_history
            self.results["total_iterations"] = self.iteration

            self.state = AgentState.COMPLETED
            logger.info("\n=== ORCHESTRATION COMPLETED ===")
            logger.info(f"Total iterations: {self.iteration}")
            logger.info(f"Final rank: {final_results.get('final_rank', 'N/A')}")

            return self.results

        except Exception as e:
            error_msg = f"Orchestration error: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _run_data_collection(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute data collection phase."""
        logger.info("Initiating data collection...")

        collection_context = {
            "competition_name": self.competition_name,
            "external_sources": context.get("external_sources", []),
            "analyze": True
        }

        results = await self.data_collector.run(collection_context)

        # Check for errors
        if self.data_collector.state == AgentState.ERROR:
            raise RuntimeError(f"Data collection failed: {self.data_collector.error}")

        return results

    async def _run_initial_training(
        self,
        data_results: dict[str, Any],
        training_config: dict[str, Any]
    ) -> dict[str, Any]:
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
            # Try to infer target column
            logger.warning("Target column not found, using last column")
            target_column = datasets[train_file]["columns"][-1]

        logger.info(f"Using training file: {train_file}")
        logger.info(f"Target column: {target_column}")

        # Prepare training context
        data_path = Path(data_results["data_path"]) / train_file

        training_context = {
            "data_path": str(data_path),
            "target_column": target_column,
            "config": training_config
        }

        results = await self.model_trainer.run(training_context)

        if self.model_trainer.state == AgentState.ERROR:
            raise RuntimeError(f"Model training failed: {self.model_trainer.error}")

        return results

    async def _run_submission(
        self,
        training_results: dict[str, Any],
        data_results: dict[str, Any]
    ) -> dict[str, Any]:
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
            "competition_name": self.competition_name,
            "model_type": training_results["model_type"],
            "submission_message": f"Iteration {self.iteration} - Automated submission",
            "auto_submit": True
        }

        # Add NLP-specific context if needed
        if training_results["model_type"] == "transformer":
            submission_context["text_column"] = training_results.get("text_column")

        results = await self.submission_agent.run(submission_context)

        if self.submission_agent.state == AgentState.ERROR:
            raise RuntimeError(f"Submission failed: {self.submission_agent.error}")

        return results

    async def _optimization_loop(
        self,
        data_results: dict[str, Any],
        training_results: dict[str, Any],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the optimization loop until target is met or max iterations reached."""
        logger.info("Entering optimization loop...")

        while self.iteration < self.max_iterations:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration}/{self.max_iterations} ---")

            # Wait for submission to be processed (Kaggle takes time)
            logger.info("Waiting for submission to be processed...")
            await asyncio.sleep(60)  # Wait 1 minute

            # Check leaderboard
            logger.info("Checking leaderboard...")
            leaderboard_context = {
                "competition_name": self.competition_name,
                "target_percentile": self.target_percentile
            }

            leaderboard_results = await self.leaderboard_monitor.run(leaderboard_context)

            if self.leaderboard_monitor.state == AgentState.ERROR:
                logger.warning(f"Leaderboard check failed: {self.leaderboard_monitor.error}")
                break

            # Log current standing
            current_rank = leaderboard_results.get("current_rank", "N/A")
            total_teams = leaderboard_results.get("total_teams", "N/A")
            current_percentile = leaderboard_results.get("current_percentile", 1.0)
            meets_target = leaderboard_results.get("meets_target", False)

            logger.info(f"Current rank: {current_rank}/{total_teams}")
            logger.info(f"Current percentile: {current_percentile * 100:.2f}%")
            logger.info(f"Meets target: {meets_target}")

            # Record iteration
            self.workflow_history.append({
                "iteration": self.iteration,
                "timestamp": datetime.now().isoformat(),
                "rank": current_rank,
                "percentile": current_percentile,
                "meets_target": meets_target,
                "recommendation": leaderboard_results.get("recommendation")
            })

            # Check if target is met
            if meets_target:
                logger.info("🎉 Target achieved! Stopping optimization loop.")
                return {
                    "final_rank": current_rank,
                    "final_percentile": current_percentile,
                    "final_score": leaderboard_results.get("score"),
                    "target_met": True
                }

            # Get recommendation
            recommendation = leaderboard_results.get("recommendation")
            logger.info(f"Recommendation: {recommendation}")

            # Decide on next action
            if recommendation in ["minor_improvement_needed", "retrain_with_tuning"]:
                logger.info("Retraining model with hyperparameter tuning...")

                # Update training config with tuning
                improved_config = self._improve_training_config(
                    context.get("training_config", {}),
                    current_percentile
                )

                # Retrain
                training_results = await self._run_initial_training(
                    data_results,
                    improved_config
                )

                # Submit new model
                submission_results = await self._run_submission(
                    training_results,
                    data_results
                )

            elif recommendation == "major_improvement_needed":
                logger.info("Major improvements needed - trying different approach...")

                # Try different model type or collect more data
                # For now, just retrain with aggressive tuning
                improved_config = self._improve_training_config(
                    context.get("training_config", {}),
                    current_percentile,
                    aggressive=True
                )

                training_results = await self._run_initial_training(
                    data_results,
                    improved_config
                )

                submission_results = await self._run_submission(
                    training_results,
                    data_results
                )

            else:
                # Maintain or slight tweaks
                logger.info("Making minor adjustments...")
                continue

        # Max iterations reached
        logger.info("Maximum iterations reached")

        # Final leaderboard check
        final_leaderboard = await self.leaderboard_monitor.run({
            "competition_name": self.competition_name
        })

        return {
            "final_rank": final_leaderboard.get("current_rank", "N/A"),
            "final_percentile": final_leaderboard.get("current_percentile", 1.0),
            "final_score": final_leaderboard.get("score"),
            "target_met": final_leaderboard.get("meets_target", False)
        }

    def _improve_training_config(
        self,
        base_config: dict[str, Any],
        current_percentile: float,
        aggressive: bool = False
    ) -> dict[str, Any]:
        """Improve training configuration based on current performance."""
        improved_config = base_config.copy()

        gap = current_percentile - self.target_percentile

        if aggressive or gap > 0.15:
            # Major improvements needed
            improved_config.update({
                "num_boost_round": improved_config.get("num_boost_round", 1000) + 500,
                "learning_rate": improved_config.get("learning_rate", 0.05) * 0.5,
                "num_leaves": improved_config.get("num_leaves", 31) + 20,
                "max_depth": improved_config.get("max_depth", 6) + 2,
                "n_estimators": improved_config.get("n_estimators", 1000) + 500
            })
        else:
            # Minor improvements
            improved_config.update({
                "num_boost_round": improved_config.get("num_boost_round", 1000) + 200,
                "learning_rate": improved_config.get("learning_rate", 0.05) * 0.8,
                "n_estimators": improved_config.get("n_estimators", 1000) + 200
            })

        logger.info(f"Updated training config: {improved_config}")
        return improved_config

    def _log_phase_results(self, phase_name: str, results: dict[str, Any]) -> None:
        """Log phase completion and key results."""
        logger.info(f"✓ {phase_name} completed")

        # Log key metrics
        for key, value in results.items():
            if key not in ["analysis_report", "leaderboard_data"]:
                logger.info(f"  {key}: {value}")

    def get_workflow_summary(self) -> dict[str, Any]:
        """Get summary of the entire workflow."""
        return {
            "competition": self.competition_name,
            "target_percentile": self.target_percentile,
            "iterations": self.iteration,
            "workflow_history": self.workflow_history,
            "agents_status": {
                "data_collector": self.data_collector.get_state(),
                "model_trainer": self.model_trainer.get_state(),
                "submission": self.submission_agent.get_state(),
                "leaderboard": self.leaderboard_monitor.get_state()
            }
        }

    def reset_workflow(self) -> None:
        """Reset the orchestrator for a new competition."""
        self.iteration = 0
        self.workflow_history = []
        self.results = {}
        self.state = AgentState.IDLE

        # Reset all agents
        self.data_collector.reset()
        self.model_trainer.reset()
        self.submission_agent.reset()
        self.leaderboard_monitor.reset()