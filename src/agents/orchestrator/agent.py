"""
Orchestrator Agent
Central control unit that coordinates all specialized agents and manages
the competition lifecycle.
"""
import logging
from typing import Any, Dict

from ..base import BaseAgent, AgentState
from ..data_collector import DataCollectorAgent
from ..model_trainer import ModelTrainerAgent
from ..submission import SubmissionAgent
from ..leaderboard import LeaderboardMonitorAgent

from .phases import (
    run_data_collection,
    run_initial_training,
    run_submission,
    log_phase_results
)
from .optimization import run_optimization_loop

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
        self.previous_submission_hash = None  # Track previous submission
        self.tried_models = []  # Track which models we've tried

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
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
            data_results = await run_data_collection(self, context)
            log_phase_results("Data Collection", data_results)

            # Phase 2: Initial Model Training
            logger.info("\n=== PHASE 2: INITIAL MODEL TRAINING ===")
            training_results = await run_initial_training(
                self,
                data_results,
                context.get("training_config", {})
            )
            log_phase_results("Initial Training", training_results)

            # Phase 3: First Submission
            logger.info("\n=== PHASE 3: FIRST SUBMISSION ===")
            submission_results = await run_submission(
                self,
                training_results,
                data_results
            )
            log_phase_results("Submission", submission_results)

            # Phase 4: Monitoring and Iteration Loop
            logger.info("\n=== PHASE 4: MONITORING & OPTIMIZATION ===")
            final_results = await run_optimization_loop(
                self,
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

    def get_workflow_summary(self) -> Dict[str, Any]:
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