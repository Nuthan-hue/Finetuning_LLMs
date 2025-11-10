"""
Orchestrator
Central coordinator that manages all workflow components and oversees
the competition lifecycle following the 10-phase architecture.
"""
import logging
from typing import Any, Dict

from ..base import BaseAgent, AgentState
from ..data_collector import DataCollector
from ..llm_agents import ProblemUnderstandingAgent
from ..model_trainer import ModelTrainer
from ..submission import Submitter
from ..leaderboard import LeaderboardMonitor
from scripts.save_phase_output import save_phase_output

from .phases import (
    run_data_collection,           # Phase 1
    run_problem_understanding,      # Phase 2
    run_data_analysis,              # Phase 3
    run_preprocessing,              # Phase 4 (conditional)
    run_planning,                   # Phase 5
    run_feature_engineering,        # Phase 6 (conditional)
    run_model_training,             # Phase 7
    run_submission,                 # Phase 8
    run_evaluation,                 # Phase 9
    run_optimization                # Phase 10 (conditional)
)

logger = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
    """Central coordinator that manages the entire competition workflow."""

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

        # Initialize workflow components
        self.data_collector = DataCollector(data_dir=data_dir)
        self.problem_understanding_agent = ProblemUnderstandingAgent()
        self.model_trainer = ModelTrainer(models_dir=models_dir)
        self.submission_agent = Submitter(submissions_dir=submissions_dir)
        self.leaderboard_monitor = LeaderboardMonitor(
            target_percentile=target_percentile
        )

        self.iteration = 0
        self.workflow_history = []
        self.previous_submission_hash = None  # Track previous submission
        self.tried_models = []  # Track which models we've tried

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for orchestration following 10-phase architecture.

        Args:
            context: Dictionary containing:
                - competition_name: str - Kaggle competition name (optional if set in __init__)
                - target_percentile: float - Target ranking (optional, default: 0.20)

        Returns:
            Dictionary containing:
                - final_rank: Final leaderboard position
                - final_score: Final score
                - iterations: Number of iterations performed
                - workflow_history: Complete workflow history
        """
        self.state = AgentState.RUNNING

        try:
            # Initialize accumulated context
            competition_name = context.get("competition_name") or self.competition_name
            if not competition_name:
                raise ValueError("competition_name must be provided")

            self.competition_name = competition_name

            # Single accumulated context dict that grows through phases
            # AI decides everything - no manual config overrides
            accumulated_context = {
                "competition_name": competition_name,
                "target_percentile": self.target_percentile
            }

            logger.info("\n" + "=" * 80)
            logger.info("🚀 STARTING KAGGLE COMPETITION AUTOMATION")
            logger.info("=" * 80)
            logger.info(f"Competition: {competition_name}")
            logger.info(f"Target: Top {self.target_percentile * 100}%")
            logger.info(f"Max iterations: {self.max_iterations}")

            # ═══════════════════════════════════════════════════════════════
            # MAIN ITERATION LOOP
            # ═══════════════════════════════════════════════════════════════
            while self.iteration < self.max_iterations:
                self.iteration += 1
                logger.info(f"\n{'=' * 80}")
                logger.info(f"📍 ITERATION {self.iteration}/{self.max_iterations}")
                logger.info(f"{'=' * 80}")

                # PHASE 1: DATA COLLECTION (only first iteration)
                if self.iteration == 1:
                    accumulated_context = await run_data_collection(self, accumulated_context)
                    print("Data Collection", accumulated_context)
                    save_phase_output(self,accumulated_context["data_path"] , "data_collection", accumulated_context)

                # PHASE 2: PROBLEM UNDERSTANDING (only first iteration)
                if self.iteration == 1:
                    accumulated_context = await run_problem_understanding(self, accumulated_context)
                    print("Problem understanding", accumulated_context)
                    save_phase_output(self, accumulated_context["data_path"],"problem_understanding", accumulated_context)

                # PHASE 3:  Exploratory DATA ANALYSIS (only first iteration)
                if self.iteration == 1:
                    accumulated_context = await run_data_analysis(self, accumulated_context)
                    print("Data Analysis", accumulated_context)
                    save_phase_output(self,accumulated_context["data_path"], "data_analysis", accumulated_context)

                # PHASE 4: PREPROCESSING (conditional, only first iteration)
                if self.iteration == 1:
                    accumulated_context = await run_preprocessing(self, accumulated_context)
                    print("Preprocessing", accumulated_context)

                # PHASE 5: FEATURE ENGINEERING (conditional)
                accumulated_context = await run_feature_engineering(self, accumulated_context)

                # PHASE 6: PLANNING (AI creates/updates execution plan)
                accumulated_context = await run_planning(self, accumulated_context)


                # PHASE 7: MODEL TRAINING (execute plan)
                accumulated_context = await run_model_training(self, accumulated_context)

                # PHASE 8: SUBMISSION
                accumulated_context = await run_submission(self, accumulated_context)

                # PHASE 9: EVALUATION
                accumulated_context = await run_evaluation(self, accumulated_context)

                # Record iteration history
                self.workflow_history.append({
                    "iteration": self.iteration,
                    "model_type": accumulated_context.get("model_type"),
                    "cv_score": accumulated_context.get("cv_score"),
                    "leaderboard_score": accumulated_context.get("leaderboard_score"),
                    "current_rank": accumulated_context.get("current_rank"),
                    "current_percentile": accumulated_context.get("current_percentile"),
                    "meets_target": accumulated_context.get("meets_target")
                })

                # Check if target achieved
                if accumulated_context.get("meets_target"):
                    logger.info("\n" + "=" * 80)
                    logger.info("🎉 TARGET ACHIEVED!")
                    logger.info("=" * 80)
                    break

                # PHASE 10: OPTIMIZATION (AI decides next move)
                accumulated_context = await run_optimization(self, accumulated_context)

                # Check if we should continue
                if not accumulated_context.get("optimization_strategy"):
                    logger.info("No optimization strategy - stopping")
                    break

            # ═══════════════════════════════════════════════════════════════
            # FINAL RESULTS
            # ═══════════════════════════════════════════════════════════════
            self.results = {
                "final_rank": accumulated_context.get("current_rank", "N/A"),
                "final_percentile": accumulated_context.get("current_percentile", 1.0),
                "final_score": accumulated_context.get("leaderboard_score"),
                "target_met": accumulated_context.get("meets_target", False),
                "total_iterations": self.iteration,
                "workflow_history": self.workflow_history,
                "final_context": accumulated_context
            }

            self.state = AgentState.COMPLETED

            logger.info("\n" + "=" * 80)
            logger.info("✅ ORCHESTRATION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total iterations: {self.iteration}")
            logger.info(f"Final rank: {self.results['final_rank']}")
            logger.info(f"Final percentile: {self.results['final_percentile'] * 100:.1f}%")
            logger.info(f"Target met: {self.results['target_met']}")

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