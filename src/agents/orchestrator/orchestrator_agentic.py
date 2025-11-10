"""
Truly Agentic Orchestrator

The orchestrator is now an EXECUTOR, not a DECIDER.
The CoordinatorAgent (AI) decides what to do next.
The orchestrator just executes the coordinator's decisions.

This is TRUE MULTI-AGENT ARCHITECTURE:
- Coordinator Agent = Brain (decides workflow)
- Orchestrator = Hands (executes actions)
- Specialist Agents = Tools (do specific tasks)
"""
import logging
from typing import Any, Dict
from pathlib import Path

from ..base import BaseAgent, AgentState
from ..data_collector import DataCollector
from ..llm_agents import (
    CoordinatorAgent,
    ProblemUnderstandingAgent,
    DataAnalysisAgent,
    PreprocessingAgent,
    PlanningAgent
)
from ..model_trainer import ModelTrainer
from ..submission import Submitter
from ..leaderboard import LeaderboardMonitor

from .phases import (
    run_data_collection,
    run_problem_understanding,
    run_data_analysis,
    run_preprocessing,
    run_planning,
    run_feature_engineering,
    run_model_training,
    run_submission,
    run_evaluation,
    run_optimization
)

logger = logging.getLogger(__name__)


class AgenticOrchestrator(BaseAgent):
    """
    Truly agentic orchestrator where AI coordinator decides the workflow.

    Key difference from old orchestrator:
    - OLD: Orchestrator decided workflow (hardcoded phases)
    - NEW: Coordinator Agent decides workflow (autonomous AI)

    The orchestrator is now just an action executor.
    """

    def __init__(
        self,
        name: str = "AgenticOrchestrator",
        competition_name: str = None,
        target_percentile: float = 0.20,
        max_actions: int = 50,
        data_dir: str = "data",
        models_dir: str = "models",
        submissions_dir: str = "submissions"
    ):
        super().__init__(name)
        self.competition_name = competition_name
        self.target_percentile = target_percentile
        self.max_actions = max_actions

        # The Brain - Coordinator Agent (AI decides workflow)
        self.coordinator = CoordinatorAgent()

        # The Hands - Worker components (execute actions)
        self.data_collector = DataCollector(data_dir=data_dir)
        self.problem_understanding_agent = ProblemUnderstandingAgent()
        self.model_trainer = ModelTrainer(models_dir=models_dir)
        self.submission_agent = Submitter(submissions_dir=submissions_dir)
        self.leaderboard_monitor = LeaderboardMonitor(target_percentile=target_percentile)

        self.action_history = []
        self.workflow_history = []

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution - coordinator decides, orchestrator executes.

        The workflow is now:
        1. Coordinator observes state
        2. Coordinator decides next action
        3. Orchestrator executes that action
        4. Update state
        5. Repeat until coordinator says "done"
        """
        self.state = AgentState.RUNNING

        try:
            # Initialize
            competition_name = context.get("competition_name") or self.competition_name
            if not competition_name:
                raise ValueError("competition_name must be provided")

            self.competition_name = competition_name

            # Initial state
            state = {
                "competition_name": competition_name,
                "target_percentile": self.target_percentile,
                "iteration": 0
            }

            # Goal for coordinator
            goal = f"Achieve top {self.target_percentile * 100}% ranking in {competition_name} Kaggle competition"

            logger.info("\n" + "=" * 80)
            logger.info("🚀 STARTING TRULY AGENTIC KAGGLE AUTOMATION")
            logger.info("=" * 80)
            logger.info(f"🎯 Goal: {goal}")
            logger.info(f"🧠 Coordinator Agent: Autonomous decision-making")
            logger.info(f"⚙️  Orchestrator: Action executor")
            logger.info(f"📊 Max actions: {self.max_actions}")

            # ═══════════════════════════════════════════════════════════════
            # AUTONOMOUS COORDINATION LOOP
            # ═══════════════════════════════════════════════════════════════

            action_count = 0

            while action_count < self.max_actions:
                action_count += 1

                logger.info(f"\n{'='*80}")
                logger.info(f"🧠 COORDINATOR DECISION #{action_count}")
                logger.info(f"{'='*80}")

                # COORDINATOR DECIDES (This is the key - AI decides!)
                decision = await self.coordinator.coordinate(
                    goal=goal,
                    current_state=state,
                    action_history=self.action_history,
                    max_actions=self.max_actions
                )

                action = decision.get("action")
                reasoning = decision.get("reasoning")

                logger.info(f"🎯 Coordinator decided: {action}")
                logger.info(f"💭 Reasoning: {reasoning}")

                # Check if coordinator says we're done
                if action == "done" or not decision.get("continue", True):
                    logger.info("✅ Coordinator declares: WORKFLOW COMPLETE")
                    state["final_coordinator_decision"] = decision
                    break

                # ORCHESTRATOR EXECUTES (We just follow coordinator's orders)
                try:
                    state = await self._execute_action(action, state)

                    # Record success
                    self.action_history.append({
                        "action_number": action_count,
                        "action": action,
                        "reasoning": reasoning,
                        "status": "success",
                        "state_after": self._summarize_state(state)
                    })

                except Exception as e:
                    logger.error(f"❌ Action '{action}' failed: {e}")

                    # Record failure
                    self.action_history.append({
                        "action_number": action_count,
                        "action": action,
                        "reasoning": reasoning,
                        "status": "failed",
                        "error": str(e)
                    })

                    # Let coordinator decide how to handle failure
                    state["last_action_failed"] = True
                    state["last_action_error"] = str(e)

            # ═══════════════════════════════════════════════════════════════
            # FINAL RESULTS
            # ═══════════════════════════════════════════════════════════════

            self.results = {
                "final_rank": state.get("current_rank", "N/A"),
                "final_percentile": state.get("current_percentile", 1.0),
                "final_score": state.get("leaderboard_score"),
                "target_met": state.get("meets_target", False),
                "total_actions": action_count,
                "action_history": self.action_history,
                "workflow_history": self.workflow_history,
                "final_state": state
            }

            self.state = AgentState.COMPLETED

            logger.info("\n" + "=" * 80)
            logger.info("✅ AGENTIC ORCHESTRATION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total actions taken: {action_count}")
            logger.info(f"Final rank: {self.results['final_rank']}")
            logger.info(f"Final percentile: {self.results['final_percentile'] * 100:.1f}%")
            logger.info(f"Target met: {self.results['target_met']}")

            return self.results

        except Exception as e:
            error_msg = f"Agentic orchestration error: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _execute_action(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action decided by coordinator.

        This is where coordinator's decisions become reality.
        """
        logger.info(f"⚙️  Executing action: {action}")

        # Map coordinator actions to phase functions
        action_map = {
            "collect_data": self._action_collect_data,
            "understand_problem": self._action_understand_problem,
            "analyze_data": self._action_analyze_data,
            "preprocess_data": self._action_preprocess_data,
            "plan_strategy": self._action_plan_strategy,
            "engineer_features": self._action_engineer_features,
            "train_model": self._action_train_model,
            "submit_predictions": self._action_submit_predictions,
            "evaluate_results": self._action_evaluate_results,
            "optimize_strategy": self._action_optimize_strategy
        }

        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")

        # Execute the action
        state = await action_map[action](state)

        return state

    # Action execution methods (these call the phase functions)

    async def _action_collect_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: collect_data"""
        logger.info("📥 Collecting competition data...")
        state = await run_data_collection(self, state)
        return state

    async def _action_understand_problem(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: understand_problem"""
        logger.info("📖 Understanding problem...")
        state = await run_problem_understanding(self, state)
        return state

    async def _action_analyze_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: analyze_data"""
        logger.info("🔍 Analyzing data...")
        state = await run_data_analysis(self, state)
        return state

    async def _action_preprocess_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: preprocess_data"""
        logger.info("🧹 Preprocessing data...")
        # Force preprocessing to run (coordinator decided it's needed)
        state["needs_preprocessing"] = True
        state = await run_preprocessing(self, state)
        return state

    async def _action_plan_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: plan_strategy"""
        logger.info("📋 Planning strategy...")
        state["iteration"] = state.get("iteration", 0) + 1
        state = await run_planning(self, state)
        return state

    async def _action_engineer_features(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: engineer_features"""
        logger.info("🔧 Engineering features...")
        # Force feature engineering (coordinator decided it's needed)
        state["needs_feature_engineering"] = True
        state = await run_feature_engineering(self, state)
        return state

    async def _action_train_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: train_model"""
        logger.info("🤖 Training model...")
        state = await run_model_training(self, state)

        # Record in workflow history
        self.workflow_history.append({
            "iteration": state.get("iteration", 0),
            "model_type": state.get("model_type"),
            "cv_score": state.get("cv_score")
        })

        return state

    async def _action_submit_predictions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: submit_predictions"""
        logger.info("📤 Submitting predictions...")
        state = await run_submission(self, state)

        # Update workflow history with submission
        if self.workflow_history:
            self.workflow_history[-1]["leaderboard_score"] = state.get("leaderboard_score")

        return state

    async def _action_evaluate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: evaluate_results"""
        logger.info("📊 Evaluating results...")
        state = await run_evaluation(self, state)

        # Update workflow history with evaluation
        if self.workflow_history:
            self.workflow_history[-1].update({
                "current_rank": state.get("current_rank"),
                "current_percentile": state.get("current_percentile"),
                "meets_target": state.get("meets_target")
            })

        return state

    async def _action_optimize_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute: optimize_strategy"""
        logger.info("🔄 Optimizing strategy...")
        state = await run_optimization(self, state)
        return state

    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create lightweight summary of state."""
        return {
            "has_data": state.get('data_path') is not None,
            "has_analysis": state.get('data_analysis') is not None,
            "has_plan": state.get('execution_plan') is not None,
            "has_model": state.get('model_path') is not None,
            "submitted": state.get('submission_id') is not None,
            "percentile": state.get('current_percentile'),
            "meets_target": state.get('meets_target')
        }

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of the entire workflow."""
        return {
            "competition": self.competition_name,
            "target_percentile": self.target_percentile,
            "total_actions": len(self.action_history),
            "action_history": self.action_history,
            "workflow_history": self.workflow_history,
            "agents_status": {
                "coordinator": "autonomous",
                "data_collector": self.data_collector.get_state(),
                "model_trainer": self.model_trainer.get_state(),
                "submission": self.submission_agent.get_state(),
                "leaderboard": self.leaderboard_monitor.get_state()
            }
        }

    def reset_workflow(self) -> None:
        """Reset for a new competition."""
        self.action_history = []
        self.workflow_history = []
        self.results = {}
        self.state = AgentState.IDLE

        # Reset all worker agents
        self.data_collector.reset()
        self.model_trainer.reset()
        self.submission_agent.reset()
        self.leaderboard_monitor.reset()