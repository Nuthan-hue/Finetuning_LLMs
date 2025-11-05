"""
Orchestrator Optimization Loop
Handles iterative model improvement based on leaderboard feedback.
"""
import logging
import asyncio
from typing import Any, Dict
from datetime import datetime

from ..base import AgentState
from .strategies import select_optimization_strategy, improve_training_config
from .phases import run_initial_training, run_submission

logger = logging.getLogger(__name__)

# Try to import AI agents
try:
    from ..llm_agents import StrategyAgent
    AI_AVAILABLE = True
    logger.info("✓ AI Strategy Agent available")
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning(f"⚠️  AI agents not available: {e}")


async def run_optimization_loop(
    orchestrator,
    data_results: Dict[str, Any],
    training_results: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the optimization loop until target is met or max iterations reached."""
    logger.info("Entering optimization loop...")

    while orchestrator.iteration < orchestrator.max_iterations:
        orchestrator.iteration += 1
        logger.info(f"\n--- Iteration {orchestrator.iteration}/{orchestrator.max_iterations} ---")

        # Wait for submission to be processed (Kaggle takes time)
        logger.info("Waiting for submission to be processed...")
        await asyncio.sleep(60)  # Wait 1 minute

        # Check leaderboard
        logger.info("Checking leaderboard...")
        leaderboard_context = {
            "competition_name": orchestrator.competition_name,
            "target_percentile": orchestrator.target_percentile
        }

        leaderboard_results = await orchestrator.leaderboard_monitor.run(leaderboard_context)

        if orchestrator.leaderboard_monitor.state == AgentState.ERROR:
            logger.warning(f"Leaderboard check failed: {orchestrator.leaderboard_monitor.error}")
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
        orchestrator.workflow_history.append({
            "iteration": orchestrator.iteration,
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

        # Get recommendation and select optimization strategy
        recommendation = leaderboard_results.get("recommendation")
        logger.info(f"Leaderboard recommendation: {recommendation}")

        # ═══════════════════════════════════════════════════════════
        # 🤖 USE AI AGENT for strategy selection (not hardcoded!)
        # ═══════════════════════════════════════════════════════════
        if AI_AVAILABLE:
            logger.info("🤖 Asking AI Strategy Agent for next move...")

            try:
                # Initialize AI agent
                strategy_agent = StrategyAgent()

                # Get AI decision
                competition_type = data_results.get("analysis_report", {}).get("competition_type", "tabular")

                strategy = await strategy_agent.select_optimization_strategy(
                    recommendation=recommendation,
                    current_model=training_results.get("model_type", "lightgbm"),
                    tried_models=orchestrator.tried_models,
                    current_percentile=current_percentile,
                    target_percentile=orchestrator.target_percentile,
                    iteration=orchestrator.iteration,
                    competition_type=competition_type,
                    performance_history=orchestrator.workflow_history
                )

                # Log AI decision
                logger.info(f"🤖 AI Strategy: {strategy['action']}")
                logger.info(f"💭 AI Reasoning: {strategy.get('reasoning', 'N/A')[:200]}...")
                logger.info(f"📊 Expected Improvement: {strategy.get('expected_improvement', 'N/A')}")
                logger.info(f"🎯 Confidence: {strategy.get('confidence', 'N/A')}")

            except Exception as e:
                logger.warning(f"⚠️  AI agent failed: {e}, using fallback...")
                # Fallback to hardcoded
                strategy = select_optimization_strategy(
                    recommendation,
                    training_results.get("model_type", "lightgbm"),
                    orchestrator.tried_models,
                    current_percentile,
                    orchestrator.target_percentile
                )
        else:
            # No AI available, use hardcoded logic
            logger.info("Using hardcoded strategy selection...")
            strategy = select_optimization_strategy(
                recommendation,
                training_results.get("model_type", "lightgbm"),
                orchestrator.tried_models,
                current_percentile,
                orchestrator.target_percentile
            )

        logger.info(f"Selected strategy: {strategy['action']}")

        # Apply strategy
        if strategy["action"] == "switch_model":
            # Switch to different model
            logger.info(f"Switching to {strategy['new_model']}...")
            new_config = context.get("training_config", {}).copy()
            new_config.update(strategy["config_updates"])
            orchestrator.tried_models.append(strategy["new_model"])

            training_results = await run_initial_training(
                orchestrator,
                data_results,
                new_config
            )

            submission_results = await run_submission(
                orchestrator,
                training_results,
                data_results
            )

        elif strategy["action"] in ["retrain", "tune_aggressive"]:
            # Improve hyperparameters and retrain
            is_aggressive = strategy.get("aggressive", False)
            logger.info(f"Retraining with {'aggressive' if is_aggressive else 'moderate'} tuning...")

            # ═══════════════════════════════════════════════════════════
            # 🤖 USE AI-SUGGESTED hyperparameters (not hardcoded formulas!)
            # ═══════════════════════════════════════════════════════════
            base_config = context.get("training_config", {}).copy()

            # Check if AI provided specific hyperparameter suggestions
            if "config_updates" in strategy and strategy["config_updates"]:
                logger.info("📋 Using AI-suggested hyperparameters:")
                for key, value in strategy["config_updates"].items():
                    logger.info(f"  {key}: {value}")
                base_config.update(strategy["config_updates"])
                improved_config = base_config
            else:
                # Fallback to hardcoded improvement
                logger.info("⚠️  No AI suggestions, using fallback hyperparameter tuning...")
                improved_config = improve_training_config(
                    base_config,
                    current_percentile,
                    orchestrator.target_percentile,
                    aggressive=is_aggressive
                )

            training_results = await run_initial_training(
                orchestrator,
                data_results,
                improved_config
            )

            submission_results = await run_submission(
                orchestrator,
                training_results,
                data_results
            )

    # Max iterations reached
    logger.info("Maximum iterations reached")

    # Final leaderboard check
    final_leaderboard = await orchestrator.leaderboard_monitor.run({
        "competition_name": orchestrator.competition_name
    })

    return {
        "final_rank": final_leaderboard.get("current_rank", "N/A"),
        "final_percentile": final_leaderboard.get("current_percentile", 1.0),
        "final_score": final_leaderboard.get("score"),
        "target_met": final_leaderboard.get("meets_target", False)
    }