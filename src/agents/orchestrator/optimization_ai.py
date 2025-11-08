"""
AI-Powered Orchestrator Optimization Loop
Uses Gemini AI agents for intelligent decision-making instead of hardcoded rules.
"""
import logging
import asyncio
from typing import Any, Dict
from datetime import datetime

from ..base import AgentState
from ..llm_agents import StrategyAgent
from .phases import run_initial_training, run_submission

logger = logging.getLogger(__name__)


async def run_optimization_loop_ai(
    orchestrator,
    data_results: Dict[str, Any],
    training_results: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run AI-powered optimization loop using Gemini for decision-making.

    This replaces hardcoded conditional logic with actual AI reasoning.
    """
    logger.info("Entering AI-powered optimization loop...")

    # Initialize AI Strategy Agent
    strategy_agent = StrategyAgent()
    logger.info("✓ Strategy Agent initialized (using Gemini AI)")

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

        # Get recommendation
        recommendation = leaderboard_results.get("recommendation")
        logger.info(f"Leaderboard recommendation: {recommendation}")

        # ═══════════════════════════════════════════════════
        # 🤖 AI AGENT DECISION MAKING (replaces hardcoded logic)
        # ═══════════════════════════════════════════════════
        logger.info("🤖 Asking AI Strategy Agent for next move...")

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
        logger.info(f"💭 AI Reasoning: {strategy.get('reasoning', 'N/A')}")
        logger.info(f"📊 Expected Improvement: {strategy.get('expected_improvement', 'N/A')}")
        logger.info(f"🎯 Confidence: {strategy.get('confidence', 'N/A')}")

        # Apply AI-recommended strategy
        if strategy["action"] == "switch_model":
            # AI decided to switch models
            new_model = strategy.get("new_model")
            logger.info(f"🔄 AI recommends switching to {new_model}...")

            orchestrator.tried_models.append(new_model)

            # AI provides all config via strategy
            new_config = strategy.get("config_updates", {})

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
            # AI decided to improve hyperparameters
            logger.info(f"⚙️ AI recommends retraining with {'aggressive' if strategy['aggressive'] else 'moderate'} tuning...")

            # Use AI-suggested config updates
            improved_config = strategy.get("config_updates", {})

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

        elif strategy["action"] == "feature_engineering":
            # AI suggests feature engineering (future implementation)
            logger.info("🔧 AI recommends feature engineering (not yet implemented)")
            logger.info("Falling back to hyperparameter tuning...")

            improved_config = strategy.get("config_updates", {})

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

        elif strategy["action"] == "ensemble":
            # AI suggests ensemble (future implementation)
            logger.info("🎭 AI recommends model ensemble (not yet implemented)")
            logger.info("Continuing with single model for now...")
            continue

        else:
            logger.warning(f"Unknown AI strategy: {strategy['action']}, continuing...")
            continue

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