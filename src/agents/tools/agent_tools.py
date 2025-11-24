"""
Agent Toolkit
Collection of tools (functions) that agents can use to gather information and collaborate.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AgentToolkit:
    """
    Toolkit of functions that agents can call to query information.

    Agents use these tools to:
    - Learn from previous attempts
    - Get optimizer recommendations
    - Access data insights
    - Compare model performance
    - Track leaderboard trends
    """

    def __init__(self, competition_name: str, memory_store, data_dir: str = "data"):
        """
        Initialize toolkit.

        Args:
            competition_name: Name of the competition
            memory_store: AgentMemory instance for shared memory access
            data_dir: Base directory for data storage
        """
        self.competition_name = competition_name
        self.memory = memory_store
        self.data_dir = Path(data_dir)
        self.test_cache_dir = self.data_dir / competition_name / "test"

    def get_available_tools(self) -> Dict[str, Dict[str, str]]:
        """
        Return descriptions of all available tools for AI to understand.

        Returns:
            Dictionary mapping tool names to their descriptions and metadata
        """
        return {
            "query_optimization_history": {
                "description": "Get previous optimization attempts and their results. Use this to learn what models/strategies were tried before and their performance.",
                "parameters": "None",
                "returns": "List of previous attempts with models, scores, and outcomes"
            },
            "get_optimizer_recommendation": {
                "description": "Get the latest recommendation from the Optimizer Agent. Use this when you need guidance on what to try next.",
                "parameters": "None",
                "returns": "Current optimization strategy with action, reasoning, and suggestions"
            },
            "query_leaderboard_history": {
                "description": "Get historical leaderboard positions and trends. Use this to understand performance trajectory over iterations.",
                "parameters": "None",
                "returns": "Time series of rank, percentile, and scores with trend analysis"
            },
            "get_data_insights": {
                "description": "Get insights from data analysis phase (Phase 3). Use this when planning features or preprocessing.",
                "parameters": "None",
                "returns": "Data statistics, correlations, missing values analysis, and recommendations"
            },
            "get_model_performance": {
                "description": "Get detailed performance metrics for previously trained models. Use this to compare model architectures.",
                "parameters": "model_type: string (optional) - Filter by specific model",
                "returns": "CV scores, training times, feature importances for models"
            },
            "get_best_attempt": {
                "description": "Get information about the best performing iteration so far.",
                "parameters": "None",
                "returns": "Best attempt details with model, scores, and rank"
            },
            "get_performance_trend": {
                "description": "Get overall performance trend analysis across iterations.",
                "parameters": "None",
                "returns": "Trend status: 'improving', 'plateauing', or 'degrading'"
            }
        }

    def query_optimization_history(self) -> Dict[str, Any]:
        """
        Tool: Get complete optimization history.

        Returns:
            Dictionary with all attempts, models tried, and summary statistics
        """
        logger.debug("Tool called: query_optimization_history()")

        attempts = self.memory.get_attempt_history()
        models_tried = self.memory.get_models_tried()

        result = {
            "total_attempts": len(attempts),
            "attempts": attempts,
            "models_tried": models_tried,
            "competition": self.competition_name
        }

        logger.info(f"Returned {len(attempts)} previous attempts")
        return result

    def get_optimizer_recommendation(self) -> Dict[str, Any]:
        """
        Tool: Get current optimizer recommendation.

        Returns:
            Optimization strategy or message if no recommendation exists
        """
        logger.debug("Tool called: get_optimizer_recommendation()")

        recommendation = self.memory.get_optimization_guidance()

        if not recommendation:
            return {
                "message": "No optimization recommendation yet. This is likely the first iteration.",
                "available": False
            }

        result = {
            "available": True,
            **recommendation
        }

        logger.info(f"Returned recommendation: {recommendation.get('action')}")
        return result

    def query_leaderboard_history(self) -> Dict[str, Any]:
        """
        Tool: Get leaderboard position history and trends.

        Returns:
            Historical rank/percentile data with trend analysis
        """
        logger.debug("Tool called: query_leaderboard_history()")

        attempts = self.memory.get_attempt_history()

        # Extract leaderboard data
        history = []
        for attempt in attempts:
            if "rank" in attempt:
                history.append({
                    "iteration": attempt["iteration"],
                    "rank": attempt["rank"],
                    "percentile": attempt.get("percentile", 1.0),
                    "score": attempt.get("leaderboard_score"),
                    "timestamp": attempt.get("timestamp")
                })

        # Analyze trend
        trend = self.memory.get_performance_trend()

        result = {
            "history": history,
            "trend": trend,
            "total_data_points": len(history)
        }

        if history:
            result["latest"] = history[-1]
            result["best"] = min(history, key=lambda x: x.get("percentile", 1.0))

        logger.info(f"Returned {len(history)} leaderboard data points, trend: {trend}")
        return result

    def get_data_insights(self) -> Dict[str, Any]:
        """
        Tool: Get data analysis insights from Phase 3.

        Returns:
            Data statistics, correlations, and analysis results
        """
        logger.debug("Tool called: get_data_insights()")

        # Load Phase 3 cache
        phase3_cache = self.test_cache_dir / "test_phase3_cache.json"

        if not phase3_cache.exists():
            logger.warning("Phase 3 cache not found")
            return {"error": "Data analysis not available. Run Phase 3 first."}

        try:
            with open(phase3_cache, 'r') as f:
                phase3_data = json.load(f)

            # Extract relevant insights
            data_analysis = phase3_data.get("data_analysis", {})

            result = {
                "data_modality": phase3_data.get("data_modality"),
                "target_column": phase3_data.get("target_column"),
                "preprocessing_required": phase3_data.get("needs_preprocessing"),
                "insights": data_analysis.get("insights", []),
                "recommendations": data_analysis.get("recommendations", [])
            }

            logger.info("Returned data insights from Phase 3")
            return result

        except Exception as e:
            logger.error(f"Failed to load data insights: {e}")
            return {"error": str(e)}

    def get_model_performance(self, model_type: str = None) -> Dict[str, Any]:
        """
        Tool: Get model performance metrics.

        Args:
            model_type: Optional model type to filter by

        Returns:
            Performance metrics for models
        """
        logger.debug(f"Tool called: get_model_performance(model_type={model_type})")

        attempts = self.memory.get_attempt_history()

        # Filter by model type if specified
        if model_type:
            attempts = [a for a in attempts if a.get("model") == model_type]

        # Extract model performance data
        models = []
        for attempt in attempts:
            if "model" in attempt:
                models.append({
                    "model": attempt.get("model"),
                    "iteration": attempt.get("iteration"),
                    "cv_score": attempt.get("cv_score"),
                    "leaderboard_score": attempt.get("leaderboard_score"),
                    "rank": attempt.get("rank"),
                    "percentile": attempt.get("percentile")
                })

        result = {
            "models": models,
            "total_models": len(models)
        }

        # Add comparison if multiple models
        if len(models) > 1:
            result["best_cv"] = max(models, key=lambda x: x.get("cv_score", 0))
            result["best_leaderboard"] = min(models, key=lambda x: x.get("percentile", 1.0))

        logger.info(f"Returned performance data for {len(models)} models")
        return result

    def get_best_attempt(self) -> Dict[str, Any]:
        """
        Tool: Get the best performing attempt so far.

        Returns:
            Best attempt details
        """
        logger.debug("Tool called: get_best_attempt()")

        best = self.memory.get_best_attempt()

        if not best:
            return {"message": "No attempts recorded yet."}

        logger.info(f"Returned best attempt: iteration {best.get('iteration')}")
        return best

    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Tool: Get performance trend analysis.

        Returns:
            Trend status and analysis
        """
        logger.debug("Tool called: get_performance_trend()")

        trend = self.memory.get_performance_trend()
        attempts = self.memory.get_attempt_history()

        result = {
            "trend": trend,
            "total_attempts": len(attempts)
        }

        if attempts and len(attempts) >= 2:
            recent = attempts[-3:] if len(attempts) >= 3 else attempts
            percentiles = [a.get("percentile", 1.0) for a in recent if "percentile" in a]

            if len(percentiles) >= 2:
                improvement = (percentiles[0] - percentiles[-1]) / percentiles[0] * 100
                result["improvement_percentage"] = round(improvement, 2)
                result["improving"] = improvement > 0

        logger.info(f"Returned trend: {trend}")
        return result

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Tool: Get a high-level summary of all memory contents.

        Returns:
            Summary of attempts, models, trends, and recommendations
        """
        logger.debug("Tool called: get_memory_summary()")

        summary = self.memory.get_summary()
        logger.info("Returned memory summary")
        return summary