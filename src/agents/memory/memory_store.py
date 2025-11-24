"""
Agent Memory Store
Shared memory that all agents can read and write to enable learning and collaboration.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Shared memory store for multi-agent collaboration.

    All agents can read and write to this memory to:
    - Record iteration results
    - Share optimization strategies
    - Learn from previous attempts
    - Track performance trends
    """

    def __init__(self, competition_name: str, data_dir: str = "data"):
        """
        Initialize memory store.

        Args:
            competition_name: Name of the competition
            data_dir: Base directory for data storage
        """
        self.competition_name = competition_name
        self.memory_file = Path(data_dir) / competition_name / "agent_memory.json"
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk or create new"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                logger.info(f"Loaded memory with {len(memory.get('attempts', []))} attempts")
                return memory
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}. Creating new memory.")

        # Initialize empty memory
        return {
            "competition_name": self.competition_name,
            "attempts": [],
            "current_optimization": None,
            "performance_trend": None,
            "models_tried": []
        }

    def _save_memory(self):
        """Save memory to disk"""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.debug(f"Memory saved to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def record_attempt(self, iteration: int, data: Dict[str, Any]):
        """
        Record the results of an optimization iteration.

        Args:
            iteration: Iteration number
            data: Dictionary containing:
                - model: Model type used
                - cv_score: Cross-validation score
                - leaderboard_score: Kaggle leaderboard score
                - rank: Leaderboard rank
                - percentile: Percentile rank
                - meets_target: Whether target was met
                - (any other relevant metrics)
        """
        attempt = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            **data
        }

        self.memory["attempts"].append(attempt)

        # Update models tried list
        if "model" in data and data["model"] not in self.memory["models_tried"]:
            self.memory["models_tried"].append(data["model"])

        # Update performance trend
        self.memory["performance_trend"] = self._analyze_trend()

        self._save_memory()
        logger.info(f"Recorded attempt #{iteration}: {data.get('model')} → {data.get('leaderboard_score')}")

    def get_attempt_history(self) -> List[Dict[str, Any]]:
        """
        Get all previous iteration attempts.

        Returns:
            List of attempt dictionaries
        """
        return self.memory.get("attempts", [])

    def get_optimization_guidance(self) -> Optional[Dict[str, Any]]:
        """
        Get the current optimization recommendation from Optimizer Agent.

        Returns:
            Optimization strategy dict or None if no recommendation
        """
        return self.memory.get("current_optimization")

    def set_optimization_strategy(self, strategy: Dict[str, Any]):
        """
        Optimizer Agent writes its recommendation here.

        Args:
            strategy: Dictionary containing:
                - action: Optimization action to take
                - reasoning: Why this action is recommended
                - new_model: Model to try (if applicable)
                - phases_to_rerun: List of phase numbers
                - config_updates: Hyperparameter updates
                - expected_improvement: Expected impact
                - confidence: Confidence level
        """
        self.memory["current_optimization"] = strategy
        self._save_memory()
        logger.info(f"Optimization strategy set: {strategy.get('action')}")

    def get_models_tried(self) -> List[str]:
        """Get list of all models that have been tried"""
        return self.memory.get("models_tried", [])

    def get_performance_trend(self) -> Optional[str]:
        """
        Get performance trend analysis.

        Returns:
            "improving", "plateauing", "degrading", or None
        """
        return self.memory.get("performance_trend")

    def _analyze_trend(self) -> Optional[str]:
        """Analyze performance trend from attempts"""
        attempts = self.memory.get("attempts", [])

        if len(attempts) < 2:
            return None

        # Get last 3 percentiles
        recent = attempts[-3:] if len(attempts) >= 3 else attempts
        percentiles = [a.get("percentile", 1.0) for a in recent if "percentile" in a]

        if len(percentiles) < 2:
            return None

        # Calculate trend
        if percentiles[-1] < percentiles[0] * 0.9:  # 10% improvement
            return "improving"
        elif percentiles[-1] > percentiles[0] * 1.1:  # 10% degradation
            return "degrading"
        else:
            return "plateauing"

    def get_best_attempt(self) -> Optional[Dict[str, Any]]:
        """Get the best performing attempt so far"""
        attempts = self.memory.get("attempts", [])

        if not attempts:
            return None

        # Find attempt with lowest percentile (best rank)
        valid_attempts = [a for a in attempts if "percentile" in a]
        if not valid_attempts:
            return None

        return min(valid_attempts, key=lambda x: x.get("percentile", 1.0))

    def get_latest_attempt(self) -> Optional[Dict[str, Any]]:
        """Get the most recent attempt"""
        attempts = self.memory.get("attempts", [])
        return attempts[-1] if attempts else None

    def clear_memory(self):
        """Clear all memory (use with caution!)"""
        self.memory = {
            "competition_name": self.competition_name,
            "attempts": [],
            "current_optimization": None,
            "performance_trend": None,
            "models_tried": []
        }
        self._save_memory()
        logger.warning("Memory cleared!")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of memory contents"""
        attempts = self.memory.get("attempts", [])
        best = self.get_best_attempt()
        latest = self.get_latest_attempt()

        return {
            "total_attempts": len(attempts),
            "models_tried": self.memory.get("models_tried", []),
            "performance_trend": self.memory.get("performance_trend"),
            "best_attempt": {
                "iteration": best.get("iteration") if best else None,
                "model": best.get("model") if best else None,
                "percentile": best.get("percentile") if best else None
            } if best else None,
            "latest_attempt": {
                "iteration": latest.get("iteration") if latest else None,
                "model": latest.get("model") if latest else None,
                "percentile": latest.get("percentile") if latest else None
            } if latest else None,
            "has_optimization": self.memory.get("current_optimization") is not None
        }