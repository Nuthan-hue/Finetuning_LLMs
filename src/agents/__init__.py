"""
Multi-Agent System for Kaggle Competitions

This package contains specialized agents for automating Kaggle competition workflows.
"""

from .base import BaseAgent, AgentState
from .orchestrator import OrchestratorAgent
from .data_collector import DataCollectorAgent
from .model_trainer import ModelTrainerAgent, TaskType, ModelType
from .submission import SubmissionAgent
from .leaderboard import LeaderboardMonitorAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "OrchestratorAgent",
    "DataCollectorAgent",
    "ModelTrainerAgent",
    "TaskType",
    "ModelType",
    "SubmissionAgent",
    "LeaderboardMonitorAgent",
]

__version__ = "0.1.0"