"""
Multi-Agent System for Kaggle Competitions

This package contains specialized workflow components and LLM agents for automating Kaggle competitions.
"""

from .base import BaseAgent, AgentState
from .orchestrator import Orchestrator
from .data_collector import DataCollector
from .model_trainer import ModelTrainer, TaskType, ModelType
from .submission import Submitter
from .leaderboard import LeaderboardMonitor

__all__ = [
    "BaseAgent",
    "AgentState",
    "Orchestrator",
    "DataCollector",
    "ModelTrainer",
    "TaskType",
    "ModelType",
    "Submitter",
    "LeaderboardMonitor",
]

__version__ = "0.1.0"