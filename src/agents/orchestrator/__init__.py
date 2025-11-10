"""
Orchestrator Package
Central coordinator for managing competition workflow.
"""
from .orchestrator import Orchestrator
from .orchestrator_agentic import AgenticOrchestrator

__all__ = [
    "Orchestrator",  # Legacy: Scripted pipeline
    "AgenticOrchestrator",  # New: Truly agentic with coordinator
]