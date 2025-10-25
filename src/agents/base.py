from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
        self.state = AgentState.IDLE
        self.error: Optional[str] = None
        self.results: Dict[str, Any] = {}

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method for the agent.

        Args:
            context: Dictionary containing the execution context

        Returns:
            Dictionary containing the results of the agent's execution
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return {
            "name": self.name,
            "state": self.state.value,
            "error": self.error,
            "results": self.results
        }

    def set_error(self, error: str) -> None:
        """Set an error state for the agent."""
        self.error = error
        self.state = AgentState.ERROR

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.state = AgentState.IDLE
        self.error = None
        self.results = {}
