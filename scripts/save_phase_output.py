import os
import json
from typing import Dict, Any

def save_phase_output(self, file_path: str, phase_name: str, context: Dict[str, Any]) -> None:
    """Save the output of a phase to a JSON file in the specified path."""
    os.makedirs(file_path, exist_ok=True)
    full_path = os.path.join(file_path, f"{phase_name}_iteration_{self.iteration}.json")
    with open(full_path, "w") as f:
        json.dump(context, f, indent=2)
