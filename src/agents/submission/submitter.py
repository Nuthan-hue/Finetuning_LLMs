"""
Submitter
Workflow component for generating predictions and submitting to Kaggle competitions.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..base import BaseAgent, AgentState
from .predictions import generate_predictions
from .formatting import format_submission
from .kaggle_api import submit_to_kaggle

logger = logging.getLogger(__name__)


class Submitter(BaseAgent):
    """Worker responsible for generating predictions and submitting solutions."""

    def __init__(
        self,
        name: str = "Submission",
        submissions_dir: str = "submissions"
    ):
        super().__init__(name)
        self.submissions_dir = Path(submissions_dir)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        # Set custom Kaggle config directory (use ~/.kaggle as standard)
        self.kaggle_config_dir = Path.home() / ".kaggle"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for submission.

        Args:
            context: Dictionary containing:
                - model_path: str - Path to trained model
                - test_data_path: str - Path to test data
                - competition_name: str - Name of Kaggle competition
                - model_type: str - Type of model (lightgbm, xgboost, etc.)
                - submission_message: str - Optional message for submission
                - format_spec: Dict - Specification for submission format
                - auto_submit: bool - Whether to auto-submit to Kaggle (default: True)

        Returns:
            Dictionary containing:
                - submission_path: Path to submission file
                - submission_status: Status of submission
                - submission_id: Kaggle submission ID (if successful)
        """
        self.state = AgentState.RUNNING

        try:
            model_path = context.get("model_path")
            test_data_path = context.get("test_data_path")
            competition_name = context.get("competition_name")
            model_type = context.get("model_type")

            if not all([model_path, test_data_path, competition_name]):
                raise ValueError("model_path, test_data_path, and competition_name are required")

            logger.info(f"Generating predictions for competition: {competition_name}")

            # Generate predictions
            predictions = await generate_predictions(
                model_path,
                test_data_path,
                model_type,
                context
            )

            # Format submission file
            submission_path = await format_submission(
                predictions,
                competition_name,
                context.get("format_spec", {}),
                self.submissions_dir
            )

            self.results["submission_path"] = str(submission_path)

            # Submit to Kaggle (require explicit confirmation)
            auto_submit = context.get("auto_submit", False)  # Changed default to False

            if auto_submit:
                logger.info(f"Submission file ready: {submission_path}")

                # Ask for confirmation if in interactive mode
                if context.get("interactive", True):
                    logger.info("\n" + "="*60)
                    logger.info("SUBMISSION READY")
                    logger.info("="*60)
                    logger.info(f"File: {submission_path.name}")
                    logger.info(f"Competition: {competition_name}")
                    response = input("\nSubmit to Kaggle? (yes/no): ").strip().lower()

                    if response not in ['yes', 'y']:
                        logger.info("Submission cancelled by user")
                        self.results["submission_status"] = "cancelled"
                        self.state = AgentState.COMPLETED
                        return self.results

                # Proceed with submission
                submission_result = await submit_to_kaggle(
                    submission_path,
                    competition_name,
                    context.get("submission_message", "Automated submission"),
                    self.kaggle_config_dir
                )
                self.results.update(submission_result)
            else:
                logger.info(f"Submission file created: {submission_path}")
                logger.info("Auto-submit is disabled. Use auto_submit=True to enable.")

            self.state = AgentState.COMPLETED
            logger.info("Submission completed successfully")

            return self.results

        except Exception as e:
            error_msg = f"Error during submission: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    def get_submission_path(self) -> Optional[Path]:
        """Get path to the latest submission file."""
        if "submission_path" in self.results:
            return Path(self.results["submission_path"])
        return None