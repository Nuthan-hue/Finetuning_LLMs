"""
Submission Agent
Responsible for generating predictions and submitting to Kaggle competitions.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import subprocess

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class SubmissionAgent(BaseAgent):
    """Agent responsible for generating predictions and submitting solutions."""

    def __init__(
        self,
        name: str = "Submission",
        submissions_dir: str = "submissions"
    ):
        super().__init__(name)
        self.submissions_dir = Path(submissions_dir)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        # Set custom Kaggle config directory
        self.kaggle_config_dir = Path("/Volumes/Crucial X9 Pro For Mac/Finetuning_LLMs/.kaggle")

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
            predictions = await self._generate_predictions(
                model_path,
                test_data_path,
                model_type,
                context
            )

            # Format submission file
            submission_path = await self._format_submission(
                predictions,
                competition_name,
                context.get("format_spec", {})
            )

            self.results["submission_path"] = str(submission_path)

            # Submit to Kaggle
            if context.get("auto_submit", True):
                submission_result = await self._submit_to_kaggle(
                    submission_path,
                    competition_name,
                    context.get("submission_message", "Automated submission")
                )
                self.results.update(submission_result)

            self.state = AgentState.COMPLETED
            logger.info("Submission completed successfully")

            return self.results

        except Exception as e:
            error_msg = f"Error during submission: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _generate_predictions(
        self,
        model_path: str,
        test_data_path: str,
        model_type: str,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate predictions using the trained model."""
        logger.info(f"Loading model from: {model_path}")

        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Test data shape: {test_df.shape}")

        # Generate predictions based on model type
        if model_type == "lightgbm":
            predictions = await self._predict_lightgbm(model_path, test_df)
        elif model_type == "xgboost":
            predictions = await self._predict_xgboost(model_path, test_df)
        elif model_type == "pytorch_mlp":
            predictions = await self._predict_pytorch_mlp(model_path, test_df, context)
        elif model_type == "transformer":
            predictions = await self._predict_transformer(model_path, test_df, context)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return predictions

    async def _predict_lightgbm(
        self,
        model_path: str,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions using LightGBM model."""
        import lightgbm as lgb

        logger.info("Loading LightGBM model...")
        model = lgb.Booster(model_file=model_path)

        # Preprocess test data (same as training)
        X_test = self._preprocess_tabular_data(test_df)

        # Generate predictions
        predictions = model.predict(X_test)

        # Create submission dataframe
        result_df = pd.DataFrame({
            'id': test_df.get('id', range(len(predictions))),
            'prediction': predictions
        })

        return result_df

    async def _predict_xgboost(
        self,
        model_path: str,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions using XGBoost model."""
        import xgboost as xgb

        logger.info("Loading XGBoost model...")
        model = xgb.Booster()
        model.load_model(model_path)

        # Preprocess test data
        X_test = self._preprocess_tabular_data(test_df)

        # Convert to DMatrix
        dtest = xgb.DMatrix(X_test)

        # Generate predictions
        predictions = model.predict(dtest)

        # Create submission dataframe
        result_df = pd.DataFrame({
            'id': test_df.get('id', range(len(predictions))),
            'prediction': predictions
        })

        return result_df

    async def _predict_pytorch_mlp(
        self,
        model_path: str,
        test_df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate predictions using PyTorch MLP model."""
        import torch
        import torch.nn as nn

        logger.info("Loading PyTorch MLP model...")

        # Load model checkpoint
        checkpoint = torch.load(model_path)
        scaler = checkpoint['scaler']
        input_size = checkpoint['input_size']
        hidden_sizes = checkpoint['hidden_sizes']

        # Recreate model architecture
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size=1):
                super().__init__()
                layers = []
                prev_size = input_size

                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_size = hidden_size

                layers.append(nn.Linear(prev_size, output_size))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLP(input_size, hidden_sizes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Preprocess test data
        X_test = self._preprocess_tabular_data(test_df)
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

        # Generate predictions
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy().flatten()

        # Create submission dataframe
        result_df = pd.DataFrame({
            'id': test_df.get('id', range(len(predictions))),
            'prediction': predictions
        })

        return result_df

    async def _predict_transformer(
        self,
        model_path: str,
        test_df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate predictions using transformer model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info("Loading transformer model...")

        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Get text column
        text_column = context.get("text_column")
        if not text_column:
            # Try to find text column
            for col in test_df.columns:
                if test_df[col].dtype == 'object':
                    avg_length = test_df[col].astype(str).str.len().mean()
                    if avg_length > 50:
                        text_column = col
                        break

        if not text_column:
            raise ValueError("Could not find text column in test data")

        # Get texts
        texts = test_df[text_column].astype(str).tolist()

        # Generate predictions in batches
        batch_size = context.get("batch_size", 16)
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Predict
            with torch.no_grad():
                outputs = model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} samples")

        # Create submission dataframe
        result_df = pd.DataFrame({
            'id': test_df.get('id', range(len(all_predictions))),
            'prediction': all_predictions
        })

        return result_df

    async def _format_submission(
        self,
        predictions: pd.DataFrame,
        competition_name: str,
        format_spec: Dict[str, Any]
    ) -> Path:
        """Format predictions into submission file."""
        logger.info("Formatting submission file...")

        # Apply custom formatting if specified
        if format_spec:
            # Rename columns if specified
            column_mapping = format_spec.get("column_mapping", {})
            if column_mapping:
                predictions = predictions.rename(columns=column_mapping)

            # Apply transformations
            transformations = format_spec.get("transformations", {})
            for col, transform in transformations.items():
                if transform == "round":
                    predictions[col] = predictions[col].round()
                elif transform == "binary":
                    predictions[col] = (predictions[col] > 0.5).astype(int)

        # Save submission file
        submission_filename = f"{competition_name}_submission.csv"
        submission_path = self.submissions_dir / submission_filename

        predictions.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")

        return submission_path

    async def _submit_to_kaggle(
        self,
        submission_path: Path,
        competition_name: str,
        message: str
    ) -> Dict[str, Any]:
        """Submit to Kaggle competition."""
        logger.info(f"Submitting to Kaggle competition: {competition_name}")

        try:
            cmd = [
                "kaggle", "competitions", "submit",
                "-c", competition_name,
                "-f", str(submission_path),
                "-m", message
            ]

            # Set custom Kaggle config directory in environment
            env = os.environ.copy()
            env["KAGGLE_CONFIG_DIR"] = str(self.kaggle_config_dir)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            logger.info("Submission successful!")
            logger.info(result.stdout)

            return {
                "submission_status": "success",
                "submission_message": result.stdout
            }

        except subprocess.CalledProcessError as e:
            error_msg = f"Kaggle submission failed: {e.stderr}"
            logger.error(error_msg)
            return {
                "submission_status": "failed",
                "submission_error": error_msg
            }

    def _preprocess_tabular_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess tabular data (same as training)."""
        df_processed = df.copy()

        # Remove ID column if present
        id_columns = ['id', 'ID', 'Id']
        for id_col in id_columns:
            if id_col in df_processed.columns:
                df_processed = df_processed.drop(columns=[id_col])

        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            # Simple label encoding
            df_processed[col] = pd.Categorical(df_processed[col]).codes

        # Fill missing values
        df_processed = df_processed.fillna(df_processed.mean())

        return df_processed

    def get_submission_path(self) -> Optional[Path]:
        """Get path to the latest submission file."""
        if "submission_path" in self.results:
            return Path(self.results["submission_path"])
        return None
