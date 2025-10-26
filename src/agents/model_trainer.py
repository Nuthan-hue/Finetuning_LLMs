"""
Model Trainer Agent
Responsible for model selection, training, and optimization based on data type.
Supports tabular, NLP, and computer vision tasks.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of ML tasks."""
    TABULAR = "tabular"
    NLP = "nlp"
    VISION = "vision"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PYTORCH_MLP = "pytorch_mlp"
    TRANSFORMER = "transformer"
    CNN = "cnn"


class ModelTrainerAgent(BaseAgent):
    """Agent responsible for model training and optimization."""

    def __init__(
        self,
        name: str = "ModelTrainer",
        models_dir: str = "models",
        task_type: Optional[TaskType] = None,
        model_type: Optional[ModelType] = None
    ):
        super().__init__(name)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "final").mkdir(exist_ok=True)

        self.task_type = task_type
        self.model_type = model_type
        self.model = None
        self.best_score = None

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for model training.

        Args:
            context: Dictionary containing:
                - data_path: str - Path to training data
                - target_column: str - Name of target column
                - task_type: str - Type of task (optional, auto-detected)
                - model_type: str - Specific model to use (optional)
                - config: Dict - Training configuration (optional)

        Returns:
            Dictionary containing:
                - model_path: Path to saved model
                - best_score: Best validation score
                - metrics: Training metrics
        """
        self.state = AgentState.RUNNING

        try:
            data_path = context.get("data_path")
            target_column = context.get("target_column")

            if not data_path or not target_column:
                raise ValueError("data_path and target_column are required")

            logger.info(f"Starting model training with data from: {data_path}")

            # Auto-detect task type if not specified
            if not self.task_type:
                self.task_type = await self._detect_task_type(data_path, context)

            logger.info(f"Task type: {self.task_type.value}")

            # Select and train model based on task type
            if self.task_type == TaskType.TABULAR:
                results = await self._train_tabular_model(data_path, target_column, context)
            elif self.task_type == TaskType.NLP:
                results = await self._train_nlp_model(data_path, target_column, context)
            elif self.task_type == TaskType.VISION:
                results = await self._train_vision_model(data_path, target_column, context)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            self.results.update(results)
            self.state = AgentState.COMPLETED
            logger.info("Model training completed successfully")

            return self.results

        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _detect_task_type(self, data_path: str, context: Dict[str, Any]) -> TaskType:
        """Auto-detect the task type based on data characteristics."""
        logger.info("Auto-detecting task type...")

        # Check if it's a directory (likely images) or file
        path = Path(data_path)

        if path.is_dir():
            # Check for image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            has_images = any(
                file.suffix.lower() in image_extensions
                for file in path.rglob('*')
            )
            if has_images:
                return TaskType.VISION

        # Load data if it's a file
        if path.suffix == '.csv':
            df = pd.read_csv(data_path)

            # Check for text columns (likely NLP task)
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains long text
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 50:  # Arbitrary threshold
                        text_columns.append(col)

            if text_columns:
                logger.info(f"Detected text columns: {text_columns}")
                return TaskType.NLP

            # Default to tabular for structured data
            return TaskType.TABULAR

        return TaskType.UNKNOWN

    async def _train_tabular_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a model for tabular data."""
        logger.info("Training tabular model...")

        # Load data
        df = pd.read_csv(data_path)

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Determine model type if not specified
        if not self.model_type:
            self.model_type = ModelType.LIGHTGBM  # Default

        config = context.get("config", {})

        if self.model_type == ModelType.LIGHTGBM:
            return await self._train_lightgbm(X, y, config)
        elif self.model_type == ModelType.XGBOOST:
            return await self._train_xgboost(X, y, config)
        elif self.model_type == ModelType.PYTORCH_MLP:
            return await self._train_pytorch_mlp(X, y, config)
        else:
            raise ValueError(f"Unsupported tabular model: {self.model_type}")

    async def _train_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train LightGBM model."""
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, mean_squared_error

        logger.info("Training LightGBM model...")

        # Preprocess data
        X_processed = self._preprocess_tabular_data(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # Determine task (classification or regression)
        is_classification = y.dtype == 'object' or len(y.unique()) < 20

        # Set parameters
        params = {
            'objective': 'binary' if is_classification else 'regression',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=config.get('num_boost_round', 1000),
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        y_pred = self.model.predict(X_val)

        if is_classification:
            y_pred_binary = (y_pred > 0.5).astype(int)
            score = accuracy_score(y_val, y_pred_binary)
            metric_name = "accuracy"
        else:
            score = mean_squared_error(y_val, y_pred, squared=False)
            metric_name = "rmse"

        logger.info(f"Validation {metric_name}: {score:.4f}")

        # Save model
        model_path = self.models_dir / "final" / "lightgbm_model.txt"
        self.model.save_model(str(model_path))

        return {
            "model_path": str(model_path),
            "best_score": score,
            "metric": metric_name,
            "model_type": "lightgbm"
        }

    async def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train XGBoost model."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, mean_squared_error

        logger.info("Training XGBoost model...")

        # Preprocess data
        X_processed = self._preprocess_tabular_data(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # Determine task
        is_classification = y.dtype == 'object' or len(y.unique()) < 20

        # Set parameters
        params = {
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'n_estimators': config.get('n_estimators', 1000),
            'objective': 'binary:logistic' if is_classification else 'reg:squarederror',
            'eval_metric': 'logloss' if is_classification else 'rmse',
            'early_stopping_rounds': 50,
            'verbose': False
        }

        # Train model
        self.model = xgb.XGBClassifier(**params) if is_classification else xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        # Evaluate
        y_pred = self.model.predict(X_val)

        if is_classification:
            score = accuracy_score(y_val, y_pred)
            metric_name = "accuracy"
        else:
            score = mean_squared_error(y_val, y_pred, squared=False)
            metric_name = "rmse"

        logger.info(f"Validation {metric_name}: {score:.4f}")

        # Save model
        model_path = self.models_dir / "final" / "xgboost_model.json"
        self.model.save_model(str(model_path))

        return {
            "model_path": str(model_path),
            "best_score": score,
            "metric": metric_name,
            "model_type": "xgboost"
        }

    async def _train_pytorch_mlp(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train PyTorch MLP model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        logger.info("Training PyTorch MLP model...")

        # Preprocess data
        X_processed = self._preprocess_tabular_data(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)

        # Define model
        input_size = X_train.shape[1]
        hidden_sizes = config.get('hidden_sizes', [128, 64, 32])

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

        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

        epochs = config.get('epochs', 100)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")

        # Save model
        model_path = self.models_dir / "final" / "pytorch_mlp.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': input_size,
            'hidden_sizes': hidden_sizes
        }, model_path)

        self.model = model

        return {
            "model_path": str(model_path),
            "best_score": best_val_loss,
            "metric": "mse",
            "model_type": "pytorch_mlp"
        }

    async def _train_nlp_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train NLP model with transformer and QLoRA fine-tuning."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TrainingArguments,
            Trainer
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        logger.info("Training NLP model with transformers...")

        # Load data
        df = pd.read_csv(data_path)
        config = context.get("config", {})

        # Find text column
        text_column = context.get("text_column")
        if not text_column:
            # Auto-detect text column
            for col in df.columns:
                if col != target_column and df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 50:
                        text_column = col
                        break

        if not text_column:
            raise ValueError("Could not find text column in dataset")

        logger.info(f"Using text column: {text_column}")

        # Prepare data
        texts = df[text_column].astype(str).tolist()
        labels = df[target_column].tolist()

        # Initialize model and tokenizer
        model_name = config.get("model_name", "distilbert-base-uncased")
        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine number of labels
        num_labels = len(set(labels))

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Apply LoRA if requested
        if config.get("use_lora", False):
            logger.info("Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                target_modules=["q_lin", "v_lin"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, lora_config)

        # Tokenize data
        max_length = config.get("max_length", 512)
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Create dataset
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        dataset = TextDataset(encodings, labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.models_dir / "checkpoints"),
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 8),
            learning_rate=config.get("learning_rate", 2e-5),
            logging_dir=str(self.models_dir / "logs"),
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset
        )

        # Train
        trainer.train()

        # Save model
        model_path = self.models_dir / "final" / "nlp_model"
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        return {
            "model_path": str(model_path),
            "model_type": "transformer",
            "text_column": text_column
        }

    async def _train_vision_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train computer vision model."""
        logger.info("Vision model training not yet fully implemented")
        raise NotImplementedError("Vision model training coming soon")

    def _preprocess_tabular_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess tabular data."""
        X_processed = X.copy()

        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            # Simple label encoding
            X_processed[col] = pd.Categorical(X_processed[col]).codes

        # Fill missing values
        X_processed = X_processed.fillna(X_processed.mean())

        return X_processed

    def get_model(self):
        """Get the trained model."""
        return self.model