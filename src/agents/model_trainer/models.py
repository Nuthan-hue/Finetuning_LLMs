"""
Model training implementations for different ML frameworks.
Supports LightGBM, XGBoost, PyTorch MLP, and Transformers.
"""
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


async def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    models_dir: Path
) -> Dict[str, Any]:
    """
    Train LightGBM model.

    Args:
        X: Training features
        y: Training target
        config: Training configuration
        models_dir: Directory to save models

    Returns:
        Dictionary with model path, score, and metrics
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error

    from .preprocessing import preprocess_tabular_data, is_classification_task

    logger.info("Training LightGBM model...")

    # Preprocess data
    X_processed = preprocess_tabular_data(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Determine task (classification or regression)
    is_classification = is_classification_task(y)

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
    model = lgb.train(
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
    y_pred = model.predict(X_val)

    if is_classification:
        y_pred_binary = (y_pred > 0.5).astype(int)
        score = accuracy_score(y_val, y_pred_binary)
        metric_name = "accuracy"
    else:
        score = mean_squared_error(y_val, y_pred, squared=False)
        metric_name = "rmse"

    logger.info(f"Validation {metric_name}: {score:.4f}")

    # Save model
    model_path = models_dir / "final" / "lightgbm_model.txt"
    model.save_model(str(model_path))

    return {
        "model": model,
        "model_path": str(model_path),
        "best_score": score,
        "metric": metric_name,
        "model_type": "lightgbm"
    }


async def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    models_dir: Path
) -> Dict[str, Any]:
    """
    Train XGBoost model.

    Args:
        X: Training features
        y: Training target
        config: Training configuration
        models_dir: Directory to save models

    Returns:
        Dictionary with model path, score, and metrics
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error

    from .preprocessing import preprocess_tabular_data, is_classification_task

    logger.info("Training XGBoost model...")

    # Preprocess data
    X_processed = preprocess_tabular_data(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Determine task
    is_classification = is_classification_task(y)

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
    model = xgb.XGBClassifier(**params) if is_classification else xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Evaluate
    y_pred = model.predict(X_val)

    if is_classification:
        score = accuracy_score(y_val, y_pred)
        metric_name = "accuracy"
    else:
        score = mean_squared_error(y_val, y_pred, squared=False)
        metric_name = "rmse"

    logger.info(f"Validation {metric_name}: {score:.4f}")

    # Save model
    model_path = models_dir / "final" / "xgboost_model.json"
    model.save_model(str(model_path))

    return {
        "model": model,
        "model_path": str(model_path),
        "best_score": score,
        "metric": metric_name,
        "model_type": "xgboost"
    }


async def train_pytorch_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    models_dir: Path
) -> Dict[str, Any]:
    """
    Train PyTorch MLP model.

    Args:
        X: Training features
        y: Training target
        config: Training configuration
        models_dir: Directory to save models

    Returns:
        Dictionary with model path, score, and metrics
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    from .preprocessing import preprocess_tabular_data, scale_features

    logger.info("Training PyTorch MLP model...")

    # Preprocess data
    X_processed = preprocess_tabular_data(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed.values, y.values, test_size=0.2, random_state=42
    )

    # Scale features
    scaler, X_train_scaled, X_val_scaled = scale_features(X_train, X_val)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)

    # Define model
    input_size = X_train_scaled.shape[1]
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
    model_path = models_dir / "final" / "pytorch_mlp.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_size': input_size,
        'hidden_sizes': hidden_sizes
    }, model_path)

    return {
        "model": model,
        "model_path": str(model_path),
        "best_score": best_val_loss,
        "metric": "mse",
        "model_type": "pytorch_mlp"
    }


async def train_nlp_model(
    data_path: str,
    target_column: str,
    context: Dict[str, Any],
    models_dir: Path
) -> Dict[str, Any]:
    """
    Train NLP model with transformer and QLoRA fine-tuning.

    Args:
        data_path: Path to training data
        target_column: Name of target column
        context: Context dictionary with configuration
        models_dir: Directory to save models

    Returns:
        Dictionary with model path and metadata
    """
    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer
    )
    from peft import LoraConfig, get_peft_model

    from .preprocessing import detect_text_column, prepare_text_data

    logger.info("Training NLP model with transformers...")

    # Load data
    df = pd.read_csv(data_path)
    config = context.get("config", {})

    # Find text column
    text_column = context.get("text_column")
    if not text_column:
        text_column = detect_text_column(df, target_column)

    logger.info(f"Using text column: {text_column}")

    # Prepare data
    texts, labels = prepare_text_data(df, text_column, target_column)

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
        output_dir=str(models_dir / "checkpoints"),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        logging_dir=str(models_dir / "logs"),
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
    model_path = models_dir / "final" / "nlp_model"
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    return {
        "model_path": str(model_path),
        "model_type": "transformer",
        "text_column": text_column
    }