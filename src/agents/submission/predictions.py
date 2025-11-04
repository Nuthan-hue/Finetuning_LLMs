"""
Prediction Generation
Generates predictions for different model types.
"""
import logging
import pandas as pd
from typing import Dict, Any

from .preprocessing import preprocess_tabular_data, detect_text_column
from .utils import create_submission_dataframe

logger = logging.getLogger(__name__)


async def predict_lightgbm(
    model_path: str,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate predictions using LightGBM model."""
    import lightgbm as lgb

    logger.info("Loading LightGBM model...")
    model = lgb.Booster(model_file=model_path)

    # Preprocess test data (same as training)
    X_test = preprocess_tabular_data(test_df)

    # Generate predictions
    predictions = model.predict(X_test)

    # Create submission dataframe
    return create_submission_dataframe(predictions, test_df)


async def predict_xgboost(
    model_path: str,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate predictions using XGBoost model."""
    import xgboost as xgb

    logger.info("Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(model_path)

    # Preprocess test data
    X_test = preprocess_tabular_data(test_df)

    # Convert to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Generate predictions
    predictions = model.predict(dtest)

    # Create submission dataframe
    return create_submission_dataframe(predictions, test_df)


async def predict_pytorch_mlp(
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
    X_test = preprocess_tabular_data(test_df)
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()

    # Create submission dataframe
    return create_submission_dataframe(predictions, test_df)


async def predict_transformer(
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
        text_column = detect_text_column(test_df)

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
    return create_submission_dataframe(all_predictions, test_df)


async def generate_predictions(
    model_path: str,
    test_data_path: str,
    model_type: str,
    context: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate predictions using the trained model.

    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        model_type: Type of model (lightgbm, xgboost, etc.)
        context: Additional context

    Returns:
        Dataframe with predictions

    Raises:
        ValueError: If model type is unsupported
    """
    logger.info(f"Loading model from: {model_path}")

    # Load test data
    test_df = pd.read_csv(test_data_path)
    logger.info(f"Test data shape: {test_df.shape}")

    # Generate predictions based on model type
    if model_type == "lightgbm":
        return await predict_lightgbm(model_path, test_df)
    elif model_type == "xgboost":
        return await predict_xgboost(model_path, test_df)
    elif model_type == "pytorch_mlp":
        return await predict_pytorch_mlp(model_path, test_df, context)
    elif model_type == "transformer":
        return await predict_transformer(model_path, test_df, context)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")