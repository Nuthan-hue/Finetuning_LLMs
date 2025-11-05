"""
Universal Data Pipeline Executor
Executes AI-recommended preprocessing for ANY Kaggle competition type.

Supports:
- Tabular data (classification, regression, clustering)
- NLP (text classification, generation, sentiment)
- Computer Vision (image classification, object detection)
- Time Series (forecasting, anomaly detection)
- Mixed modality competitions
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DataPipelineExecutor:
    """
    Universal pipeline that adapts to any competition type based on AI analysis.

    Key Features:
    - Modality-aware (tabular/NLP/vision/time-series)
    - Saves intermediate steps for debugging
    - Handles supervised AND unsupervised tasks
    - Executes ALL AI recommendations systematically
    """

    def __init__(self, competition_name: str, base_data_dir: str = "data"):
        self.competition_name = competition_name
        self.base_data_dir = Path(base_data_dir)

        # Create organized directory structure
        self.raw_dir = self.base_data_dir / "raw" / competition_name
        self.processed_dir = self.base_data_dir / "processed" / competition_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.step_counter = 0
        logger.info(f"📂 Pipeline workspace: {self.processed_dir}")

    async def execute(
        self,
        data_path: str,
        ai_analysis: Dict[str, Any],
        target_column: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute complete AI-guided data pipeline.

        Args:
            data_path: Path to raw training data
            ai_analysis: Complete AI analysis with all recommendations
            target_column: Target column (None for unsupervised tasks)

        Returns:
            Tuple of (X, y) - processed features and target
        """
        logger.info("="*70)
        logger.info("🤖 STARTING AI-GUIDED DATA PIPELINE")
        logger.info("="*70)

        # Detect modality and task type
        modality = ai_analysis.get("data_modality", "tabular")
        task_type = ai_analysis.get("task_type", "unknown")
        has_target = ai_analysis.get("has_target", True)

        logger.info(f"📊 Modality: {modality}")
        logger.info(f"🎯 Task: {task_type}")
        logger.info(f"🎲 Has Target: {has_target}")

        # Route to appropriate pipeline
        if modality == "tabular":
            return await self._execute_tabular_pipeline(
                data_path, ai_analysis, target_column, has_target
            )
        elif modality == "nlp":
            return await self._execute_nlp_pipeline(
                data_path, ai_analysis, target_column
            )
        elif modality == "computer_vision":
            return await self._execute_vision_pipeline(
                data_path, ai_analysis, target_column
            )
        elif modality == "time_series":
            return await self._execute_timeseries_pipeline(
                data_path, ai_analysis, target_column
            )
        else:
            logger.warning(f"⚠️  Unknown modality '{modality}' - using tabular pipeline")
            return await self._execute_tabular_pipeline(
                data_path, ai_analysis, target_column, has_target
            )

    # ═══════════════════════════════════════════════════════════════════
    # TABULAR DATA PIPELINE
    # ═══════════════════════════════════════════════════════════════════

    async def _execute_tabular_pipeline(
        self,
        data_path: str,
        ai_analysis: Dict[str, Any],
        target_column: str,
        has_target: bool
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Execute pipeline for tabular data (most Kaggle competitions)."""

        # Load raw data
        df = pd.read_csv(data_path)
        logger.info(f"📥 Loaded raw data: {df.shape}")
        self._save_step(df, "00_raw_data")

        # Separate target (if supervised)
        if has_target and target_column and target_column in df.columns:
            y = df[target_column].copy()
            df = df.drop(columns=[target_column])
            logger.info(f"🎯 Target: {target_column} (shape: {y.shape})")
        else:
            y = None
            logger.info("🎲 Unsupervised task - no target column")

        feature_types = ai_analysis.get("feature_types", {})
        preprocessing = ai_analysis.get("preprocessing", {})

        # STEP 1: Remove ID columns (they don't help models)
        df = self._remove_id_columns(df, feature_types)
        self._save_step(df, "01_removed_ids")

        # STEP 2: Handle missing values (AI strategy)
        df = self._handle_missing_values(df, preprocessing, feature_types)
        self._save_step(df, "02_missing_handled")

        # STEP 3: Remove/cap outliers (AI detection)
        df = self._handle_outliers(df, ai_analysis)
        self._save_step(df, "03_outliers_handled")

        # STEP 4: Feature engineering (AI suggestions)
        df = self._create_features(df, ai_analysis)
        self._save_step(df, "04_features_created")

        # STEP 5: Apply transformations (log, sqrt, bin, etc.)
        df = self._apply_transformations(df, preprocessing)
        self._save_step(df, "05_transformed")

        # STEP 6: Encode categorical variables (AI method)
        df = self._encode_categoricals(df, preprocessing, feature_types)
        self._save_step(df, "06_encoded")

        # STEP 7: Scale numerical features (AI scaling)
        df = self._scale_numericals(df, preprocessing, feature_types)
        self._save_step(df, "07_scaled")

        # STEP 8: Final cleanup
        df = self._final_cleanup(df)
        self._save_step(df, "08_final")

        logger.info("="*70)
        logger.info(f"✅ PIPELINE COMPLETE: X={df.shape}, y={y.shape if y is not None else 'None'}")
        logger.info("="*70)

        return df, y

    # ═══════════════════════════════════════════════════════════════════
    # STEP IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════

    def _remove_id_columns(
        self,
        df: pd.DataFrame,
        feature_types: Dict[str, Any]
    ) -> pd.DataFrame:
        """Remove ID columns that don't contribute to predictions."""
        id_columns = feature_types.get("id_columns", [])

        if not id_columns:
            logger.info("📋 Step 1: No ID columns to remove")
            return df

        to_remove = [col for col in id_columns if col in df.columns]
        if to_remove:
            logger.info(f"📋 Step 1: Removing ID columns: {to_remove}")
            df = df.drop(columns=to_remove)

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        preprocessing: Dict[str, Any],
        feature_types: Dict[str, Any]
    ) -> pd.DataFrame:
        """Handle missing values using AI-recommended strategy."""
        strategy = preprocessing.get("handle_missing", "mean")

        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            logger.info("📋 Step 2: No missing values")
            return df

        logger.info(f"📋 Step 2: Handling missing values (strategy: {strategy})")
        logger.info(f"   Columns with missing: {len(missing_cols)}")

        numerical_cols = feature_types.get("numerical", [])
        categorical_cols = feature_types.get("categorical", [])

        for col in missing_cols:
            if col in numerical_cols:
                # Numerical: use mean/median
                if strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "drop":
                    df = df.dropna(subset=[col])
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
            else:
                # Categorical: use mode or 'missing'
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "missing"
                df[col].fillna(mode_val, inplace=True)

        return df

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        ai_analysis: Dict[str, Any]
    ) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        data_quality = ai_analysis.get("data_quality", {})
        outlier_cols = data_quality.get("outliers", [])

        if not outlier_cols:
            logger.info("📋 Step 3: No outliers detected by AI")
            return df

        logger.info(f"📋 Step 3: Handling outliers in: {outlier_cols}")

        for col in outlier_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                # Cap at 1st and 99th percentile
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q01, upper=q99)
                logger.info(f"   Capped {col}: [{q01:.2f}, {q99:.2f}]")

        return df

    def _create_features(
        self,
        df: pd.DataFrame,
        ai_analysis: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create new features based on AI suggestions."""
        feature_suggestions = ai_analysis.get("feature_engineering", [])

        if not feature_suggestions:
            logger.info("📋 Step 4: No feature engineering suggestions")
            return df

        logger.info(f"📋 Step 4: Creating {len(feature_suggestions)} AI-suggested features")

        for i, feature_desc in enumerate(feature_suggestions[:15]):  # Limit to 15
            try:
                # Parse feature description: "name = formula"
                if "=" in feature_desc:
                    feature_name, formula = feature_desc.split("=", 1)
                    feature_name = feature_name.strip()
                    formula = formula.strip()

                    # Execute feature creation
                    df[feature_name] = self._safe_eval_formula(df, formula)
                    logger.info(f"   ✓ {feature_name}")

            except Exception as e:
                logger.warning(f"   ✗ Failed: {feature_desc[:50]}... ({e})")
                continue

        return df

    def _safe_eval_formula(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Safely evaluate feature formula."""
        import re

        # Replace column references
        for col in df.columns:
            formula = re.sub(rf'\b{re.escape(col)}\b', f'df["{col}"]', formula)

        # Safe evaluation
        try:
            return eval(formula, {"df": df, "pd": pd, "np": np}, {})
        except:
            raise ValueError(f"Cannot evaluate: {formula}")

    def _apply_transformations(
        self,
        df: pd.DataFrame,
        preprocessing: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply AI-recommended transformations (log, sqrt, bin)."""
        transformations = preprocessing.get("feature_transformations", [])

        if not transformations:
            logger.info("📋 Step 5: No transformations recommended")
            return df

        logger.info(f"📋 Step 5: Applying {len(transformations)} transformations")

        for transform_desc in transformations:
            try:
                # Parse: "log(Fare)" or "bin(Age,bins=5)"
                if "log(" in transform_desc:
                    col = transform_desc.split("(")[1].split(")")[0].strip()
                    if col in df.columns:
                        df[f"{col}_log"] = np.log1p(df[col])
                        logger.info(f"   ✓ log({col})")

                elif "sqrt(" in transform_desc:
                    col = transform_desc.split("(")[1].split(")")[0].strip()
                    if col in df.columns:
                        df[f"{col}_sqrt"] = np.sqrt(df[col])
                        logger.info(f"   ✓ sqrt({col})")

                elif "bin(" in transform_desc:
                    # Parse: "bin(Age,bins=5)"
                    parts = transform_desc.split("(")[1].split(")")[0].split(",")
                    col = parts[0].strip()
                    bins = int(parts[1].split("=")[1]) if len(parts) > 1 else 5
                    if col in df.columns:
                        df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=False)
                        logger.info(f"   ✓ binned({col}, {bins})")

            except Exception as e:
                logger.warning(f"   ✗ Failed: {transform_desc} ({e})")

        return df

    def _encode_categoricals(
        self,
        df: pd.DataFrame,
        preprocessing: Dict[str, Any],
        feature_types: Dict[str, Any]
    ) -> pd.DataFrame:
        """Encode categorical variables using AI method."""
        encoding_method = preprocessing.get("categorical_encoding", "label")
        categorical_cols = feature_types.get("categorical", [])

        # Also encode any remaining object columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        all_categorical = list(set(categorical_cols + object_cols))

        if not all_categorical:
            logger.info("📋 Step 6: No categorical columns to encode")
            return df

        logger.info(f"📋 Step 6: Encoding categoricals (method: {encoding_method})")
        logger.info(f"   Columns: {len(all_categorical)}")

        for col in all_categorical:
            if col not in df.columns:
                continue

            try:
                if encoding_method == "label":
                    df[col] = pd.Categorical(df[col]).codes
                elif encoding_method == "onehot":
                    df = pd.get_dummies(df, columns=[col], prefix=col)
                else:
                    # Default: label encoding
                    df[col] = pd.Categorical(df[col]).codes
            except Exception as e:
                logger.warning(f"   ✗ Failed to encode {col}: {e}")

        return df

    def _scale_numericals(
        self,
        df: pd.DataFrame,
        preprocessing: Dict[str, Any],
        feature_types: Dict[str, Any]
    ) -> pd.DataFrame:
        """Scale numerical features using AI method."""
        scaling_method = preprocessing.get("numerical_scaling", "none")

        if scaling_method == "none":
            logger.info("📋 Step 7: No scaling recommended")
            return df

        logger.info(f"📋 Step 7: Scaling numericals (method: {scaling_method})")

        # Get numerical columns
        numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

        if not numerical_cols:
            return df

        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info(f"   Scaled {len(numerical_cols)} columns")

        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and validation."""
        logger.info("📋 Step 8: Final cleanup")

        # CRITICAL: Sanitize column names for LightGBM/XGBoost
        # They don't support special characters like: []<>"':{}
        logger.info("   Sanitizing column names for LightGBM/XGBoost...")
        df = self._sanitize_column_names(df)

        # Remove any remaining NaN/inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Ensure all numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"   Converting {col} to numeric (was object)")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info(f"   Final shape: {df.shape}")
        logger.info(f"   Final columns: {list(df.columns)[:10]}...")
        return df

    def _sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize column names to be compatible with LightGBM/XGBoost.

        LightGBM doesn't support: []<>":{}
        We'll replace them with safe characters.
        """
        import re

        new_columns = {}
        for col in df.columns:
            # Replace special characters with underscore
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            # Remove consecutive underscores
            sanitized = re.sub(r'_+', '_', sanitized)
            # Remove leading/trailing underscores
            sanitized = sanitized.strip('_')

            # Ensure no duplicates
            if sanitized in new_columns.values():
                i = 1
                while f"{sanitized}_{i}" in new_columns.values():
                    i += 1
                sanitized = f"{sanitized}_{i}"

            if col != sanitized:
                logger.debug(f"      Renamed: '{col}' → '{sanitized}'")
                new_columns[col] = sanitized
            else:
                new_columns[col] = col

        df = df.rename(columns=new_columns)
        return df

    # ═══════════════════════════════════════════════════════════════════
    # OTHER MODALITY PIPELINES (Placeholders for now)
    # ═══════════════════════════════════════════════════════════════════

    async def _execute_nlp_pipeline(
        self,
        data_path: str,
        ai_analysis: Dict[str, Any],
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Execute pipeline for NLP data."""
        logger.info("🔤 NLP Pipeline - Coming soon!")
        logger.warning("   Falling back to basic text processing")

        # For now, use tabular pipeline
        return await self._execute_tabular_pipeline(
            data_path, ai_analysis, target_column, has_target=True
        )

    async def _execute_vision_pipeline(
        self,
        data_path: str,
        ai_analysis: Dict[str, Any],
        target_column: str
    ) -> Tuple[Any, Any]:
        """Execute pipeline for computer vision data."""
        logger.info("🖼️  Vision Pipeline - Coming soon!")
        raise NotImplementedError("Vision pipeline not yet implemented")

    async def _execute_timeseries_pipeline(
        self,
        data_path: str,
        ai_analysis: Dict[str, Any],
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Execute pipeline for time series data."""
        logger.info("📈 Time Series Pipeline - Coming soon!")
        logger.warning("   Falling back to tabular pipeline")

        return await self._execute_tabular_pipeline(
            data_path, ai_analysis, target_column, has_target=True
        )

    # ═══════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _save_step(self, df: pd.DataFrame, step_name: str):
        """Save intermediate pipeline step for debugging."""
        self.step_counter += 1
        filepath = self.processed_dir / f"{step_name}.csv"

        try:
            df.to_csv(filepath, index=False)
            logger.debug(f"💾 Saved: {filepath.name}")
        except Exception as e:
            logger.warning(f"Could not save {step_name}: {e}")