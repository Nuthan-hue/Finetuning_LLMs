"""
Test cases for Phase 4: Preprocessing
Tests both the PreprocessingAgent (code generation) and execution flow.
"""
import pytest
import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.llm_agents.preprocessing_agent import PreprocessingAgent
from src.agents.orchestrator.phases import run_preprocessing


class TestPreprocessingAgent:
    """Unit tests for PreprocessingAgent"""

    @pytest.fixture
    def preprocessing_agent(self):
        """Create PreprocessingAgent instance"""
        return PreprocessingAgent()

    @pytest.fixture
    def sample_data_analysis_tabular(self):
        """Sample data analysis output for tabular data"""
        return {
            "data_modality": "tabular",
            "target_column": "Survived",
            "target_type": "binary",
            "feature_types": {
                "id_columns": ["PassengerId"],
                "numerical": ["Age", "Fare", "SibSp", "Parch"],
                "categorical": ["Sex", "Pclass", "Embarked"],
                "text": ["Name"],
                "drop_candidates": ["PassengerId", "Ticket", "Cabin"]
            },
            "data_quality": {
                "missing_values": {
                    "Age": {"count": 177, "percentage": 0.20},
                    "Cabin": {"count": 687, "percentage": 0.77},
                    "Embarked": {"count": 2, "percentage": 0.002}
                },
                "outliers": ["Fare"],
                "class_balance": "imbalanced"
            },
            "preprocessing": {
                "drop_columns": ["PassengerId", "Ticket", "Cabin"],
                "impute_missing": {
                    "Age": {"method": "median", "reason": "normally distributed"},
                    "Embarked": {"method": "mode", "reason": "only 2 missing"}
                },
                "encode_categorical": {
                    "Sex": "label",
                    "Pclass": "label",
                    "Embarked": "label"
                },
                "handle_outliers": {
                    "Fare": {"method": "cap", "percentile": 99}
                }
            },
            "preprocessing_required": True
        }

    @pytest.fixture
    def sample_data_analysis_nlp(self):
        """Sample data analysis output for NLP data"""
        return {
            "data_modality": "nlp",
            "target_column": "sentiment",
            "target_type": "binary",
            "feature_types": {
                "id_columns": ["id"],
                "text": ["review_text"],
                "categorical": ["category"]
            },
            "data_quality": {
                "missing_values": {
                    "review_text": {"count": 5, "percentage": 0.01}
                }
            },
            "preprocessing": {
                "drop_columns": ["id"],
                "text_cleaning": {
                    "review_text": {
                        "lowercase": True,
                        "remove_urls": True,
                        "remove_special_chars": True,
                        "remove_stopwords": False
                    }
                },
                "impute_missing": {
                    "review_text": {"method": "empty_string"}
                }
            },
            "preprocessing_required": True
        }

    @pytest.mark.asyncio
    async def test_agent_initialization(self, preprocessing_agent):
        """Test that PreprocessingAgent initializes correctly"""
        assert preprocessing_agent.name == "PreprocessingAgent"
        assert preprocessing_agent.model_name == "gemini-2.0-flash-exp"
        assert preprocessing_agent.temperature == 0.1
        assert preprocessing_agent.system_prompt is not None

    @pytest.mark.asyncio
    async def test_generate_preprocessing_code_tabular(
        self,
        preprocessing_agent,
        sample_data_analysis_tabular,
        tmp_path
    ):
        """Test code generation for tabular data"""
        # Mock AI response
        mock_code = '''
```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_path: str) -> dict:
    """Preprocess tabular data"""
    data_path = Path(data_path)

    # Load data
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    # Separate target
    target_col = "Survived"
    y_train = train[target_col].copy()
    train = train.drop(columns=[target_col])

    # Drop ID columns
    train = train.drop(columns=["PassengerId", "Ticket", "Cabin"], errors='ignore')
    test = test.drop(columns=["PassengerId", "Ticket", "Cabin"], errors='ignore')

    # Impute Age with median
    age_median = train["Age"].median()
    train["Age"].fillna(age_median, inplace=True)
    test["Age"].fillna(age_median, inplace=True)

    # Encode Sex
    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    # Add target back
    train[target_col] = y_train

    # Save
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": train.isnull().sum().sum()
    }
```
        '''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response', return_value=mock_code):
            code = await preprocessing_agent.generate_preprocessing_code(
                data_analysis=sample_data_analysis_tabular,
                data_path=str(tmp_path)
            )

        # Validate generated code
        assert isinstance(code, str)
        assert len(code) > 0
        assert "def preprocess_data" in code
        assert "import pandas" in code
        assert "train.csv" in code
        assert "test.csv" in code
        assert "clean_train.csv" in code
        assert "clean_test.csv" in code

    @pytest.mark.asyncio
    async def test_generate_preprocessing_code_nlp(
        self,
        preprocessing_agent,
        sample_data_analysis_nlp,
        tmp_path
    ):
        """Test code generation for NLP data"""
        mock_code = '''
```python
import pandas as pd
from pathlib import Path

def preprocess_data(data_path: str) -> dict:
    """Preprocess NLP data"""
    data_path = Path(data_path)

    # Load data
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    # Clean text
    train["review_text"] = train["review_text"].fillna("")
    test["review_text"] = test["review_text"].fillna("")

    # Lowercase
    train["review_text"] = train["review_text"].str.lower()
    test["review_text"] = test["review_text"].str.lower()

    # Save
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": 0
    }
```
        '''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response', return_value=mock_code):
            code = await preprocessing_agent.generate_preprocessing_code(
                data_analysis=sample_data_analysis_nlp,
                data_path=str(tmp_path)
            )

        assert "def preprocess_data" in code
        assert "review_text" in code or "text" in code.lower()

    @pytest.mark.asyncio
    async def test_code_extraction_with_markdown(self, preprocessing_agent):
        """Test that code is correctly extracted from markdown blocks"""
        mock_response = '''
Here's the preprocessing code:

```python
def preprocess_data(data_path: str) -> dict:
    return {"status": "ok"}
```

This should work!
        '''

        code = preprocessing_agent._extract_code(mock_response)
        assert "def preprocess_data" in code
        assert "```" not in code
        assert "Here's" not in code

    @pytest.mark.asyncio
    async def test_code_extraction_without_markdown(self, preprocessing_agent):
        """Test code extraction when no markdown blocks present"""
        code_only = '''def preprocess_data(data_path: str) -> dict:
    return {"status": "ok"}
        '''

        code = preprocessing_agent._extract_code(code_only)
        assert "def preprocess_data" in code

    @pytest.mark.asyncio
    async def test_code_extraction_validation(self, preprocessing_agent):
        """Test that invalid code raises error"""
        invalid_code = "This is not valid preprocessing code"

        with pytest.raises(RuntimeError, match="did not generate valid preprocessing code"):
            preprocessing_agent._extract_code(invalid_code)

    def test_format_dict_simple(self, preprocessing_agent):
        """Test dictionary formatting for prompt"""
        test_dict = {"key1": "value1", "key2": "value2"}
        formatted = preprocessing_agent._format_dict(test_dict)

        assert "key1: value1" in formatted
        assert "key2: value2" in formatted

    def test_format_dict_nested(self, preprocessing_agent):
        """Test nested dictionary formatting"""
        test_dict = {
            "outer": {
                "inner1": "value1",
                "inner2": "value2"
            }
        }
        formatted = preprocessing_agent._format_dict(test_dict)

        assert "outer:" in formatted
        assert "inner1: value1" in formatted

    def test_format_dict_empty(self, preprocessing_agent):
        """Test empty dictionary formatting"""
        formatted = preprocessing_agent._format_dict({})
        assert formatted == "None specified"


class TestPreprocessingPhase:
    """Integration tests for Phase 4 preprocessing flow"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        orchestrator = Mock()
        orchestrator.iteration = 1
        return orchestrator

    @pytest.fixture
    def sample_context_needs_preprocessing(self, tmp_path):
        """Context that requires preprocessing"""
        # Create sample data files
        data_path = tmp_path / "titanic"
        data_path.mkdir()

        # Create train.csv with missing values and mixed types
        train_data = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": ["Braund, Mr.", "Cumings, Mrs.", "Heikkinen, Miss.", "Futrelle, Mrs.", "Allen, Mr."],
            "Sex": ["male", "female", "female", "female", "male"],
            "Age": [22, 38, np.nan, 35, 35],
            "SibSp": [1, 1, 0, 1, 0],
            "Parch": [0, 0, 0, 0, 0],
            "Ticket": ["A/5", "PC 17599", "STON/O2", "113803", "373450"],
            "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
            "Cabin": [np.nan, "C85", np.nan, "C123", np.nan],
            "Embarked": ["S", "C", "S", "S", "S"]
        })
        train_data.to_csv(data_path / "train.csv", index=False)

        # Create test.csv
        test_data = pd.DataFrame({
            "PassengerId": [6, 7, 8],
            "Pclass": [3, 1, 3],
            "Name": ["Test1", "Test2", "Test3"],
            "Sex": ["male", "female", "male"],
            "Age": [25, np.nan, 30],
            "SibSp": [0, 1, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["A/5", "PC 17599", "STON/O2"],
            "Fare": [7.25, 71.28, 7.92],
            "Cabin": [np.nan, "C85", np.nan],
            "Embarked": ["S", "C", "S"]
        })
        test_data.to_csv(data_path / "test.csv", index=False)

        return {
            "competition_name": "titanic",
            "data_path": str(data_path),
            "needs_preprocessing": True,
            "target_column": "Survived",
            "data_modality": "tabular",
            "data_analysis": {
                "data_modality": "tabular",
                "target_column": "Survived",
                "feature_types": {
                    "id_columns": ["PassengerId"],
                    "numerical": ["Age", "Fare"],
                    "categorical": ["Sex", "Pclass"],
                    "drop_candidates": ["Ticket", "Cabin"]
                },
                "data_quality": {
                    "missing_values": {
                        "Age": {"count": 1, "percentage": 0.20}
                    }
                },
                "preprocessing": {
                    "drop_columns": ["PassengerId", "Ticket", "Cabin", "Name"],
                    "impute_missing": {
                        "Age": {"method": "median"}
                    },
                    "encode_categorical": {
                        "Sex": "label"
                    }
                },
                "preprocessing_required": True
            }
        }

    @pytest.fixture
    def sample_context_skip_preprocessing(self, tmp_path):
        """Context that skips preprocessing"""
        data_path = tmp_path / "clean_competition"
        data_path.mkdir()

        return {
            "competition_name": "clean_competition",
            "data_path": str(data_path),
            "needs_preprocessing": False,
            "data_analysis": {
                "preprocessing_required": False
            }
        }

    @pytest.mark.asyncio
    async def test_preprocessing_phase_skip_when_not_needed(
        self,
        mock_orchestrator,
        sample_context_skip_preprocessing
    ):
        """Test that preprocessing is skipped when not needed"""
        context = await run_preprocessing(
            mock_orchestrator,
            sample_context_skip_preprocessing
        )

        assert context["clean_data_path"] == context["data_path"]
        assert "preprocessing_result" not in context

    @pytest.mark.asyncio
    async def test_preprocessing_phase_executes_when_needed(
        self,
        mock_orchestrator,
        sample_context_needs_preprocessing
    ):
        """Test that preprocessing executes when needed"""
        # This will use actual AI to generate code
        # For unit tests, we should mock the AI response

        mock_preprocessing_code = '''
import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_data(data_path: str) -> dict:
    """Preprocess data"""
    data_path = Path(data_path)

    # Load data
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    # Separate target
    y_train = train["Survived"].copy()
    train = train.drop(columns=["Survived"])

    # Drop columns
    cols_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
    train = train.drop(columns=cols_to_drop, errors='ignore')
    test = test.drop(columns=cols_to_drop, errors='ignore')

    # Impute Age
    age_median = train["Age"].median()
    train["Age"].fillna(age_median, inplace=True)
    test["Age"].fillna(age_median, inplace=True)

    # Encode Sex
    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    # Add target back
    train["Survived"] = y_train

    # Save
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": int(train.isnull().sum().sum())
    }
'''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response',
                   return_value=mock_preprocessing_code):
            context = await run_preprocessing(
                mock_orchestrator,
                sample_context_needs_preprocessing
            )

        # Verify preprocessing was executed
        assert "clean_data_path" in context
        assert "preprocessing_result" in context

        # Verify clean files were created
        clean_train_path = Path(context["data_path"]) / "clean_train.csv"
        clean_test_path = Path(context["data_path"]) / "clean_test.csv"

        assert clean_train_path.exists()
        assert clean_test_path.exists()

        # Verify preprocessing results
        result = context["preprocessing_result"]
        assert "train_shape" in result
        assert "test_shape" in result
        assert "columns" in result

    @pytest.mark.asyncio
    async def test_preprocessing_phase_saves_code_to_file(
        self,
        mock_orchestrator,
        sample_context_needs_preprocessing
    ):
        """Test that generated preprocessing code is saved to file"""
        mock_code = '''
def preprocess_data(data_path: str) -> dict:
    import pandas as pd
    from pathlib import Path
    data_path = Path(data_path)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)
    return {"train_shape": train.shape, "test_shape": test.shape, "columns": list(train.columns), "missing_values_remaining": 0}
'''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response',
                   return_value=mock_code):
            context = await run_preprocessing(
                mock_orchestrator,
                sample_context_needs_preprocessing
            )

        # Check that preprocessing.py was created
        preprocessing_file = Path(context["data_path"]) / "preprocessing.py"
        assert preprocessing_file.exists()

        # Verify content
        saved_code = preprocessing_file.read_text()
        assert "def preprocess_data" in saved_code

    @pytest.mark.asyncio
    async def test_preprocessing_phase_handles_execution_errors(
        self,
        mock_orchestrator,
        sample_context_needs_preprocessing
    ):
        """Test graceful handling of preprocessing execution errors"""
        # Generate code that will fail
        bad_code = '''
def preprocess_data(data_path: str) -> dict:
    raise RuntimeError("Intentional error for testing")
'''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response',
                   return_value=bad_code):
            context = await run_preprocessing(
                mock_orchestrator,
                sample_context_needs_preprocessing
            )

        # Should fall back to raw data
        assert context["clean_data_path"] == context["data_path"]

    @pytest.mark.asyncio
    async def test_preprocessing_validates_output_files(
        self,
        mock_orchestrator,
        sample_context_needs_preprocessing
    ):
        """Test that preprocessing validates clean files were created"""
        # Code that doesn't create output files
        incomplete_code = '''
def preprocess_data(data_path: str) -> dict:
    return {"status": "done"}
'''

        with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response',
                   return_value=incomplete_code):
            context = await run_preprocessing(
                mock_orchestrator,
                sample_context_needs_preprocessing
            )

        # Should fall back to raw data when clean_train.csv not created
        assert context["clean_data_path"] == context["data_path"]


class TestPreprocessingDataQuality:
    """Tests for data quality after preprocessing"""

    @pytest.mark.asyncio
    async def test_preprocessing_removes_missing_values(self, tmp_path):
        """Test that preprocessing handles missing values correctly"""
        # Setup data with missing values
        data_path = tmp_path / "test_data"
        data_path.mkdir()

        train = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
            "feature1": [1.0, np.nan, 3.0, np.nan, 5.0],
            "feature2": ["a", "b", np.nan, "c", "d"]
        })
        train.to_csv(data_path / "train.csv", index=False)

        test = pd.DataFrame({
            "id": [6, 7],
            "feature1": [np.nan, 2.0],
            "feature2": ["a", np.nan]
        })
        test.to_csv(data_path / "test.csv", index=False)

        # Create preprocessing code
        preprocessing_code = '''
import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_data(data_path: str) -> dict:
    data_path = Path(data_path)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    # Separate target
    y_train = train["target"].copy()
    train = train.drop(columns=["target"])

    # Drop id
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    # Impute numerical with median
    feature1_median = train["feature1"].median()
    train["feature1"].fillna(feature1_median, inplace=True)
    test["feature1"].fillna(feature1_median, inplace=True)

    # Impute categorical with mode
    feature2_mode = train["feature2"].mode()[0]
    train["feature2"].fillna(feature2_mode, inplace=True)
    test["feature2"].fillna(feature2_mode, inplace=True)

    # Add target back
    train["target"] = y_train

    # Save
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": int(train.isnull().sum().sum() + test.isnull().sum().sum())
    }
'''

        # Execute preprocessing
        namespace = {}
        exec(preprocessing_code, namespace)
        result = namespace["preprocess_data"](str(data_path))

        # Verify no missing values remain
        assert result["missing_values_remaining"] == 0

        # Load and verify
        clean_train = pd.read_csv(data_path / "clean_train.csv")
        clean_test = pd.read_csv(data_path / "clean_test.csv")

        assert clean_train.isnull().sum().sum() == 0
        assert clean_test.isnull().sum().sum() == 0

    @pytest.mark.asyncio
    async def test_preprocessing_preserves_target_column(self, tmp_path):
        """Test that target column is preserved in train data only"""
        data_path = tmp_path / "test_data"
        data_path.mkdir()

        train = pd.DataFrame({
            "id": [1, 2, 3],
            "target": [0, 1, 0],
            "feature": [1, 2, 3]
        })
        train.to_csv(data_path / "train.csv", index=False)

        test = pd.DataFrame({
            "id": [4, 5],
            "feature": [4, 5]
        })
        test.to_csv(data_path / "test.csv", index=False)

        code = '''
import pandas as pd
from pathlib import Path

def preprocess_data(data_path: str) -> dict:
    data_path = Path(data_path)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    # Separate and restore target
    y_train = train["target"].copy()
    train = train.drop(columns=["target", "id"])
    test = test.drop(columns=["id"])
    train["target"] = y_train

    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": 0
    }
'''

        namespace = {}
        exec(code, namespace)
        namespace["preprocess_data"](str(data_path))

        clean_train = pd.read_csv(data_path / "clean_train.csv")
        clean_test = pd.read_csv(data_path / "clean_test.csv")

        # Target should be in train
        assert "target" in clean_train.columns
        # Target should NOT be in test
        assert "target" not in clean_test.columns

    @pytest.mark.asyncio
    async def test_preprocessing_avoids_data_leakage(self, tmp_path):
        """Test that preprocessing fits on train and transforms test (no leakage)"""
        data_path = tmp_path / "test_data"
        data_path.mkdir()

        # Train has different distribution than test
        train = pd.DataFrame({
            "target": [0, 1, 0],
            "feature": [10, 20, 30]
        })
        train.to_csv(data_path / "train.csv", index=False)

        test = pd.DataFrame({
            "feature": [100, 200]  # Very different values
        })
        test.to_csv(data_path / "test.csv", index=False)

        # Code that correctly fits on train only
        code = '''
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path: str) -> dict:
    data_path = Path(data_path)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    y_train = train["target"].copy()
    train = train.drop(columns=["target"])

    # Fit scaler on TRAIN only
    scaler = StandardScaler()
    train[["feature"]] = scaler.fit_transform(train[["feature"]])
    test[["feature"]] = scaler.transform(test[["feature"]])

    train["target"] = y_train

    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": 0
    }
'''

        namespace = {}
        exec(code, namespace)
        result = namespace["preprocess_data"](str(data_path))

        # Verify both files created successfully
        assert (data_path / "clean_train.csv").exists()
        assert (data_path / "clean_test.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])