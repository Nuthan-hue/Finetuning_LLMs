"""
Integration Test for Phase 4 Preprocessing
Run this to test the full preprocessing pipeline with real AI code generation.

Usage:
    python tests/test_preprocessing_integration.py
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)
# Also add parent for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_agents.preprocessing_agent import PreprocessingAgent
from src.agents.orchestrator.phases import run_preprocessing


def create_sample_titanic_data(data_dir: Path):
    """Create sample Titanic-like data for testing"""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create train.csv with typical issues
    train_data = pd.DataFrame({
        "PassengerId": range(1, 11),
        "Survived": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry",
            "Moran, Mr. James",
            "McCarthy, Mr. Timothy J",
            "Palsson, Master. Gosta Leonard",
            "Johnson, Mrs. Oscar W",
            "Nasser, Mrs. Nicholas"
        ],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
        "Age": [22, 38, 26, 35, 35, np.nan, 54, 2, 27, np.nan],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450",
                   "330877", "17463", "349909", "347742", "237736"],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
        "Cabin": [np.nan, "C85", np.nan, "C123", np.nan, np.nan, "E46", np.nan, np.nan, np.nan],
        "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"]
    })
    train_data.to_csv(data_dir / "train.csv", index=False)

    # Create test.csv
    test_data = pd.DataFrame({
        "PassengerId": range(11, 16),
        "Pclass": [3, 1, 2, 3, 3],
        "Name": ["Test1", "Test2", "Test3", "Test4", "Test5"],
        "Sex": ["male", "female", "male", "female", "male"],
        "Age": [25, np.nan, 30, 40, np.nan],
        "SibSp": [0, 1, 0, 2, 1],
        "Parch": [0, 0, 0, 1, 0],
        "Ticket": ["A123", "B456", "C789", "D012", "E345"],
        "Fare": [7.75, 80.0, 15.5, 25.0, 8.0],
        "Cabin": [np.nan, "C100", np.nan, np.nan, np.nan],
        "Embarked": ["S", "C", "S", "Q", "S"]
    })
    test_data.to_csv(data_dir / "test.csv", index=False)

    print(f"✅ Created sample data in {data_dir}")
    print(f"   - train.csv: {train_data.shape}")
    print(f"   - test.csv: {test_data.shape}")
    print(f"   - Missing values in Age: {train_data['Age'].isnull().sum()}")

    return train_data, test_data


async def test_preprocessing_agent_alone():
    """Test 1: PreprocessingAgent code generation only"""
    print("\n" + "=" * 70)
    print("TEST 1: PreprocessingAgent Code Generation")
    print("=" * 70)

    # Setup
    data_dir = Path("data/test_preprocessing")
    train_data, test_data = create_sample_titanic_data(data_dir)

    # Simulate data analysis output
    data_analysis = {
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
                "Age": {"count": 2, "percentage": 0.20},
                "Cabin": {"count": 8, "percentage": 0.80}
            },
            "outliers": ["Fare"]
        },
        "preprocessing": {
            "drop_columns": ["PassengerId", "Name", "Ticket", "Cabin"],
            "impute_missing": {
                "Age": {"method": "median", "reason": "normally distributed"}
            },
            "encode_categorical": {
                "Sex": "label",
                "Pclass": "label",
                "Embarked": "label"
            },
            "handle_outliers": {
                "Fare": {"method": "cap", "percentile": 99}
            }
        }
    }

    # Generate code
    agent = PreprocessingAgent()
    print("\n🤖 Generating preprocessing code with AI...")

    try:
        code = await agent.generate_preprocessing_code(
            data_analysis=data_analysis,
            data_path=str(data_dir)
        )

        print(f"\n✅ Code generated successfully ({len(code)} chars)")
        print("\n" + "-" * 70)
        print("Generated Code Preview (first 500 chars):")
        print("-" * 70)
        print(code[:500] + "...")

        # Save code
        code_file = data_dir / "generated_preprocessing.py"
        code_file.write_text(code)
        print(f"\n💾 Saved code to: {code_file}")

        return code, data_dir

    except Exception as e:
        print(f"\n❌ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_code_execution(code: str, data_dir: Path):
    """Test 2: Execute generated preprocessing code"""
    if not code or not data_dir:
        print("\n⏭️  Skipping execution test (no code generated)")
        return False

    print("\n" + "=" * 70)
    print("TEST 2: Executing Generated Preprocessing Code")
    print("=" * 70)

    try:
        # Execute code
        namespace = {}
        exec(code, namespace)

        if "preprocess_data" not in namespace:
            print("❌ Generated code missing 'preprocess_data' function")
            return False

        # Run preprocessing
        print("\n⚙️  Running preprocessing function...")
        preprocess_func = namespace["preprocess_data"]
        result = preprocess_func(str(data_dir))

        print("\n✅ Preprocessing executed successfully!")
        print(f"\nResults:")
        print(f"  - Train shape: {result.get('train_shape')}")
        print(f"  - Test shape: {result.get('test_shape')}")
        print(f"  - Columns: {result.get('columns')}")
        print(f"  - Missing values remaining: {result.get('missing_values_remaining', 'N/A')}")

        # Verify files
        clean_train = data_dir / "clean_train.csv"
        clean_test = data_dir / "clean_test.csv"

        if not clean_train.exists():
            print("\n❌ clean_train.csv not created")
            return False

        if not clean_test.exists():
            print("\n❌ clean_test.csv not created")
            return False

        print("\n✅ Output files created successfully")

        # Load and inspect
        df_train = pd.read_csv(clean_train)
        df_test = pd.read_csv(clean_test)

        print(f"\n📊 Clean Data Inspection:")
        print(f"  Train shape: {df_train.shape}")
        print(f"  Test shape: {df_test.shape}")
        print(f"  Train columns: {list(df_train.columns)}")
        print(f"  Missing in train: {df_train.isnull().sum().sum()}")
        print(f"  Missing in test: {df_test.isnull().sum().sum()}")

        # Check target preserved
        if "Survived" in df_train.columns:
            print(f"  ✅ Target column preserved in train")
        else:
            print(f"  ❌ Target column missing from train")

        if "Survived" not in df_test.columns:
            print(f"  ✅ Target column correctly absent from test")
        else:
            print(f"  ⚠️  Warning: Target column found in test")

        return True

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_preprocessing_phase():
    """Test 3: Full Phase 4 integration with orchestrator"""
    print("\n" + "=" * 70)
    print("TEST 3: Full Preprocessing Phase (with Orchestrator)")
    print("=" * 70)

    # Setup
    data_dir = Path("data/test_preprocessing_phase")
    train_data, test_data = create_sample_titanic_data(data_dir)

    # Create context (as would be passed from previous phases)
    context = {
        "competition_name": "titanic",
        "data_path": str(data_dir),
        "needs_preprocessing": True,
        "target_column": "Survived",
        "data_modality": "tabular",
        "data_analysis": {
            "data_modality": "tabular",
            "target_column": "Survived",
            "feature_types": {
                "id_columns": ["PassengerId"],
                "numerical": ["Age", "Fare", "SibSp", "Parch"],
                "categorical": ["Sex", "Pclass", "Embarked"],
                "drop_candidates": ["PassengerId", "Ticket", "Cabin", "Name"]
            },
            "data_quality": {
                "missing_values": {
                    "Age": {"count": 2, "percentage": 0.20}
                }
            },
            "preprocessing": {
                "drop_columns": ["PassengerId", "Ticket", "Cabin", "Name"],
                "impute_missing": {
                    "Age": {"method": "median"}
                },
                "encode_categorical": {
                    "Sex": "label",
                    "Pclass": "label",
                    "Embarked": "label"
                }
            },
            "preprocessing_required": True
        }
    }

    # Mock orchestrator
    class MockOrchestrator:
        def __init__(self):
            self.iteration = 1

    orchestrator = MockOrchestrator()

    try:
        print("\n🚀 Running Phase 4: Preprocessing...")
        updated_context = await run_preprocessing(orchestrator, context)

        print("\n✅ Phase 4 completed!")
        print(f"\nUpdated Context:")
        print(f"  - clean_data_path: {updated_context.get('clean_data_path')}")
        print(f"  - preprocessing_result: {updated_context.get('preprocessing_result')}")

        # Verify
        if "preprocessing_result" in updated_context:
            result = updated_context["preprocessing_result"]
            print(f"\n📊 Preprocessing Results:")
            print(f"  - Train shape: {result.get('train_shape')}")
            print(f"  - Test shape: {result.get('test_shape')}")
            print(f"  - Columns: {len(result.get('columns', []))}")
            print(f"  - Missing values: {result.get('missing_values_remaining')}")

        # Check saved code
        code_file = Path(updated_context["data_path"]) / "preprocessing.py"
        if code_file.exists():
            print(f"\n✅ Preprocessing code saved to: {code_file}")
        else:
            print(f"\n⚠️  Preprocessing code not saved")

        return True

    except Exception as e:
        print(f"\n❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_skip_preprocessing():
    """Test 4: Verify preprocessing is skipped when not needed"""
    print("\n" + "=" * 70)
    print("TEST 4: Skip Preprocessing When Not Needed")
    print("=" * 70)

    data_dir = Path("data/test_skip_preprocessing")
    data_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "competition_name": "clean_competition",
        "data_path": str(data_dir),
        "needs_preprocessing": False,
        "data_analysis": {
            "preprocessing_required": False
        }
    }

    class MockOrchestrator:
        iteration = 1

    orchestrator = MockOrchestrator()

    try:
        updated_context = await run_preprocessing(orchestrator, context)

        print("\n✅ Preprocessing correctly skipped")
        print(f"  - clean_data_path: {updated_context.get('clean_data_path')}")
        print(f"  - Uses original path: {updated_context['clean_data_path'] == updated_context['data_path']}")

        return updated_context['clean_data_path'] == updated_context['data_path']

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("PHASE 4 PREPROCESSING - INTEGRATION TESTS")
    print("=" * 70)

    results = []

    # Test 1: Code generation
    code, data_dir = await test_preprocessing_agent_alone()
    results.append(("Code Generation", code is not None))

    # Test 2: Code execution
    if code and data_dir:
        success = await test_code_execution(code, data_dir)
        results.append(("Code Execution", success))

    # Test 3: Full phase
    success = await test_full_preprocessing_phase()
    results.append(("Full Phase Integration", success))

    # Test 4: Skip logic
    success = await test_skip_preprocessing()
    results.append(("Skip Preprocessing Logic", success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)