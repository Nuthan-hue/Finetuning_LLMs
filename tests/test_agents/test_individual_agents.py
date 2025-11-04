"""
Test script for individual agents
"""
import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.data_collector import DataCollectorAgent
from agents.model_trainer import ModelTrainerAgent
from agents.submission import SubmissionAgent
from agents.leaderboard import LeaderboardMonitorAgent


async def test_data_collector():
    """Test DataCollectorAgent on Titanic competition"""
    print("=" * 60)
    print("TEST 1: DataCollectorAgent")
    print("=" * 60)

    agent = DataCollectorAgent(data_dir="data")

    context = {
        "competition_name": "titanic",
        "analyze": True
    }

    results = await agent.run(context)

    print("\n✓ DataCollectorAgent Results:")
    print(f"  - Data Path: {results['data_path']}")
    print(f"  - Files Found: {results['analysis_report']['files']}")

    for filename, info in results['analysis_report']['datasets'].items():
        print(f"\n  Dataset: {filename}")
        print(f"    Shape: {info['shape']}")
        print(f"    Columns: {len(info['columns'])}")
        print(f"    Memory: {info['memory_usage_mb']:.2f} MB")

    return results


async def test_model_trainer(data_results):
    """Test ModelTrainerAgent with collected data"""
    print("\n" + "=" * 60)
    print("TEST 2: ModelTrainerAgent")
    print("=" * 60)

    agent = ModelTrainerAgent()

    # Get train data path
    data_path = Path(data_results['data_path'])
    train_file = data_path / "train.csv"

    context = {
        "competition_name": "titanic",
        "data_path": str(train_file),
        "model_type": "lightgbm",  # or "xgboost", "pytorch_mlp"
        "target_column": "Survived"
    }

    results = await agent.run(context)

    print("\n✓ ModelTrainerAgent Results:")
    print(f"  - Model Path: {results['model_path']}")
    print(f"  - Task Type: {results.get('task_type', 'N/A')}")
    print(f"  - Best Score: {results.get('best_score', 'N/A')}")
    print(f"  - Model Type: {results.get('model_type', 'N/A')}")

    return results


async def test_submission(model_results, data_results):
    """Test SubmissionAgent to generate predictions"""
    print("\n" + "=" * 60)
    print("TEST 3: SubmissionAgent")
    print("=" * 60)

    agent = SubmissionAgent()

    data_path = Path(data_results['data_path'])
    test_file = data_path / "test.csv"

    context = {
        "competition_name": "titanic",
        "model_path": model_results['model_path'],
        "test_data_path": str(test_file),
        "model_type": model_results['model_type'],
        "auto_submit": True,  # Set to False to skip actual submission
        "submission_message": "Test submission from autonomous agent",
        "format_spec": {
            "column_mapping": {
                "id": "PassengerId",
                "prediction": "Survived"
            },
            "transformations": {
                "Survived": "binary"  # Convert probabilities to 0/1
            }
        }
    }

    results = await agent.run(context)

    print("\n✓ SubmissionAgent Results:")
    print(f"  - Submission File: {results['submission_path']}")
    if results.get('submission_status'):
        print(f"  - Submission Status: {results['submission_status']}")

    return results


async def test_leaderboard(submission_results):
    """Test LeaderboardMonitorAgent"""
    print("\n" + "=" * 60)
    print("TEST 4: LeaderboardMonitorAgent")
    print("=" * 60)

    agent = LeaderboardMonitorAgent(target_percentile=0.20)

    context = {
        "competition_name": "titanic",
        "submission_file": submission_results.get('submission_path')
    }

    results = await agent.run(context)

    print("\n✓ LeaderboardMonitorAgent Results:")
    print(f"  - Current Rank: {results.get('current_rank', 'N/A')}")

    current_pct = results.get('current_percentile', 'N/A')
    if isinstance(current_pct, (int, float)):
        print(f"  - Current Percentile: {current_pct:.2%}")
    else:
        print(f"  - Current Percentile: {current_pct}")

    target_pct = results.get('target_percentile', 'N/A')
    if isinstance(target_pct, (int, float)):
        print(f"  - Target Percentile: {target_pct:.2%}")
    else:
        print(f"  - Target Percentile: {target_pct}")

    print(f"  - Recommendation: {results.get('recommendation', 'N/A')}")

    return results


async def main():
    """Run all agent tests sequentially"""
    try:
        # Test 1: Data Collection
        data_results = await test_data_collector()

        # Test 2: Model Training
        model_results = await test_model_trainer(data_results)

        # Test 3: Submission
        submission_results = await test_submission(model_results, data_results)

        # Test 4: Leaderboard Monitoring
        leaderboard_results = await test_leaderboard(submission_results)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())