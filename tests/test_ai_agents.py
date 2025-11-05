#!/usr/bin/env python3
"""
Quick test to verify AI agents are working
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.llm_agents import StrategyAgent, DataAnalysisAgent


async def test_strategy_agent():
    """Test Strategy Agent"""
    print("=" * 60)
    print("Testing Strategy Agent")
    print("=" * 60)

    try:
        agent = StrategyAgent()
        print("✓ StrategyAgent initialized successfully")

        # Test decision making
        print("\nAsking AI for strategy decision...")
        strategy = await agent.select_optimization_strategy(
            recommendation="major_improvement_needed",
            current_model="lightgbm",
            tried_models=["lightgbm"],
            current_percentile=0.75,
            target_percentile=0.20,
            iteration=1,
            competition_type="tabular",
            performance_history=[]
        )

        print("\n✓ AI Strategy Decision:")
        print(f"  Action: {strategy['action']}")
        print(f"  Reasoning: {strategy.get('reasoning', 'N/A')[:200]}...")
        print(f"  Expected Improvement: {strategy.get('expected_improvement', 'N/A')}")
        print(f"  Confidence: {strategy.get('confidence', 'N/A')}")

        if strategy.get('config_updates'):
            print(f"\n  Hyperparameter Suggestions:")
            for key, value in strategy['config_updates'].items():
                print(f"    {key}: {value}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_analysis_agent():
    """Test Data Analysis Agent"""
    print("\n" + "=" * 60)
    print("Testing Data Analysis Agent")
    print("=" * 60)

    try:
        agent = DataAnalysisAgent()
        print("✓ DataAnalysisAgent initialized successfully")

        # Test with sample Titanic dataset info
        dataset_info = {
            "datasets": {
                "train.csv": {
                    "rows": 891,
                    "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
                    "dtypes": {
                        "PassengerId": "int64",
                        "Survived": "int64",
                        "Pclass": "int64",
                        "Name": "object",
                        "Sex": "object",
                        "Age": "float64"
                    },
                    "missing_values": {
                        "Age": 177,
                        "Cabin": 687,
                        "Embarked": 2
                    }
                }
            }
        }

        print("\nAsking AI to analyze dataset...")
        analysis = await agent.analyze_and_suggest(
            dataset_info=dataset_info,
            competition_name="titanic"
        )

        print("\n✓ AI Data Analysis:")
        print(f"  Target Column: {analysis.get('target_column', 'N/A')}")
        print(f"  Confidence: {analysis.get('target_confidence', 'N/A')}")
        print(f"  Task Type: {analysis.get('task_type', 'N/A')}")

        if analysis.get('preprocessing'):
            print(f"\n  Preprocessing Suggestions:")
            for key, value in analysis['preprocessing'].items():
                print(f"    {key}: {value}")

        if analysis.get('feature_engineering'):
            print(f"\n  Feature Engineering Ideas:")
            for i, idea in enumerate(analysis['feature_engineering'][:3], 1):
                print(f"    {i}. {idea}")

        if analysis.get('recommended_models'):
            print(f"\n  Recommended Models: {', '.join(analysis['recommended_models'])}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🤖 Testing AI Agents Integration\n")

    # Test both agents
    strategy_ok = await test_strategy_agent()
    data_ok = await test_data_analysis_agent()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Strategy Agent: {'✓ PASS' if strategy_ok else '✗ FAIL'}")
    print(f"Data Analysis Agent: {'✓ PASS' if data_ok else '✗ FAIL'}")

    if strategy_ok and data_ok:
        print("\n🎉 All AI agents working correctly!")
        print("\nYou can now run the full system with:")
        print("  python3 kaggle_agent.py")
        print("\nThe AI will intelligently:")
        print("  - Identify target columns")
        print("  - Select optimization strategies")
        print("  - Suggest hyperparameters")
        print("  - Recommend feature engineering")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())