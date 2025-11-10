"""
Quick test script for the updated DataAnalysisAgent
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.llm_agents.data_analysis_agent import DataAnalysisAgent


async def test_data_analysis_agent():
    """Test the DataAnalysisAgent with Titanic data"""

    print("=" * 70)
    print("TESTING DATA ANALYSIS AGENT")
    print("=" * 70)

    # Setup
    data_path = "data/raw/titanic"
    competition_name = "titanic"
    problem_understanding = {
        "competition_type": "binary_classification",
        "task_description": "Predict passenger survival on Titanic",
        "evaluation_metric": "accuracy",
        "target_variable": "Survived"
    }

    # Initialize agent
    print("\n1. Initializing DataAnalysisAgent...")
    agent = DataAnalysisAgent()

    # Run analysis
    print("\n2. Running statistical analysis...")
    try:
        analysis = await agent.analyze_and_suggest(
            data_path=data_path,
            competition_name=competition_name,
            problem_understanding=problem_understanding
        )

        print("\n✅ Analysis completed successfully!")
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)

        # Display key results
        print(f"\n📊 Data Modality: {analysis.get('data_modality')}")
        print(f"🎯 Target Column: {analysis.get('target_column')}")
        print(f"📈 Target Type: {analysis.get('target_type')}")
        print(f"⚖️  Is Imbalanced: {analysis.get('is_imbalanced')}")
        print(f"🔧 Preprocessing Required: {analysis.get('preprocessing_required')}")

        # Feature types
        if "feature_types" in analysis:
            print("\n🔍 Feature Types:")
            feature_types = analysis["feature_types"]
            print(f"  • ID columns: {feature_types.get('id_columns', [])}")
            print(f"  • Numerical: {len(feature_types.get('numerical', []))} columns")
            print(f"  • Categorical: {len(feature_types.get('categorical', []))} columns")
            print(f"  • Text: {len(feature_types.get('text', []))} columns")
            print(f"  • Drop candidates: {feature_types.get('drop_candidates', [])}")

        # Data quality
        if "data_quality" in analysis:
            print("\n⚠️  Data Quality Issues:")
            quality = analysis["data_quality"]
            if quality.get("missing_values"):
                print(f"  • Missing values in: {list(quality['missing_values'].keys())}")
            if quality.get("outliers"):
                print(f"  • Outliers in: {quality['outliers']}")

        # Preprocessing recommendations
        if "preprocessing" in analysis:
            print("\n🔧 Preprocessing Recommendations:")
            preprocessing = analysis["preprocessing"]
            if preprocessing.get("drop_columns"):
                print(f"  • Drop columns: {preprocessing['drop_columns']}")
            if preprocessing.get("impute_missing"):
                print(f"  • Impute missing: {len(preprocessing['impute_missing'])} columns")
            if preprocessing.get("encode_categorical"):
                print(f"  • Encode categorical: {len(preprocessing['encode_categorical'])} columns")

        # Key insights
        if "key_insights" in analysis:
            print("\n💡 Key Insights:")
            for insight in analysis["key_insights"]:
                print(f"  • {insight}")

        # Save full analysis
        output_file = Path("data/raw/titanic/test_data_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\n💾 Full analysis saved to: {output_file}")

        print("\n" + "=" * 70)
        print("✅ TEST PASSED - DataAnalysisAgent is working correctly!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_data_analysis_agent())
    sys.exit(0 if success else 1)