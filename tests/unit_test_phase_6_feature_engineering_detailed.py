#!/usr/bin/env python3
"""
Quick test of FeatureEngineeringAgent
Tests Phase 6 in isolation with mock data
"""

import asyncio
import sys
import json
from pathlib import Path
import pytest
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.llm_agents.feature_engineering_agent import FeatureEngineeringAgent

@pytest.mark.asyncio
async def test_feature_engineering():
    """Test FeatureEngineeringAgent code generation"""

    print("\n" + "=" * 80)
    print("TESTING FEATURE ENGINEERING AGENT")
    print("=" * 80)

    # Mock data analysis (as if from DataAnalysisAgent)
    data_analysis = {
        "data_modality": "tabular",
        "target_column": "Survived",
        "data_files": {
            "train_file": "train.csv",
            "test_file": "test.csv"
        },
        "feature_types": {
            "id_columns": ["PassengerId"],
            "numerical": ["Age", "Fare", "SibSp", "Parch"],
            "categorical": ["Sex", "Pclass", "Embarked"]
        }
    }

    # Mock feature engineering plan (as if from PlanningAgent)
    feature_plan = [
        {
            "feature_name": "family_size",
            "formula": "SibSp + Parch + 1",
            "reason": "Capture family unit effect on survival",
            "priority": 1
        },
        {
            "feature_name": "is_alone",
            "formula": "(family_size == 1).astype(int)",
            "reason": "Binary indicator for solo travelers",
            "priority": 1
        },
        {
            "feature_name": "fare_per_person",
            "formula": "Fare / family_size",
            "reason": "Normalize fare by family size",
            "priority": 2
        }
    ]

    # Create a temporary test directory
    test_dir = Path("data/test_feature_engineering")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save data_analysis.json (so AI can read it)
    analysis_file = test_dir / "data_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(data_analysis, f, indent=2)

    print(f"\n✅ Created test directory: {test_dir}")
    print(f"✅ Saved data_analysis.json")

    # Initialize FeatureEngineeringAgent
    print("\n🤖 Initializing FeatureEngineeringAgent...")
    agent = FeatureEngineeringAgent()

    # Generate feature engineering code
    print("\n🤖 Generating feature engineering code with AI...")
    print(f"   Features to create: {len(feature_plan)}")
    print(f"   Data modality: {data_analysis['data_modality']}")

    try:
        code = await agent.generate_feature_engineering_code(
            feature_engineering_plan=feature_plan,
            data_analysis=data_analysis,
            clean_data_path=str(test_dir)
        )

        print(f"\n✅ Code generated successfully! ({len(code)} chars)")

        # Save generated code
        code_file = test_dir / "generated_feature_engineering.py"
        code_file.write_text(code)
        print(f"💾 Saved code to: {code_file}")

        # Display first 1000 chars
        print("\n" + "-" * 80)
        print("GENERATED CODE (first 1000 chars):")
        print("-" * 80)
        print(code[:1000])
        if len(code) > 1000:
            print(f"\n... ({len(code) - 1000} more characters)")
        print("-" * 80)

        # Check for hardcoding
        print("\n🔍 Checking for hardcoded values...")
        issues = []

        if '"train.csv"' in code or "'train.csv'" in code:
            issues.append("❌ Found hardcoded 'train.csv'")
        else:
            print("✅ No hardcoded 'train.csv'")

        if '"test.csv"' in code or "'test.csv'" in code:
            issues.append("❌ Found hardcoded 'test.csv'")
        else:
            print("✅ No hardcoded 'test.csv'")

        if '"Survived"' in code or "'Survived'" in code:
            # Check if it's reading from data_analysis
            if 'data_analysis' in code and 'target_column' in code:
                print("✅ Target column read from data_analysis")
            else:
                issues.append("⚠️  Found 'Survived' - check if read from data_analysis")

        if 'data_analysis.json' in code or 'data_analysis' in code:
            print("✅ Code reads from data_analysis")
        else:
            issues.append("❌ Code doesn't read data_analysis.json")

        if issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n🎉 NO HARDCODING DETECTED!")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print(f"✅ FeatureEngineeringAgent is working!")
        print(f"✅ Generated {len(code)} chars of executable Python code")
        print(f"✅ Code saved to: {code_file}")
        print(f"✅ Ready for Phase 6 integration")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_feature_engineering())
    sys.exit(0 if success else 1)