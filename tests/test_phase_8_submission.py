#!/usr/bin/env python3
"""
Test Phase 8: Submission
Tests the SubmissionAgent's ability to generate and submit predictions.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.submission import SubmissionAgent


async def test_phase_8_submission():
    """
    Test Phase 8: Submission

    Tests:
    - Load trained model
    - Generate predictions on test data
    - Format submission file
    - Optional: Submit to Kaggle
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 8: SUBMISSION - TEST")
    print("=" * 70)

    agent = SubmissionAgent()

    context = {
        "competition_name": "titanic",
        "model_path": "models/titanic_model.pkl",  # From Phase 7
        "model_type": "lightgbm",
        "test_data_path": "data/titanic/test.csv",
        "target_column": "Survived",
        "auto_submit": False,  # Set to True to actually submit
        "submission_message": "Test submission"
    }

    try:
        print("\n📤 Running Phase 8: Submission...")
        print("   Generating predictions...")

        result = await agent.run(context)

        # Validate outputs
        assert "submission_path" in result, "Missing submission_path"
        assert Path(result["submission_path"]).exists(), "Submission file not created"

        print("\n✅ Phase 8 PASSED")
        print(f"   Submission file: {result['submission_path']}")
        if result.get("submission_status"):
            print(f"   Status: {result['submission_status']}")

        return True

    except Exception as e:
        print(f"\n❌ Phase 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 8 test"""
    success = await test_phase_8_submission()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())