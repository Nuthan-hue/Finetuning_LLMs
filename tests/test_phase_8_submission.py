#!/usr/bin/env python3
"""
Test Phase 8: Submission
Tests the SubmissionAgent's ability to generate and submit predictions.
"""
import asyncio
import sys
import json
from pathlib import Path
from unittest.mock import patch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.phases import run_submission


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        from src.agents.submission import Submitter
        self.submission_agent = Submitter()
        self.iteration = 1

@pytest.mark.asyncio
async def test_phase_8_submission(competition_name: str = "titanic", submit_to_kaggle: bool = False):
    """
    Test Phase 8: Submission

    Tests:
    - Load trained model
    - Generate predictions on test data
    - Format submission file
    - Optional: Submit to Kaggle (if submit_to_kaggle=True)
    - Context propagation

    Args:
        competition_name: Name of the competition to test
        submit_to_kaggle: If True, actually submit to Kaggle. If False, just create submission file.
    """
    print("\n" + "=" * 70)
    print("PHASE 8: SUBMISSION - TEST")
    print("=" * 70)

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase8_cache.json"

    # Check cache
    if cache_file.exists():
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {"competition_name": competition_name}
        context.update(cached)

        assert "submission_path" in context or "submission_file" in context, "Missing submission_path/submission_file in cache"
        print("\n✅ Phase 8 PASSED (cached)")
        print(f"   Submission file: {context.get('submission_path') or context.get('submission_file')}")
        if context.get("leaderboard_score"):
            print(f"   Score: {context.get('leaderboard_score')}")
        return True

    # Load dependencies from previous phases
    phase3_cache = test_cache_dir / "test_phase3_cache.json"
    phase4_cache = test_cache_dir / "test_phase4_cache.json"
    phase7_cache = test_cache_dir / "test_phase7_cache.json"

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": competition_name,
        "data_path": f"data/{competition_name}/raw",
    }

    # Load Phase 3 data (data_files, target column)
    if phase3_cache.exists():
        with open(phase3_cache, 'r') as f:
            cached_phase3 = json.load(f)
            context["target_column"] = cached_phase3.get("target_column", "Survived")
            context["data_files"] = cached_phase3.get("data_files", {"train_file": "train.csv", "test_file": "test.csv"})
    else:
        context["target_column"] = "Survived"
        context["data_files"] = {"train_file": "train.csv", "test_file": "test.csv"}

    # Load Phase 4 data (clean data path)
    if phase4_cache.exists():
        with open(phase4_cache, 'r') as f:
            cached_phase4 = json.load(f)
            if cached_phase4.get("clean_data_path"):
                context["clean_data_path"] = cached_phase4["clean_data_path"]

    # Load Phase 7 data (model)
    if phase7_cache.exists():
        with open(phase7_cache, 'r') as f:
            cached_phase7 = json.load(f)
            context.update(cached_phase7)
    else:
        # Fallback to mock model data
        context["model_path"] = "models/final/lightgbm_model.txt"
        context["model_type"] = "lightgbm"

    try:
        print("\n📤 Running Phase 8: Submission...")
        print("   (Generating predictions...)")

        if submit_to_kaggle:
            # Real submission to Kaggle - mock input to say "yes"
            print("   🚀 REAL SUBMISSION MODE - Will submit to Kaggle!")
            with patch('builtins.input', return_value='yes'):
                context = await asyncio.wait_for(
                    run_submission(orchestrator, context),
                    timeout=120
                )
        else:
            # Interactive mode - ask user if they want to submit
            print("\n" + "=" * 70)
            print("📤 SUBMISSION READY")
            print("=" * 70)
            print(f"   Competition: {competition_name}")
            print(f"   Model: {context.get('model_type', 'unknown')}")
            if context.get('cv_score'):
                print(f"   CV Score: {context.get('cv_score'):.4f}")
            print()

            user_answer = input("🤔 Submit to Kaggle leaderboard? (yes/no): ").strip().lower()

            if user_answer == 'yes':
                print("   ✅ Submitting to Kaggle...")
                # Don't mock - let it actually ask (user already answered above)
                context = await asyncio.wait_for(
                    run_submission(orchestrator, context),
                    timeout=120
                )
            else:
                print("   ⏭️  Creating submission file only (no upload)")
                with patch('builtins.input', return_value='no'):
                    context = await asyncio.wait_for(
                        run_submission(orchestrator, context),
                        timeout=120
                    )

        assert "submission_file" in context, "Missing submission_file"

        # Save to cache (only JSON-serializable data)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "submission_file": context.get("submission_file"),
                "submission_id": context.get("submission_id"),
                "leaderboard_score": context.get("leaderboard_score")
            }, f, indent=2)

        print("\n✅ Phase 8 PASSED")
        print(f"   Submission file: {context.get('submission_file')}")
        if context.get("leaderboard_score"):
            print(f"   Score: {context.get('leaderboard_score')}")
        return True

    except asyncio.TimeoutError:
        print("\n❌ Phase 8 TIMED OUT")
        return False
    except Exception as e:
        print(f"\n❌ Phase 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """
    Run Phase 8 test

    Usage:
        python test_phase_8_submission.py [competition_name] [--submit]

    Examples:
        python test_phase_8_submission.py                    # Test mode (no submission)
        python test_phase_8_submission.py titanic            # Test mode for titanic
        python test_phase_8_submission.py titanic --submit   # Real submission to Kaggle
    """
    competition_name = "titanic"
    submit_to_kaggle = False

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] != "--submit":
            competition_name = sys.argv[1]

    if "--submit" in sys.argv:
        submit_to_kaggle = True
        print(f"\n⚠️  REAL SUBMISSION MODE ENABLED!")
        print(f"    Will submit to Kaggle competition: {competition_name}")
        print(f"    Press Ctrl+C within 3 seconds to cancel...\n")
        await asyncio.sleep(3)

    success = await test_phase_8_submission(competition_name, submit_to_kaggle)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())