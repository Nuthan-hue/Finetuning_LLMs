#!/usr/bin/env python3
"""
Test script to debug leaderboard monitoring
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.leaderboard import LeaderboardMonitorAgent


async def test_leaderboard():
    """Test leaderboard fetching"""
    print("=" * 60)
    print("Testing Leaderboard Monitor")
    print("=" * 60)

    agent = LeaderboardMonitorAgent(target_percentile=0.20)

    # Test with titanic competition
    context = {
        "competition_name": "titanic"
    }

    print("\nFetching leaderboard for: titanic")
    print("-" * 60)

    try:
        results = await agent.run(context)

        print("\n✓ Results:")
        for key, value in results.items():
            if key not in ["leaderboard_data", "submission_history"]:
                print(f"  {key}: {value}")

        # Show submissions if any
        if "submission_history" in results and results["submission_history"]:
            print("\nSubmission History:")
            for sub in results["submission_history"][:5]:  # Show first 5
                print(f"  - {sub}")
        else:
            print("\n⚠️  No submissions found")
            print("\nTroubleshooting:")
            print("1. Check if you've accepted the competition rules at: https://www.kaggle.com/c/titanic")
            print("2. Verify your submission was uploaded: kaggle competitions submissions titanic")
            print("3. Wait a few minutes - Kaggle takes time to process submissions")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_leaderboard())