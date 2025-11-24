#!/usr/bin/env python3
"""
Quick runner for autonomous Kaggle competition workflow
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import run_full_competition


async def main():
    """Run autonomous workflow"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    target_percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20

    print("=" * 80)
    print("🤖 AUTONOMOUS KAGGLE COMPETITION SYSTEM")
    print("=" * 80)
    print(f"Competition: {competition_name}")
    print(f"Target: Top {target_percentile * 100:.0f}% (percentile {target_percentile})")
    print(f"Mode: FULLY AUTONOMOUS (AI decides everything)")
    print("=" * 80)
    print()

    results = await run_full_competition(competition_name, target_percentile)

    print("\n" + "=" * 80)
    print("🎉 FINAL RESULTS")
    print("=" * 80)
    print(f"Final Rank: {results.get('final_rank', 'N/A')}")
    print(f"Final Percentile: {results.get('final_percentile', 1.0) * 100:.2f}%")
    print(f"Target Met: {'✅ YES' if results.get('target_met') else '❌ NO'}")
    print(f"Total Actions: {results.get('total_actions', 0)}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())