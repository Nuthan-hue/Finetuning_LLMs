#!/usr/bin/env python3
"""
Quick launcher for Agentic Orchestrator

Usage:
    python run_agentic.py                          # Interactive mode
    python run_agentic.py titanic                  # Auto-run titanic
    python run_agentic.py titanic --target 0.10    # Top 10% goal
    python run_agentic.py titanic --actions 100    # More iterations
"""
import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.agents import AgenticOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kaggle_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def run_agentic(
    competition_name: str,
    target_percentile: float = 0.20,
    max_actions: int = 50
):
    """
    Run agentic orchestrator for a competition.

    Args:
        competition_name: Kaggle competition name (e.g., 'titanic')
        target_percentile: Target ranking (default: 0.20 for top 20%)
        max_actions: Maximum actions the coordinator can take (default: 50)
    """
    print("\n" + "=" * 80)
    print("🧠 AGENTIC ORCHESTRATOR LAUNCHER")
    print("=" * 80)
    print(f"Competition: {competition_name}")
    print(f"Target: Top {target_percentile * 100}%")
    print(f"Max actions: {max_actions}")
    print(f"Mode: 🧠 TRULY AGENTIC (AI coordinator decides workflow)")
    print("=" * 80 + "\n")

    # Create orchestrator
    orchestrator = AgenticOrchestrator(
        competition_name=competition_name,
        target_percentile=target_percentile,
        max_actions=max_actions
    )

    # Run it!
    context = {"competition_name": competition_name}

    try:
        results = await orchestrator.run(context)

        # Display results
        print("\n" + "=" * 80)
        print("🎉 FINAL RESULTS")
        print("=" * 80)
        print(f"Competition: {competition_name}")
        print(f"Final Rank: {results['final_rank']}")
        print(f"Final Percentile: {results['final_percentile']:.1%}")
        print(f"Target Met: {'✅ YES' if results['target_met'] else '❌ NO'}")
        print(f"Total Actions: {results['total_actions']}/{max_actions}")
        print("=" * 80)

        # Action summary
        print("\n📊 Action History:")
        for action in results['action_history'][-5:]:  # Last 5 actions
            status_icon = "✅" if action['status'] == 'success' else "❌"
            print(f"  {status_icon} #{action['action_number']}: {action['action']}")
            print(f"      → {action['reasoning'][:80]}...")

        print("\n💾 Full results saved to logs/kaggle_agent.log")

        return results

    except Exception as e:
        logger.error(f"Error running agentic orchestrator: {e}")
        raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run Agentic Orchestrator for Kaggle competitions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agentic.py                          # Interactive mode
  python run_agentic.py titanic                  # Run titanic competition
  python run_agentic.py house-prices --target 0.10   # Top 10% goal
  python run_agentic.py nlp-getting-started --actions 100  # More iterations

Popular competitions:
  - titanic                                      # Classic binary classification
  - house-prices-advanced-regression-techniques  # Regression
  - nlp-getting-started                          # NLP text classification
  - digit-recognizer                             # Computer vision (MNIST)
        """
    )

    parser.add_argument(
        'competition',
        nargs='?',
        default=None,
        help='Kaggle competition name (e.g., titanic)'
    )

    parser.add_argument(
        '--target',
        type=float,
        default=0.20,
        help='Target percentile (default: 0.20 for top 20%%)'
    )

    parser.add_argument(
        '--actions',
        type=int,
        default=50,
        help='Maximum actions for coordinator (default: 50)'
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("submissions").mkdir(exist_ok=True)

    # Interactive or direct mode
    if args.competition:
        # Direct mode - run immediately
        asyncio.run(run_agentic(
            competition_name=args.competition,
            target_percentile=args.target,
            max_actions=args.actions
        ))
    else:
        # Interactive mode
        print("\n" + "=" * 80)
        print("🧠 AGENTIC ORCHESTRATOR LAUNCHER")
        print("=" * 80)
        print("\nThis runs the AGENTIC AI mode where the coordinator agent decides the workflow\n")

        competition = input("Enter competition name: ").strip()
        if not competition:
            print("❌ Competition name required!")
            return

        target = input("Target percentile (default 0.20 for top 20%): ").strip()
        target_percentile = float(target) if target else 0.20

        actions = input("Max actions (default 50): ").strip()
        max_actions = int(actions) if actions else 50

        asyncio.run(run_agentic(
            competition_name=competition,
            target_percentile=target_percentile,
            max_actions=max_actions
        ))


if __name__ == "__main__":
    main()