"""
Command Line Interface for Kaggle Competition Multi-Agent System
"""
import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional

from agents.orchestrator import AgenticOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║    🤖  KAGGLE-SLAYING MULTI-AGENT TEAM  🤖                    ║
║                                                                ║
║    Autonomous AI system for Kaggle competitions                ║
║    Target: Top 20% Rankings                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_instructions():
    """Print usage instructions"""
    print("\n📋 How it works:")
    print("  1. Enter a Kaggle competition name")
    print("  2. System collects data automatically")
    print("  3. Trains multiple models")
    print("  4. Submits predictions to Kaggle")
    print("  5. Monitors leaderboard and iterates")
    print("  6. Keeps improving until top 20% reached\n")


def get_competition_input() -> Optional[str]:
    """Get competition name from user"""
    print("╔" + "═" * 60 + "╗")
    print("║" + " " * 20 + "COMPETITION SETUP" + " " * 23 + "║")
    print("╚" + "═" * 60 + "╝\n")

    while True:
        competition_name = input("🎯 Enter Kaggle competition name (or 'quit' to exit): ").strip()

        if competition_name.lower() in ['quit', 'exit', 'q']:
            return None

        if not competition_name:
            print("❌ Competition name cannot be empty. Please try again.\n")
            continue

        # Confirm with user
        print(f"\n📌 Competition: {competition_name}")
        confirm = input("   Proceed? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            return competition_name

        print("❌ Cancelled. Try again.\n")


def get_advanced_settings() -> dict:
    """Get advanced settings from user (optional)"""
    print("\n⚙️  Advanced Settings (press Enter for defaults)")

    # Target percentile
    while True:
        target_input = input("   Target percentile (default: 20% = 0.20): ").strip()
        if not target_input:
            target_percentile = 0.20
            break
        try:
            target_percentile = float(target_input)
            if 0 < target_percentile <= 1:
                break
            print("   ❌ Must be between 0 and 1 (e.g., 0.20 for top 20%)")
        except ValueError:
            print("   ❌ Invalid number. Try again.")

    # Max actions
    while True:
        actions_input = input("   Max actions (default: 50): ").strip()
        if not actions_input:
            max_actions = 50
            break
        try:
            max_actions = int(actions_input)
            if max_actions > 0:
                break
            print("   ❌ Must be positive integer")
        except ValueError:
            print("   ❌ Invalid number. Try again.")

    return {
        "target_percentile": target_percentile,
        "max_actions": max_actions
    }


def check_prerequisites():
    """Check if prerequisites are met"""
    print("\n🔍 Checking prerequisites...")

    # Check Kaggle credentials
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        print("❌ Kaggle API credentials not found!")
        print("\n📥 Setup Instructions:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print(f"  4. Move kaggle.json to {kaggle_path}")
        print(f"  5. Run: chmod 600 {kaggle_path}\n")
        return False

    print("✓ Kaggle API credentials found")

    # Check data directory
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("submissions").mkdir(exist_ok=True)
    print("✓ Directories initialized")

    return True


async def run_competition(competition_name: str, settings: dict):
    """Run the full competition workflow using agentic AI"""
    print("\n" + "=" * 62)
    print("🚀 STARTING AGENTIC AI WORKFLOW")
    print("=" * 62)

    print(f"\n📊 Configuration:")
    print(f"   Competition: {competition_name}")
    print(f"   Target: Top {settings['target_percentile']*100:.0f}%")
    print(f"   Max Actions: {settings['max_actions']}")

    # Create agentic orchestrator
    orchestrator = AgenticOrchestrator(
        competition_name=competition_name,
        target_percentile=settings['target_percentile'],
        max_actions=settings['max_actions']
    )

    # Run workflow
    try:
        results = await orchestrator.run({})

        # Display results
        print("\n" + "=" * 62)
        print("🎉 WORKFLOW COMPLETED!")
        print("=" * 62)

        print(f"\n📈 Final Results:")
        print(f"   Rank: {results.get('final_rank', 'N/A')}")
        print(f"   Percentile: {results.get('final_percentile', 1.0)*100:.1f}%")
        print(f"   Iterations: {results.get('total_iterations', 0)}")
        print(f"   Target Met: {'✓ YES' if results.get('target_met') else '✗ NO'}")

        if results.get('workflow_history'):
            print(f"\n📜 Iteration History:")
            for entry in results['workflow_history']:
                iteration = entry['iteration']
                rank = entry.get('rank', 'N/A')
                percentile = entry.get('percentile', 1.0)
                print(f"   Iteration {iteration}: Rank {rank} ({percentile*100:.1f}%)")

        return True

    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        logger.exception("Workflow failed")
        return False


def main():
    """Main CLI entry point"""
    try:
        print_banner()

        # Check prerequisites
        if not check_prerequisites():
            sys.exit(1)

        print_instructions()

        # Get competition name
        competition_name = get_competition_input()
        if not competition_name:
            print("\n👋 Goodbye!")
            sys.exit(0)

        # Get settings
        use_advanced = input("\n⚙️  Use advanced settings? (yes/no, default: no): ").strip().lower()
        if use_advanced in ['yes', 'y']:
            settings = get_advanced_settings()
        else:
            settings = {
                "target_percentile": 0.20,
                "max_actions": 50
            }

        # Important reminders
        print("\n⚠️  Important Reminders:")
        print(f"   • Join the competition at: https://www.kaggle.com/c/{competition_name}")
        print("   • Accept the competition rules")
        print("   • The system will submit predictions automatically")
        print("   • Leaderboard updates may take a few minutes")

        proceed = input("\n✓ Ready to start? (yes/no): ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("\n👋 Cancelled. Goodbye!")
            sys.exit(0)

        # Run the competition
        asyncio.run(run_competition(competition_name, settings))

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error in CLI")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
