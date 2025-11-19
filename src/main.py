"""
Main entry point for Kaggle Competition Multi-Agent System

This module demonstrates how to use the orchestrator and specialized agents
to automate Kaggle competition participation.
"""
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.agents import (
    AgenticOrchestrator,
    DataCollector,
    ModelTrainer,
    Submitter,
    LeaderboardMonitor
)

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


async def run_full_competition(competition_name: str, target_percentile: float = 0.20):
    """
    Run the full competition workflow using the agentic orchestrator.

    Args:
        competition_name: Name of the Kaggle competition
        target_percentile: Target ranking (default: top 20%)
    """
    print(f"Starting full competition workflow for: {competition_name}")
    print("Mode: Agentic AI (Coordinator Agent decides workflow dynamically)")

    # Initialize agentic orchestrator
    orchestrator = AgenticOrchestrator(
        competition_name=competition_name,
        target_percentile=target_percentile,
        max_actions=50
    )

    # Prepare context
    # AI decides everything - no manual overrides needed
    context = {
        "competition_name": competition_name
    }

    try:
        # Run orchestration
        results = await orchestrator.run(context)

        # Get workflow summary
        summary = orchestrator.get_workflow_summary()
        print(f"\nWorkflow History: {len(summary['workflow_history'])} iterations")

        return results

    except Exception as e:
        print(f"Error in competition workflow: {str(e)}")
        raise


async def run_data_collection_only(competition_name: str):
    """
    Run only the data collection phase.

    Useful for exploring a competition before starting the full workflow.
    """
    print(f"Running data collection for: {competition_name}")

    # Initialize data collector
    collector = DataCollector()

    # Run data collection
    context = {
        "competition_name": competition_name,
        "analyze": True
    }

    results = await collector.run(context)

    # Display analysis
    analysis = results.get("analysis_report", {})
    print("\nData Analysis:")
    print(f"Files found: {len(analysis.get('files', []))}")

    for filename, info in analysis.get("datasets", {}).items():
        print(f"\n{filename}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {len(info['columns'])}")
        print(f"  Target: {info.get('target_column', 'Unknown')}")

    return results


async def run_training_only(data_path: str, target_column: str):
    """
    Run only the model training phase.

    Useful for testing different models on already downloaded data.
    """
    print("Running model training...")

    # Initialize trainer
    trainer = ModelTrainer()

    # Run training
    context = {
        "data_path": data_path,
        "target_column": target_column,
        "config": {
            "learning_rate": 0.05,
            "num_boost_round": 1000,
        }
    }

    results = await trainer.run(context)
    return results


def display_menu():
    """Display interactive menu."""
    print("Kaggle Competition Multi-Agent System")
    print("\nOptions:")
    print("1. Run Full Competition Workflow (Automated)")
    print("2. Data Collection Only")
    print("3. Model Training Only")
    print("4. Check Leaderboard")
    print("5. Exit")



async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("submissions").mkdir(exist_ok=True)

    # Check Kaggle credentials
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("Kaggle credentials not found in environment!")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        print("Or ensure ~/.kaggle/kaggle.json is properly configured")

    while True:
        display_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        try:
            if choice == "1":
                # Full workflow
                competition_name = input("Enter competition name: ").strip()
                target = input("Target percentile (default 0.20 for top 20%): ").strip()
                target_percentile = float(target) if target else 0.20

                await run_full_competition(competition_name, target_percentile)

            elif choice == "2":
                # Data collection only
                competition_name = input("Enter competition name: ").strip()
                await run_data_collection_only(competition_name)

            elif choice == "3":
                # Training only
                data_path = input("Enter path to training data: ").strip()
                target_column = input("Enter target column name: ").strip()
                await run_training_only(data_path, target_column)

            elif choice == "4":
                # Check leaderboard
                competition_name = input("Enter competition name: ").strip()
                monitor = LeaderboardMonitor()
                results = await monitor.run({"competition_name": competition_name})

                print("\nLeaderboard Status:")
                print(f"Current Rank: {results.get('current_rank', 'N/A')}")
                print(f"Total Teams: {results.get('total_teams', 'N/A')}")
                print(f"Percentile: {results.get('current_percentile', 'N/A'):.2%}")

            elif choice == "5":
                # Exit
                print("\nGoodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"\nError occurred: {str(e)}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())