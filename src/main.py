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

from agents import (
    Orchestrator,  # Legacy: Scripted pipeline
    AgenticOrchestrator,  # New: Truly agentic
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


async def run_full_competition(competition_name: str, target_percentile: float = 0.20, use_agentic: bool = False):
    """
    Run the full competition workflow using the orchestrator.

    Args:
        competition_name: Name of the Kaggle competition
        target_percentile: Target ranking (default: top 20%)
        use_agentic: Use truly agentic coordinator (default: False for legacy)
    """
    logger.info(f"Starting full competition workflow for: {competition_name}")
    logger.info(f"Mode: {'🧠 TRULY AGENTIC (Coordinator decides workflow)' if use_agentic else '📋 Legacy (Scripted pipeline)'}")

    # Initialize orchestrator
    if use_agentic:
        orchestrator = AgenticOrchestrator(
            competition_name=competition_name,
            target_percentile=target_percentile,
            max_actions=50  # Agentic uses actions instead of iterations
        )
    else:
        orchestrator = Orchestrator(
            competition_name=competition_name,
            target_percentile=target_percentile,
            max_iterations=5
        )

    # Prepare context
    # AI decides everything - no manual overrides needed
    context = {
        "competition_name": competition_name
    }

    try:
        # Run orchestration
        results = await orchestrator.run(context)

        # Display results
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS")
        logger.info("="*50)
        logger.info(f"Competition: {competition_name}")
        logger.info(f"Final Rank: {results.get('final_rank', 'N/A')}")
        logger.info(f"Final Percentile: {results.get('final_percentile', 'N/A'):.2%}")
        logger.info(f"Target Met: {results.get('target_met', False)}")
        logger.info(f"Total Iterations: {results.get('total_iterations', 0)}")

        # Get workflow summary
        summary = orchestrator.get_workflow_summary()
        logger.info(f"\nWorkflow History: {len(summary['workflow_history'])} iterations")

        return results

    except Exception as e:
        logger.error(f"Error in competition workflow: {str(e)}")
        raise


async def run_data_collection_only(competition_name: str):
    """
    Run only the data collection phase.

    Useful for exploring a competition before starting the full workflow.
    """
    logger.info(f"Running data collection for: {competition_name}")

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
    logger.info("\nData Analysis:")
    logger.info(f"Files found: {len(analysis.get('files', []))}")

    for filename, info in analysis.get("datasets", {}).items():
        logger.info(f"\n{filename}:")
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Columns: {len(info['columns'])}")
        logger.info(f"  Target: {info.get('target_column', 'Unknown')}")

    return results


async def run_training_only(data_path: str, target_column: str):
    """
    Run only the model training phase.

    Useful for testing different models on already downloaded data.
    """
    logger.info("Running model training...")

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

    logger.info("\nTraining Results:")
    logger.info(f"Model Type: {results.get('model_type')}")
    logger.info(f"Model Path: {results.get('model_path')}")
    logger.info(f"Best Score: {results.get('best_score'):.4f}")
    logger.info(f"Metric: {results.get('metric')}")

    return results


def display_menu():
    """Display interactive menu."""
    print("\n" + "="*60)
    print("Kaggle Competition Multi-Agent System")
    print("="*60)
    print("\nOptions:")
    print("1. Run Full Competition Workflow (Automated)")
    print("2. Data Collection Only")
    print("3. Model Training Only")
    print("4. Check Leaderboard")
    print("5. Exit")
    print("\n" + "="*60)


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
        logger.warning("Kaggle credentials not found in environment!")
        logger.warning("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        logger.warning("Or ensure ~/.kaggle/kaggle.json is properly configured")

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
            logger.error(f"Error: {str(e)}")
            print(f"\nError occurred: {str(e)}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())