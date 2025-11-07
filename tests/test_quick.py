"""
Quick test of refactored orchestrator on Titanic
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path (go up one level from tests/ to project root, then into src/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def quick_test():
    """Quick test - 1 iteration only"""

    logger.info("🧪 QUICK TEST: Refactored Orchestrator on Titanic")
    logger.info("=" * 60)

    # Get project root (go up from tests/ to project root)
    project_root = Path(__file__).parent.parent

    # Create orchestrator with absolute paths
    orchestrator = Orchestrator(
        competition_name="titanic",
        data_dir=str(project_root / "data"),
        models_dir=str(project_root / "models"),
        submissions_dir=str(project_root / "submissions"),
        target_percentile=0.20,
        max_iterations=1  # Just 1 iteration for quick test
    )

    context = {
        "competition_name": "titanic",
        "training_config": {
            "num_boost_round": 50  # Quick training
        }
    }

    try:
        results = await orchestrator.run(context)

        logger.info("\n" + "=" * 60)
        logger.info("✅ TEST COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Iterations: {results['total_iterations']}")
        logger.info(f"Final rank: {results['final_rank']}")
        logger.info(f"Target met: {results['target_met']}")

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(quick_test())