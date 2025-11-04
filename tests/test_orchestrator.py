"""
Test Orchestrator Agent - Full End-to-End Workflow
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.orchestrator import OrchestratorAgent


async def main():
    """Test full orchestration workflow"""
    print("=" * 70)
    print("TESTING FULL ORCHESTRATOR WORKFLOW")
    print("=" * 70)

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        competition_name="titanic",
        target_percentile=0.20,  # Aim for top 20%
        max_iterations=3  # Limit iterations for testing
    )

    print(f"\nConfiguration:")
    print(f"  Competition: titanic")
    print(f"  Target: Top 20%")
    print(f"  Max Iterations: 3")
    print()

    # Run full workflow
    results = await orchestrator.run({})

    print("\n" + "=" * 70)
    print("ORCHESTRATOR RESULTS")
    print("=" * 70)

    print(f"\nFinal Status: {results.get('final_status')}")
    print(f"Total Iterations: {results.get('iterations_completed')}")
    print(f"Target Met: {results.get('target_met')}")

    if 'best_submission' in results:
        best = results['best_submission']
        print(f"\nBest Submission:")
        print(f"  Score: {best.get('score')}")
        print(f"  Rank: {best.get('rank')}")
        print(f"  Percentile: {best.get('percentile'):.2%}" if best.get('percentile') else "  Percentile: N/A")

    print(f"\nWorkflow History:")
    for i, entry in enumerate(results.get('workflow_history', []), 1):
        print(f"  Iteration {i}: {entry}")

    print("\n" + "=" * 70)
    print("ORCHESTRATOR TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())