"""
Test Script for Problem Understanding Agent
Demonstrates AI reading and understanding Kaggle competition BEFORE looking at data.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.llm_agents import ProblemUnderstandingAgent


async def test_problem_understanding(competition_name: str):
    """
    Test the Problem Understanding Agent on a competition.

    Args:
        competition_name: Name of Kaggle competition to analyze
    """
    print("=" * 80)
    print("PROBLEM UNDERSTANDING AGENT TEST")
    print("=" * 80)
    print(f"\nCompetition: {competition_name}")
    print("\nThis demonstrates the AI-first approach:")
    print("  1. AI reads competition information")
    print("  2. AI understands the problem WITHOUT looking at data")
    print("  3. AI provides comprehensive understanding and strategy")
    print("\nStarting analysis...\n")

    # Initialize agent
    agent = ProblemUnderstandingAgent()

    try:
        # Understand the competition
        understanding = await agent.understand_competition(competition_name)

        # Display results
        summary = agent.get_problem_summary(understanding)
        print(summary)

        # Display raw understanding for inspection
        print("\n" + "=" * 80)
        print("RAW AI UNDERSTANDING (JSON)")
        print("=" * 80)
        import json
        print(json.dumps(understanding, indent=2))

        return understanding

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


async def main():
    """Main test function."""
    # Default to Titanic competition
    competition = "titanic" if len(sys.argv) < 2 else sys.argv[1]

    print(f"\n🤖 Testing Problem Understanding Agent")
    print(f"📊 Competition: {competition}\n")

    # Run test
    understanding = await test_problem_understanding(competition)

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n✅ AI understood the problem as: {understanding['task_type']}")
    print(f"✅ Recommended metric: {understanding['evaluation_metric']}")
    print(f"✅ Confidence: {understanding.get('confidence', 'N/A')}")

    print("\n💡 Next Steps:")
    print("  1. This understanding will be used by Data Analysis Agent")
    print("  2. Planning Agent will create execution plan")
    print("  3. Workers will execute the plan")
    print("\nThis is the foundation of the problem-first approach!")


if __name__ == "__main__":
    asyncio.run(main())