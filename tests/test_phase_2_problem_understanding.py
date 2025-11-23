#!/usr/bin/env python3
"""
Test Phase 2: Problem Understanding
Tests the ProblemUnderstandingAgent's ability to analyze competition objectives.
"""
import asyncio
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_agents import ProblemUnderstandingAgent
from src.agents.orchestrator.phases import run_problem_understanding


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        self.problem_understanding_agent = ProblemUnderstandingAgent()

@pytest.mark.asyncio
async def test_phase_2_problem_understanding():
    """
    Test Phase 2: Problem Understanding

    Tests:
    - Analyze competition description
    - Identify problem type (classification, regression, etc.)
    - Extract evaluation metric
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 2: PROBLEM UNDERSTANDING - TEST")
    print("=" * 70)

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": "titanic",
        "data_path": "data/titanic"  # Simulated from Phase 1
    }

    try:
        print("\n🧠 Running Phase 2: Problem Understanding...")
        context = await run_problem_understanding(orchestrator, context)

        # Validate outputs
        assert "problem_understanding" in context, "Missing problem_understanding in context"
        assert "competition_type" in context["problem_understanding"], "Missing competition_type"
        assert "evaluation_metric" in context["problem_understanding"], "Missing evaluation_metric"

        print("\n✅ Phase 2 PASSED")
        print(f"   Task: {context['problem_understanding'].get('competition_type')}")
        print(f"   Metric: {context['problem_understanding'].get('evaluation_metric')}")
        print(f"   Description: {context['problem_understanding'].get('description', 'N/A')[:100]}...")

        return True

    except Exception as e:
        print(f"\n❌ Phase 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 2 test"""
    success = await test_phase_2_problem_understanding()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())