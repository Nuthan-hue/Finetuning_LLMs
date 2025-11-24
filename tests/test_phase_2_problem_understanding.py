#!/usr/bin/env python3
"""
Test Phase 2: Problem Understanding
Tests the ProblemUnderstandingAgent's ability to analyze competition objectives.
"""
import asyncio
import sys
import json
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
async def test_phase_2_problem_understanding(competition_name: str = "titanic"):
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

    # Setup test cache directory
    cache_file = Path("data") / competition_name / "test" / "test_phase2_cache.json"

    # Check if we already have cached data
    if cache_file.exists():
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {
            "competition_name": competition_name,
            "data_path": f"data/{competition_name}"
        }
        context.update(cached)

        # Validate cached outputs
        assert "problem_understanding" in context, "Missing problem_understanding in cache"
        assert "competition_type" in context["problem_understanding"], "Missing competition_type in cache"
        assert "evaluation_metric" in context["problem_understanding"], "Missing evaluation_metric in cache"

        print("\n✅ Phase 2 PASSED (cached)")
        print(f"   Task: {context['problem_understanding'].get('competition_type')}")
        print(f"   Metric: {context['problem_understanding'].get('evaluation_metric')}")
        return True

    # No cache - run fresh
    orchestrator = MockOrchestrator()
    context = {
        "competition_name": competition_name,
        "data_path": f"data/{competition_name}"
    }

    try:
        print("\n🧠 Running Phase 2: Problem Understanding...")
        context = await run_problem_understanding(orchestrator, context)

        # Validate outputs
        assert "problem_understanding" in context, "Missing problem_understanding in context"
        assert "competition_type" in context["problem_understanding"], "Missing competition_type"
        assert "evaluation_metric" in context["problem_understanding"], "Missing evaluation_metric"

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "problem_understanding": context["problem_understanding"],
                "overview_text": context.get("overview_text")
            }, f, indent=2)

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
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_2_problem_understanding(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())