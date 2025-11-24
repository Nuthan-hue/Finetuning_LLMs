#!/usr/bin/env python3
"""
Test Phase 1: Data Collection
Tests the DataCollector agent's ability to fetch and analyze Kaggle competition data.
"""
import asyncio
import sys
import json
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_collector import DataCollector
from src.agents.orchestrator.phases import run_data_collection


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        self.data_collector = DataCollector()


@pytest.mark.asyncio
async def test_phase_1_data_collection(competition_name: str = "titanic"):
    """Test Phase 1: Data Collection"""

    # Setup cache in test/
    cache_file = Path("data") / competition_name / "test" / "test_phase1_cache.json"

    # Check if we already have this data
    if cache_file.exists():
        print("Already has this data")
        with open(cache_file, 'r') as f:
            context = json.load(f)

        # Assertions
        assert "data_path" in context
        assert "files" in context
        assert Path(context["data_path"]).exists()

        print(f"✅ Phase 1 passed (cached)")
        return True

    # No cache - run data collection
    orchestrator = MockOrchestrator()
    context = {"competition_name": competition_name}

    context = await run_data_collection(orchestrator, context)

    # Assertions
    assert "data_path" in context
    assert "files" in context
    assert Path(context["data_path"]).exists()

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({"data_path": context["data_path"], "files": context["files"]}, f, indent=2)

    print(f"✅ Phase 1 passed")
    return True


async def main():
    """Run Phase 1 test"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"

    try:
        await test_phase_1_data_collection(competition_name)
        print("✅ TEST PASSED")
        return 0
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))