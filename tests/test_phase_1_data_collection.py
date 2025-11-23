#!/usr/bin/env python3
"""
Test Phase 1: Data Collection
Tests the DataCollector agent's ability to fetch and analyze Kaggle competition data.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_collector import DataCollector
from src.agents.orchestrator.phases import run_data_collection


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        self.data_collector = DataCollector()


async def test_phase_1_data_collection():
    """
    Test Phase 1: Data Collection

    Tests:
    - Download competition data from Kaggle
    - Identify available files
    - Basic data analysis
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION - TEST")
    print("=" * 70)

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": "titanic"
    }

    try:
        print("\n📦 Running Phase 1: Data Collection...")
        context = await run_data_collection(orchestrator, context)

        # Validate outputs
        assert "data_path" in context, "Missing data_path in context"
        assert "files" in context, "Missing files in context"
        assert Path(context["data_path"]).exists(), f"Data path doesn't exist: {context['data_path']}"

        print("\n✅ Phase 1 PASSED")
        print(f"   Data path: {context['data_path']}")
        print(f"   Files found: {len(context['files'])}")
        for file_info in context['files']:
            print(f"     - {file_info['name']} ({file_info.get('size', 'N/A')})")

        return True

    except Exception as e:
        print(f"\n❌ Phase 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 1 test"""
    success = await test_phase_1_data_collection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
