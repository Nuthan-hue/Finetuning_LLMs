#!/usr/bin/env python3
"""
Test Phase 7: Model Training
Tests the ModelTrainer's ability to train and evaluate models.
"""
import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.model_trainer import ModelTrainer
from src.agents.orchestrator.phases import run_model_training


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.iteration = 1


async def test_phase_7_model_training(competition_name: str = "titanic"):
    """
    Test Phase 7: Model Training

    Tests:
    - Train model on prepared data
    - Cross-validation
    - Model saving
    - Performance metrics
    - Context propagation
    """
    print("\n" + "=" * 70)
    print("PHASE 7: MODEL TRAINING - TEST")
    print("=" * 70)

    # Setup test cache directory
    test_cache_dir = Path("data") / competition_name / "test"
    cache_file = test_cache_dir / "test_phase7_cache.json"

    # Check cache
    if cache_file.exists():
        print("📦 Loading from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        context = {"competition_name": competition_name}
        context.update(cached)

        assert "model_path" in context, "Missing model_path in cache"
        print("\n✅ Phase 7 PASSED (cached)")
        print(f"   Model type: {context.get('model_type')}")
        print(f"   CV Score: {context.get('cv_score', 'N/A')}")
        return True

    # Load dependencies
    phase3_cache = test_cache_dir / "test_phase3_cache.json"
    phase5_cache = test_cache_dir / "test_phase5_cache.json"
    phase6_cache = test_cache_dir / "test_phase6_cache.json"

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": competition_name,
        "data_path": f"data/{competition_name}",
    }

    # Load Phase 3 data
    if phase3_cache.exists():
        with open(phase3_cache, 'r') as f:
            context.update(json.load(f))
    else:
        context["target_column"] = "Survived"
        context["data_files"] = {"train_file": "train.csv", "test_file": "test.csv"}
        context["data_analysis"] = {"data_modality": "tabular"}

    # Load Phase 5 data
    if phase5_cache.exists():
        with open(phase5_cache, 'r') as f:
            context.update(json.load(f))
    else:
        context["execution_plan"] = {"models_to_try": [{"model": "lightgbm", "config": {}}]}

    # Load Phase 6 data
    if phase6_cache.exists():
        with open(phase6_cache, 'r') as f:
            context.update(json.load(f))
    else:
        context["featured_data_path"] = context["data_path"]

    try:
        print("\n🏋️ Running Phase 7: Model Training...")
        print("   (This may take a few minutes...)")

        context = await asyncio.wait_for(
            run_model_training(orchestrator, context),
            timeout=300
        )

        assert "model_path" in context, "Missing model_path"
        assert "model_type" in context, "Missing model_type"

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "model_path": context["model_path"],
                "model_type": context["model_type"],
                "cv_score": context.get("cv_score"),
                "training_results": context.get("training_results")
            }, f, indent=2)

        print("\n✅ Phase 7 PASSED")
        print(f"   Model type: {context.get('model_type')}")
        print(f"   CV Score: {context.get('cv_score', 'N/A')}")
        return True

    except asyncio.TimeoutError:
        print("\n❌ Phase 7 TIMED OUT")
        return False
    except Exception as e:
        print(f"\n❌ Phase 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 7 test"""
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    success = await test_phase_7_model_training(competition_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())