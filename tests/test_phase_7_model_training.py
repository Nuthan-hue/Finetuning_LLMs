#!/usr/bin/env python3
"""
Test Phase 7: Model Training
Tests the ModelTrainer's ability to train and evaluate models.
"""
import asyncio
import sys
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


async def test_phase_7_model_training():
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

    orchestrator = MockOrchestrator()
    context = {
        "competition_name": "titanic",
        "data_path": "data/titanic",
        "featured_data_path": "data/titanic",
        "target_column": "Survived",
        "execution_plan": {
            "models_to_try": [
                {"model": "lightgbm", "config": {}}
            ]
        }
    }

    try:
        print("\n🏋️ Running Phase 7: Model Training...")
        print("   (This may take a few minutes...)")

        context = await asyncio.wait_for(
            run_model_training(orchestrator, context),
            timeout=300  # 5 minutes
        )

        # Validate outputs
        assert "model_path" in context, "Missing model_path in context"
        assert "model_type" in context, "Missing model_type in context"
        assert "training_results" in context, "Missing training_results in context"

        # Verify model file exists
        if context.get("model_path"):
            model_path = Path(context["model_path"])
            assert model_path.exists(), f"Model file doesn't exist: {model_path}"

        print("\n✅ Phase 7 PASSED")
        print(f"   Model type: {context.get('model_type')}")
        print(f"   Model path: {context.get('model_path')}")
        print(f"   CV Score: {context.get('cv_score', 'N/A')}")

        return True

    except asyncio.TimeoutError:
        print("\n❌ Phase 7 TIMED OUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ Phase 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Phase 7 test"""
    success = await test_phase_7_model_training()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())