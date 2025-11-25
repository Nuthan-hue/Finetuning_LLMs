#!/usr/bin/env python3
"""
Test Optimization Loop
Tests the complete optimization loop functionality including cache clearing,
phase selection, and iteration management.
"""
import asyncio
import sys
import json
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_optimization_loop(competition_name: str = "titanic", target_percentile: float = 0.20, max_iterations: int = 3):
    """
    Test the optimization loop functionality.

    Tests:
    - Iteration 1: Runs all phases 1-10
    - Iteration 2+: Reads phases_to_rerun from memory and runs only those phases
    - Memory accumulation across iterations
    - Target achievement detection
    - Cache management

    Args:
        competition_name: Competition to test with
        target_percentile: Target percentile (e.g., 0.20 for top 20%)
        max_iterations: Maximum iterations to test
    """
    print("\n" + "=" * 70)
    print("OPTIMIZATION LOOP - TEST")
    print("=" * 70)
    print(f"Competition: {competition_name}")
    print(f"Target: Top {target_percentile * 100:.0f}%")
    print(f"Max Iterations: {max_iterations}")
    print("=" * 70)

    test_cache_dir = Path("data") / competition_name / "test"
    memory_file = Path("data") / competition_name / "agent_memory.json"

    # Track phases run in each iteration
    iterations_log = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration}")
        print("=" * 70)

        # Determine which phases to run
        if iteration == 1:
            phases_to_run = list(range(1, 11))
            print(f"   First iteration: Running all phases {phases_to_run}")
        else:
            # Read from Phase 10 strategy
            phase10_cache = test_cache_dir / "test_phase10_cache.json"

            if not phase10_cache.exists():
                print(f"   ❌ Phase 10 cache not found, cannot determine phases to rerun")
                return False

            try:
                with open(phase10_cache, 'r') as f:
                    phase10_data = json.load(f)
                    strategy = phase10_data.get("optimization_strategy")

                    if not strategy:
                        print(f"   ⚠️  No optimization strategy found")
                        phases_to_run = [5, 7, 8, 9, 10]
                    else:
                        phases_to_run = strategy.get("phases_to_rerun", [5, 7, 8, 9])
                        # Always add Phase 10 if not present
                        if 10 not in phases_to_run:
                            phases_to_run.append(10)
                        phases_to_run = sorted(phases_to_run)

                print(f"   Strategy: {strategy.get('action') if strategy else 'N/A'}")
                print(f"   Phases to rerun: {phases_to_run}")

            except Exception as e:
                print(f"   ❌ Error reading Phase 10 cache: {e}")
                return False

        # Clear caches for phases that need to rerun
        if iteration > 1:
            print(f"\n   🗑️  Clearing caches for phases: {phases_to_run}")
            for phase in phases_to_run:
                cache_file = test_cache_dir / f"test_phase{phase}_cache.json"
                if cache_file.exists():
                    cache_file.unlink()
                    print(f"      ✓ Cleared phase {phase} cache")

        # Run each phase
        print(f"\n   🔄 Running phases...")
        iteration_results = {
            "iteration": iteration,
            "phases_run": phases_to_run,
            "phase_results": {}
        }

        for phase in phases_to_run:
            # Find test file for this phase
            test_files = list(Path("tests").glob(f"test_phase_{phase}_*.py"))

            if not test_files:
                print(f"   ❌ No test file found for phase {phase}")
                iteration_results["phase_results"][phase] = "not_found"
                continue

            test_file = test_files[0]
            print(f"   ▶️  Phase {phase}: {test_file.name}")

            try:
                # Phase 8 requires user input for submission - auto-answer "yes"
                if phase == 8:
                    result = subprocess.run(
                        [sys.executable, str(test_file), competition_name],
                        input="yes\n",  # Auto-submit to Kaggle
                        capture_output=True,
                        text=True,
                        timeout=180
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, str(test_file), competition_name],
                        capture_output=True,
                        text=True,
                        timeout=180
                    )

                if result.returncode == 0:
                    print(f"      ✅ Phase {phase} passed")
                    iteration_results["phase_results"][phase] = "passed"
                else:
                    print(f"      ❌ Phase {phase} failed")
                    print(f"         Error: {result.stderr[:200]}")
                    iteration_results["phase_results"][phase] = "failed"
                    return False

            except subprocess.TimeoutExpired:
                print(f"      ❌ Phase {phase} timed out")
                iteration_results["phase_results"][phase] = "timeout"
                return False
            except Exception as e:
                print(f"      ❌ Phase {phase} error: {e}")
                iteration_results["phase_results"][phase] = "error"
                return False

        # Check iteration results
        print(f"\n   📊 Iteration {iteration} Summary:")

        # Load Phase 9 results
        phase9_cache = test_cache_dir / "test_phase9_cache.json"
        if phase9_cache.exists():
            with open(phase9_cache, 'r') as f:
                phase9_data = json.load(f)
                iteration_results["evaluation"] = phase9_data
                print(f"      Rank: {phase9_data.get('current_rank', 'N/A')}")
                print(f"      Percentile: {phase9_data.get('current_percentile', 'N/A')}")
                print(f"      Meets Target: {phase9_data.get('meets_target', False)}")

                # Check if target achieved
                if phase9_data.get('meets_target', False):
                    print(f"\n{'=' * 70}")
                    print("🎉 TARGET ACHIEVED!")
                    print("=" * 70)
                    print(f"   Reached target in {iteration} iteration(s)")
                    print(f"   Competition: {competition_name}")
                    print("=" * 70)

                    iterations_log.append(iteration_results)
                    _display_final_summary(iterations_log, memory_file)
                    return True

        iterations_log.append(iteration_results)

    # Max iterations reached
    print(f"\n{'=' * 70}")
    print("⚠️  MAX ITERATIONS REACHED")
    print("=" * 70)
    print(f"   Completed {max_iterations} iterations without reaching target")

    _display_final_summary(iterations_log, memory_file)

    # Still return True if all phases passed (test succeeded even if target not met)
    return True


def _display_final_summary(iterations_log: list, memory_file: Path):
    """Display final summary of all iterations."""
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    for iteration_result in iterations_log:
        iteration = iteration_result["iteration"]
        phases = iteration_result["phases_run"]
        evaluation = iteration_result.get("evaluation", {})

        print(f"\nIteration {iteration}:")
        print(f"   Phases run: {phases}")
        if evaluation:
            print(f"   Percentile: {evaluation.get('current_percentile', 'N/A')}")
            print(f"   Rank: {evaluation.get('current_rank', 'N/A')}")

    # Load memory and show best attempt
    if memory_file.exists():
        try:
            with open(memory_file, 'r') as f:
                memory = json.load(f)
                attempts = memory.get("attempts", [])

                if attempts:
                    best = min(attempts, key=lambda x: x.get("percentile", 1.0))
                    print(f"\n   Best Result:")
                    print(f"      Iteration: {best.get('iteration')}")
                    print(f"      Model: {best.get('model')}")
                    print(f"      Percentile: {best.get('percentile')}")
                    print(f"      Rank: {best.get('rank')}")
        except Exception as e:
            print(f"   ⚠️  Could not read memory: {e}")

    print("=" * 70)


async def main():
    """Run optimization loop test."""
    if len(sys.argv) < 2:
        print("Usage: python test_optimization_loop.py <competition_name> [target_percentile] [max_iterations]")
        print("Example: python test_optimization_loop.py titanic 0.20 3")
        sys.exit(1)

    competition_name = sys.argv[1]
    target_percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    success = await test_optimization_loop(competition_name, target_percentile, max_iterations)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())