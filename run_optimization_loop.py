#!/usr/bin/env python3
"""
Optimization Loop Script

This script orchestrates the iterative optimization process:
1. Runs phases based on iteration number (1-10 for first iteration, then phases_to_rerun)
2. Uses test cache for efficiency (cheap API calls)
3. Prompts user before submission (Phase 8)
4. Reads optimization strategy from memory (Phase 10)
5. Loops until target achieved or max iterations reached

Usage:
    python run_optimization_loop.py <competition_name> <target_percentile> [max_iterations]

Example:
    python run_optimization_loop.py titanic 0.20 5
"""
import asyncio
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


class OptimizationLoop:
    """Manages the optimization loop for Kaggle competitions."""

    def __init__(self, competition_name: str, target_percentile: float, max_iterations: int = 10):
        """
        Initialize optimization loop.

        Args:
            competition_name: Name of the Kaggle competition
            target_percentile: Target percentile to achieve (e.g., 0.20 for top 20%)
            max_iterations: Maximum number of iterations to attempt
        """
        self.competition_name = competition_name
        self.target_percentile = target_percentile
        self.max_iterations = max_iterations
        self.test_cache_dir = Path("data") / competition_name / "test"
        self.memory_file = Path("data") / competition_name / "agent_memory.json"

    def load_memory(self) -> Dict[str, Any]:
        """Load agent memory from disk."""
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {
            "competition_name": self.competition_name,
            "attempts": [],
            "current_optimization": None,
            "performance_trend": None,
            "models_tried": []
        }

    def get_phases_to_run(self, iteration: int) -> List[int]:
        """
        Determine which phases to run based on iteration number.

        Args:
            iteration: Current iteration number

        Returns:
            List of phase numbers to run
        """
        if iteration == 1:
            # First iteration: Run all phases 1-10
            return list(range(1, 11))
        else:
            # Subsequent iterations: Read from memory's optimization strategy
            memory = self.load_memory()
            optimization = memory.get("current_optimization")

            if not optimization:
                print("   ⚠️  No optimization strategy found in memory")
                print("   Using default: [5, 7, 8, 9, 10]")
                return [5, 7, 8, 9, 10]

            phases_to_rerun = optimization.get("phases_to_rerun", [5, 7, 8, 9])
            # Always add Phase 10 if not present (to check if target achieved)
            if 10 not in phases_to_rerun:
                phases_to_rerun.append(10)

            return sorted(phases_to_rerun)

    def clear_phase_caches(self, phases: List[int]):
        """
        Clear test caches for specific phases so they rerun.

        Args:
            phases: List of phase numbers whose caches should be cleared
        """
        print(f"   🗑️  Clearing caches for phases: {phases}")

        for phase in phases:
            cache_file = self.test_cache_dir / f"test_phase{phase}_cache.json"
            if cache_file.exists():
                cache_file.unlink()
                print(f"      ✓ Cleared phase {phase} cache")

    def run_phase_test(self, phase: int) -> bool:
        """
        Run a single phase test.

        Args:
            phase: Phase number to run

        Returns:
            True if phase passed, False otherwise
        """
        test_file = Path(f"tests/test_phase_{phase}_*.py")
        test_files = list(Path("tests").glob(f"test_phase_{phase}_*.py"))

        if not test_files:
            print(f"   ❌ No test file found for phase {phase}")
            return False

        test_file = test_files[0]
        print(f"   ▶️  Running {test_file.name}...")

        try:
            result = subprocess.run(
                [sys.executable, str(test_file), self.competition_name],
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"   ❌ Error running phase {phase}: {e}")
            return False

    def check_if_target_achieved(self) -> bool:
        """
        Check if target percentile has been achieved.

        Returns:
            True if target achieved, False otherwise
        """
        # Check Phase 9 cache for evaluation results
        phase9_cache = self.test_cache_dir / "test_phase9_cache.json"

        if not phase9_cache.exists():
            return False

        try:
            with open(phase9_cache, 'r') as f:
                phase9_data = json.load(f)
                return phase9_data.get("meets_target", False)
        except Exception:
            return False

    def display_iteration_plan(self, iteration: int, phases: List[int]):
        """
        Display the plan for this iteration.

        Args:
            iteration: Current iteration number
            phases: Phases to run
        """
        print("\n" + "=" * 70)
        print(f"ITERATION {iteration} PLAN")
        print("=" * 70)
        print(f"   Competition: {self.competition_name}")
        print(f"   Target: Top {self.target_percentile * 100:.0f}%")
        print(f"   Phases to run: {phases}")

        if iteration > 1:
            # Show optimization strategy
            memory = self.load_memory()
            optimization = memory.get("current_optimization")

            if optimization:
                print(f"\n   📊 Optimization Strategy:")
                print(f"      Action: {optimization.get('action', 'N/A')}")
                print(f"      Reasoning: {optimization.get('reasoning', 'N/A')[:80]}...")
                print(f"      Confidence: {optimization.get('confidence', 'N/A')}")

        print("=" * 70)

    def display_iteration_summary(self, iteration: int):
        """
        Display summary after iteration completes.

        Args:
            iteration: Current iteration number
        """
        print("\n" + "=" * 70)
        print(f"ITERATION {iteration} SUMMARY")
        print("=" * 70)

        # Load Phase 9 results
        phase9_cache = self.test_cache_dir / "test_phase9_cache.json"
        if phase9_cache.exists():
            with open(phase9_cache, 'r') as f:
                phase9_data = json.load(f)
                print(f"   Current Rank: {phase9_data.get('current_rank', 'N/A')}")
                print(f"   Current Percentile: {phase9_data.get('current_percentile', 'N/A')}")
                print(f"   Meets Target: {phase9_data.get('meets_target', False)}")

        # Load Phase 10 strategy
        phase10_cache = self.test_cache_dir / "test_phase10_cache.json"
        if phase10_cache.exists():
            with open(phase10_cache, 'r') as f:
                phase10_data = json.load(f)
                strategy = phase10_data.get("optimization_strategy")
                if strategy:
                    print(f"\n   Next Strategy: {strategy.get('action', 'N/A')}")
                    print(f"   Phases to rerun: {strategy.get('phases_to_rerun', 'N/A')}")

        print("=" * 70)

    async def run(self):
        """
        Run the optimization loop.

        Returns:
            True if target achieved, False otherwise
        """
        print("\n" + "=" * 70)
        print("KAGGLE OPTIMIZATION LOOP")
        print("=" * 70)
        print(f"Competition: {self.competition_name}")
        print(f"Target: Top {self.target_percentile * 100:.0f}%")
        print(f"Max Iterations: {self.max_iterations}")
        print("=" * 70)

        for iteration in range(1, self.max_iterations + 1):
            # Determine which phases to run
            phases_to_run = self.get_phases_to_run(iteration)

            # Display iteration plan
            self.display_iteration_plan(iteration, phases_to_run)

            # Clear caches for phases that need to rerun
            if iteration > 1:
                self.clear_phase_caches(phases_to_run)

            # Run each phase
            print(f"\n🔄 Running Iteration {iteration}...")
            all_passed = True

            for phase in phases_to_run:
                success = self.run_phase_test(phase)
                if not success:
                    print(f"\n   ❌ Phase {phase} failed!")
                    all_passed = False
                    break

            if not all_passed:
                print(f"\n❌ Iteration {iteration} FAILED")
                print("   Fix errors and try again")
                return False

            # Display iteration summary
            self.display_iteration_summary(iteration)

            # Check if target achieved
            if self.check_if_target_achieved():
                print("\n" + "=" * 70)
                print("🎉 TARGET ACHIEVED!")
                print("=" * 70)
                print(f"   Reached target in {iteration} iteration(s)")
                print(f"   Competition: {self.competition_name}")
                print("=" * 70)
                return True

            # Not achieved yet, prepare for next iteration
            if iteration < self.max_iterations:
                print(f"\n   ⏭️  Target not achieved, preparing iteration {iteration + 1}...")
            else:
                print(f"\n   ⚠️  Max iterations ({self.max_iterations}) reached")

        # Max iterations reached without achieving target
        print("\n" + "=" * 70)
        print("⚠️  MAX ITERATIONS REACHED")
        print("=" * 70)
        print(f"   Completed {self.max_iterations} iterations without reaching target")
        print(f"   Competition: {self.competition_name}")

        # Show best attempt
        memory = self.load_memory()
        attempts = memory.get("attempts", [])
        if attempts:
            best = min(attempts, key=lambda x: x.get("percentile", 1.0))
            print(f"\n   Best Result:")
            print(f"      Iteration: {best.get('iteration')}")
            print(f"      Model: {best.get('model')}")
            print(f"      Percentile: {best.get('percentile')}")
            print(f"      Rank: {best.get('rank')}")

        print("=" * 70)
        return False


async def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python run_optimization_loop.py <competition_name> <target_percentile> [max_iterations]")
        print("Example: python run_optimization_loop.py titanic 0.20 5")
        sys.exit(1)

    competition_name = sys.argv[1]
    target_percentile = float(sys.argv[2])
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    loop = OptimizationLoop(competition_name, target_percentile, max_iterations)
    success = await loop.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())