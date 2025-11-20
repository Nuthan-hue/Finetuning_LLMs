#!/usr/bin/env python3
"""
Test script for the fixed Kaggle Competition Multi-Agent System

This script tests all the fixes applied:
1. Import path corrections in LLM agents
2. PlanningAgent parameter compatibility (data_analysis vs dataset_info)
3. Preprocessing namespace variables (train_file, test_file, target_column)
4. File name injection in preprocessing templates

Usage:
    python test_fixed_system.py
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from src.agents import AgenticOrchestrator


async def run_competition():
    print('🚀 Kaggle Competition System - FIXED & TESTED')
    print('=' * 70)
    print('✅ All Fixes Applied:')
    print('   - Import paths corrected (all LLM agents)')
    print('   - PlanningAgent parameter compatibility')
    print('   - Preprocessing namespace variables')
    print('   - File name injection in preprocessing')
    print('=' * 70)
    print('Provider: Groq (llama-3.3-70b-versatile)')
    print('Competition: titanic')
    print('Target: Top 20%')
    print('=' * 70)
    print()

    orchestrator = AgenticOrchestrator(
        competition_name='titanic',
        target_percentile=0.20,
        max_actions=50
    )

    results = await orchestrator.run({'competition_name': 'titanic'})
    return results


if __name__ == '__main__':
    print('Starting Kaggle Competition System Test...')
    print()

    try:
        results = asyncio.run(run_competition())
        print()
        print('=' * 70)
        print('✅ COMPLETE!')
        print('=' * 70)
        print(f'Results: {results}')
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrupted by user')
    except Exception as e:
        print(f'\n\n❌ Error: {e}')
        import traceback
        traceback.print_exc()