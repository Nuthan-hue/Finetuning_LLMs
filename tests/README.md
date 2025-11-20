# Test Suite - Kaggle Multi-Agent System

Organized test files for testing individual phases and full workflows.

## Quick Test: Phase 3 Data Analysis

**COMPREHENSIVE TEST - Run this first:**
```bash
python tests/test_phase_3_data_analysis.py
```

This test covers:
- ✅ File identification (train/test/submission)
- ✅ Data modality detection  
- ✅ Target column identification
- ✅ Feature analysis
- ✅ Data quality assessment
- ✅ Preprocessing recommendations
- ✅ Context propagation
- ✅ Artifact validation

**API Usage**: 2-3 LLM calls (~2,000 tokens)

## Test Organization

```
tests/
├── README.md                              # This file
├── test_phase_3_data_analysis.py          # COMPREHENSIVE Phase 3 test
└── (future: other phase tests)
```

## How to Run

```bash
# From project root
cd /Volumes/SD_Card/Finetuning_LLMs

# Run Phase 3 comprehensive test
python tests/test_phase_3_data_analysis.py
```

## Expected Output

```
======================================================================
  PHASE 3: DATA ANALYSIS - COMPREHENSIVE TEST
======================================================================
✅ Train file identified: train.csv
✅ Test file identified: test.csv
✅ Data modality detected: tabular
✅ Target column identified: Survived
✅ Feature types analyzed
✅ Preprocessing recommendations provided
✅ ALL CHECKS PASSED!
```

## Caching

- **First run**: Executes Phases 1-3
- **Second run**: Uses cached results from Phases 1-2, only runs Phase 3

Saves 1-2 API calls per re-run!

## When API Limits Hit

Wait for quota reset:
- **Groq**: Daily limit resets at midnight UTC
- **Gemini**: Per-minute limit resets after ~15-20 seconds

Check quota:
- Groq: https://console.groq.com/settings/billing
- Gemini: https://ai.dev/usage

---

**Ready to Test!** Run `python tests/test_phase_3_data_analysis.py`
