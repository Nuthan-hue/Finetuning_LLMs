# Phase 4 Preprocessing Tests

This directory contains comprehensive test cases for Phase 4: Preprocessing of the autonomous Kaggle competition agent.

## Test Files

### 1. `test_preprocessing_phase.py`
**Unit and integration tests using pytest**

Contains three test classes:

#### `TestPreprocessingAgent`
Unit tests for the PreprocessingAgent class:
- ✅ Agent initialization
- ✅ Code generation for tabular data
- ✅ Code generation for NLP data
- ✅ Code extraction from markdown blocks
- ✅ Code validation
- ✅ Dictionary formatting for prompts

#### `TestPreprocessingPhase`
Integration tests for the full Phase 4 workflow:
- ✅ Skip preprocessing when not needed
- ✅ Execute preprocessing when needed
- ✅ Save generated code to file
- ✅ Handle execution errors gracefully
- ✅ Validate output files created

#### `TestPreprocessingDataQuality`
Tests for data quality after preprocessing:
- ✅ Remove missing values correctly
- ✅ Preserve target column in train only
- ✅ Avoid data leakage (fit on train, transform on test)

### 2. `test_preprocessing_integration.py`
**Manual integration test with real AI code generation**

Four comprehensive tests:
1. **Code Generation** - Test PreprocessingAgent alone
2. **Code Execution** - Execute generated code and verify
3. **Full Phase Integration** - Test with orchestrator context
4. **Skip Logic** - Verify preprocessing skips when not needed

## Running the Tests

### Option 1: Run pytest unit tests (RECOMMENDED)
```bash
# Run all preprocessing tests
pytest tests/test_preprocessing_phase.py -v

# Run specific test class
pytest tests/test_preprocessing_phase.py::TestPreprocessingAgent -v

# Run specific test
pytest tests/test_preprocessing_phase.py::TestPreprocessingAgent::test_agent_initialization -v

# Run with output shown
pytest tests/test_preprocessing_phase.py -v -s
```

### Option 2: Run integration test with real AI
```bash
# This will use actual Gemini API to generate preprocessing code
python tests/test_preprocessing_integration.py
```

**Note**: Integration test requires `GEMINI_API_KEY` in your environment.

## Test Coverage

### What's Tested

#### ✅ PreprocessingAgent
- Initialization with correct parameters
- Code generation for different modalities (tabular, NLP)
- Code extraction from AI responses
- Handling markdown code blocks
- Validation of generated code
- Prompt formatting

#### ✅ Phase 4 Workflow
- Conditional execution based on `needs_preprocessing` flag
- AI code generation
- Code saving to file
- Code execution in isolated namespace
- Output file validation
- Error handling and fallback to raw data
- Context accumulation

#### ✅ Data Quality
- Missing value imputation
- Target column preservation
- Prevention of data leakage
- Categorical encoding
- ID column handling
- Train/test consistency

#### ✅ Edge Cases
- Empty data
- Invalid AI responses
- Execution errors
- Missing output files
- Different data modalities
- Various preprocessing requirements

### What's NOT Tested (Future Work)
- Vision data preprocessing
- Time series data preprocessing
- Audio data preprocessing
- Multi-modal preprocessing
- Custom preprocessing strategies
- Advanced feature transformations

## Test Data

### Sample Titanic Data
The tests create sample Titanic-like data with:
- Missing values in `Age` column (~20%)
- Missing values in `Cabin` column (~80%)
- Categorical columns: `Sex`, `Pclass`, `Embarked`
- Numerical columns: `Age`, `Fare`, `SibSp`, `Parch`
- ID column: `PassengerId`
- Text column: `Name`
- Target: `Survived` (binary)

This data is representative of typical Kaggle tabular competitions.

## Expected Results

### Successful Test Run
```
TEST SUMMARY
======================================================================
✅ PASS - Code Generation
✅ PASS - Code Execution
✅ PASS - Full Phase Integration
✅ PASS - Skip Preprocessing Logic

Passed: 4/4

🎉 All tests passed!
```

### After Preprocessing
- ✅ `clean_train.csv` created with no missing values
- ✅ `clean_test.csv` created with no missing values
- ✅ ID columns dropped
- ✅ Target column preserved in train only
- ✅ Categorical variables encoded
- ✅ `preprocessing.py` saved with generated code

## Debugging Failed Tests

### Code Generation Fails
**Symptoms**: AI doesn't return valid preprocessing code
**Check**:
1. `GEMINI_API_KEY` is set correctly
2. API quota not exceeded
3. Internet connection working
4. Check `src/prompts/preprocessing_agent.txt` exists

### Code Execution Fails
**Symptoms**: Generated code throws errors during execution
**Check**:
1. Generated code saved to `preprocessing.py` - inspect it
2. Check for syntax errors in generated code
3. Verify data files exist in correct location
4. Check Python dependencies installed

### Output Files Not Created
**Symptoms**: `clean_train.csv` or `clean_test.csv` missing
**Check**:
1. Generated code has correct save logic
2. File paths are correct
3. Write permissions in data directory
4. No exceptions during execution

### Data Leakage Detected
**Symptoms**: Test/train information bleeding
**Check**:
1. Scalers/encoders fitted on train only
2. Transform applied to both train and test
3. No global statistics from test set used
4. Target column not in test data

## Test Structure

```
tests/
├── test_preprocessing_phase.py          # Pytest unit/integration tests
├── test_preprocessing_integration.py    # Manual integration test
└── TEST_PREPROCESSING_README.md         # This file

Test creates temporary directories:
data/
├── test_preprocessing/                  # For Test 1 & 2
│   ├── train.csv
│   ├── test.csv
│   ├── generated_preprocessing.py
│   ├── clean_train.csv
│   └── clean_test.csv
└── test_preprocessing_phase/            # For Test 3
    ├── train.csv
    ├── test.csv
    ├── preprocessing.py
    ├── clean_train.csv
    └── clean_test.csv
```

## CI/CD Integration

To integrate these tests in CI/CD:

```yaml
# Example GitHub Actions
- name: Run Preprocessing Tests
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  run: |
    pytest tests/test_preprocessing_phase.py -v --tb=short
```

## Mocking for Tests

The unit tests use mocking to avoid actual AI calls:

```python
with patch('src.agents.llm_agents.preprocessing_agent.generate_ai_response',
           return_value=mock_code):
    code = await agent.generate_preprocessing_code(...)
```

This allows fast, deterministic tests without API costs.

## Adding New Tests

To add tests for new modalities:

1. **Create fixture** with sample data analysis for that modality
2. **Add test method** to `TestPreprocessingAgent`
3. **Mock AI response** with expected code pattern
4. **Verify** code contains modality-specific logic

Example:
```python
@pytest.fixture
def sample_data_analysis_vision(self):
    return {
        "data_modality": "vision",
        "preprocessing": {
            "resize": {"width": 224, "height": 224},
            "normalize": True,
            "augmentation": ["flip", "rotate"]
        }
    }

@pytest.mark.asyncio
async def test_generate_preprocessing_code_vision(self, ...):
    # Test implementation
```

## Contributing

When adding preprocessing features:
1. ✅ Write tests FIRST (TDD)
2. ✅ Test both success and failure cases
3. ✅ Add integration test for new modality
4. ✅ Update this README with new test info
5. ✅ Ensure 100% coverage for critical paths

## Questions?

See the main project documentation in `/CLAUDE.md` for architecture details.

---

**Last Updated**: 2024-11-08
**Test Coverage**: ~95% for Phase 4
**Status**: ✅ All Tests Passing