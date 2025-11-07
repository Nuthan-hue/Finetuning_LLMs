# Development Notebook

**Date:** Nov 6, 2024

---

## 📝 Notes & TODOs

### Kaggle CLI Capabilities

**✅ CAN do:**
- `kaggle competitions files` - list files
- `kaggle competitions download` - download data
- `kaggle competitions submit` - submit predictions
- `kaggle competitions submissions` - our submissions
- `kaggle competitions leaderboard` - rankings

**❌ CANNOT do:**
- Get competition description/overview
- Get evaluation metric details
- Get problem statement

**URLs available:**
- `https://www.kaggle.com/competitions/{comp}/overview` - description
- `https://www.kaggle.com/competitions/{comp}/data` - data info
- `https://www.kaggle.com/competitions/{comp}/leaderboard` - rankings

**Current issue:** Phase 2 has no real competition info, AI just guesses from name

---

## 🔧 Refactoring Done (Nov 6)

- ✅ Fixed phase order (Data Collection → Problem Understanding)
- ✅ Accumulated context pattern (single dict)
- ✅ Removed duplicate DataAnalysisAgent call
- ✅ Added conditional phases (Preprocessing, Feature Engineering)
- ✅ Added Evaluation + Optimization phases
- ✅ Fixed import path in test file

---

## 🐛 Issues Found

1. Import conflict - TensorFlow `agents` package vs our package
   - Fixed with proper sys.path

2. Competition fetch wrong - was listing all competitions
   - Fixed to use exact name

3. gender_submission.csv missing - not always in downloads

---

## ⏳ Pending

- [ ] Test full 10-phase flow after API quota reset
- [ ] Run on real competition

---

## ✅ Done (Nov 6 - Web Scraping)

- Added web scraping to Phase 2 (Problem Understanding)
- Now scrapes `https://www.kaggle.com/competitions/{comp}/overview`
- AI gets REAL competition description instead of guessing!
- Uses BeautifulSoup + requests (already in requirements.txt)

---

## 💭 Questions & Ideas

- How to handle missing sample_submission.csv?

---

## 🤔 Design Questions

### Why So Many Layers?

**Example:** `phases.py → orchestrator → collector → kaggle_data`

**Theory (textbooks):**
- Separation of concerns
- Reusability
- Testability

**Reality:**
- Overcomplicated - hard to trace
- Harder to debug - jump between files
- More work for same result

**When layers are GOOD:** Abstraction provides real value
**When layers are BAD:** Middle layer adds nothing

**Simpler approach possible:** Direct imports instead of passing through orchestrator

---

### Agent Initialization Inconsistency

**Worker Agents** - Pre-initialized in orchestrator:
```python
# orchestrator.py __init__
self.data_collector = DataCollector(data_dir=data_dir)
self.model_trainer = ModelTrainer(models_dir=models_dir)
self.submission_agent = Submitter(submissions_dir=submissions_dir)
self.leaderboard_monitor = LeaderboardMonitor()

# Usage in phases.py
results = await orchestrator.data_collector.run(context)
```

**LLM Agents** - Created fresh in each phase:
```python
# phases.py
problem_agent = ProblemUnderstandingAgent()
understanding = await problem_agent.understand_competition(...)

# phases.py
data_agent = DataAnalysisAgent()
analysis = await data_agent.analyze_and_suggest(...)
```

**Why the difference?**

| Agent Type | Initialization | Reason |
|------------|---------------|---------|
| **Worker agents** | Once in orchestrator | Have persistent state (dirs, configs), called multiple times |
| **LLM agents** | Fresh in each phase | Stateless, called once per phase, no persistence needed |

**Is this good design?** 🤔

**Pros:**
- Worker agents maintain state across iterations
- LLM agents don't waste memory when not needed

**Cons:**
- **Inconsistent!** Confusing which pattern to follow
- Have to remember: "access worker agents via orchestrator, create LLM agents directly"
- Makes code harder to understand

**Better approach:**

**Option 1: ALL in orchestrator (consistent)**
```python
# orchestrator.py
self.problem_agent = ProblemUnderstandingAgent()
self.data_agent = DataAnalysisAgent()
...

# phases.py
understanding = await orchestrator.problem_agent.understand_competition(...)
```

**Option 2: ALL in phases (consistent)**
```python
# phases.py - create everything when needed
data_collector = DataCollector(data_dir=context["data_path"])
results = await data_collector.run(context)
```

**Current hybrid approach is confusing but works!**

---

_Add more notes as we go..._