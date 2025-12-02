# CortexAI — Phase 2 Production Pipeline

Phase 2 converts the research notebooks from Phase 1 into a modular,
production-quality Python package.

This phase includes:

- Clean and testable pipeline modules
- A Streamlit UI for CSV → AutoML inference
- Optional LLM reasoning layer for insights
- Monetization-ready architecture (Pro features, limits, reports)
- Documentation + logging + model saving

## 📁 Folder Structure

```text

Phase2_Pipeline/
│
├── pipeline/
│ ├── loader.py # CSV loading and validation
│ ├── schema.py # Auto schema detection
│ ├── cleaner.py # Production cleaning engine
│ ├── eda.py # Automated EDA engine
│ ├── trainer.py # Model training + selection
│ └── init.py
│
├── README.md
└── notes.md

```

## 🎯 Goals of Phase 2

- Convert Phase 1 notebook logic into real `.py` modules
- Build a Streamlit UI for real usage
- Add LLM insight features (Pro tier)
- Architect the project to be monetizable at Phase 2 end
- Maintain clean engineering workflow
- Prepare the codebase for future deployment (Phase 4)

## 🧩 Status

- Folder structure created  
- Currently implementing `loader.py`  
- Modules will be fleshed out sequentially

## Upcoming Steps

1. Implement loader module
2. Implement schema detector module
3. Implement cleaner module
4. Implement EDA engine
5. Implement trainer engine
6. Build Streamlit UI
7. Add Pro features (LLM insights + PDF reports)
