# CortexAI — Phase 2 Production Pipeline

Phase 2 converts the research notebooks from Phase 1 into a modular,
production-quality Python package.

This phase includes:

- Clean and testable pipeline modules
- A Streamlit UI for CSV → AutoML inference
- Optional LLM reasoning layer for insights
- Documentation + logging + model saving

## 📁 Folder Structure

```text

Phase2_Pipeline/
│
├── pipeline/
│   ├── loader.py        # Safe CSV loading + validation
│   ├── schema.py        # Column type + ID + target detection
│   ├── cleaner.py       # Automated data cleaning engine
│   ├── eda.py           # (To build) Automated EDA module
│   ├── trainer.py       # (To build) ML model training + evaluation
│   └── __init__.py      # Makes this a Python package
│
├── test_loader.py       # Module tester (Loader)
├── test_schema.py       # Module tester (Schema)
├── test_cleaner.py      # Module tester (Cleaner)
│
├── README.md            # Phase 2 documentation + usage
└── notes.md             # Engineering notes & progress logs


```

## 🎯 Goals of Phase 2

- Convert Phase 1 notebook logic into real `.py` modules
- Build a Streamlit UI for real usage
- Add LLM insight features 
- Maintain clean engineering workflow
- Prepare the codebase for future deployment (Phase 4)

## 🧩 Status

- Folder structure created  
- Currently implementing `loader.py`  
- Modules will be fleshed out sequentially


