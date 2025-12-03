# Phase 2 Notes + logs

This file tracks major updates, design decisions, and progress logs
for the production pipeline.

## 📅 Start Date
02 Dec 2025

---

## 🧱 Initial Setup

- Created `Phase2_Pipeline/` folder
- Added `pipeline/` module directory
- Added empty module files:
  - loader.py
  - schema.py
  - cleaner.py
  - eda.py
  - trainer.py

Notes:
- Phase 2 focuses on converting research notebooks into modular, production-grade code.
- Streamlit UI & monetization info will be added later.

---

---

## 📌 Module: loader.py — Completed (3/12/25)

### 🔹 Overview
The DataLoader module has been fully implemented as the first production-quality module of Phase 2. It converts the CSV-loading logic from Phase 1 notebooks into a robust, safe, reusable Python component.

This module ensures CSV files are handled professionally for real-world usage in the upcoming Streamlit UI and LLM-driven reasoning flow.

---

## Features Implemented

### Module Header & Imports
- Added clean docstring.
- Imported pandas, os, logging, typing.
- Set up module-level logging config.

### Class Skeleton
- Created `DataLoader` class.
- Added constructor storing `file_path`.
- Added placeholder load().

### Path & Extension Validation
- `_check_exists()` → ensures file exists.
- `_check_extension()` → ensures .csv extension.

###  Safe CSV Loading with Fallback Encodings
- Added `_safe_read_csv()` with:
  - UTF-8
  - ISO-8859-1
  - Latin-1  
- Prevents crashes from encoding issues.

###  Automatic Separator Detection
- `_detect_separator()` reads first 2KB of file.
- Detects: `,`, `;`, `\t`, `|`
- Ensures correct parsing before pandas loads.

###  File Size Protection
- `_check_file_size()` limits to 200MB by default.
- Prevents UI freezes and memory overuse.

###  Logging Integration
- All major actions log:
  - File checks  
  - Separator detection  
  - Encoding tries  
  - Success/failure messages  
- Useful for debugging, UI display, and future Pro-tier logs.

###  Metadata Return
- `load()` now returns `(df, metadata)`.
- Metadata includes:
  - file path  
  - file size  
  - rows, columns  
  - encoding used  
  - separator used  

###  Final Polishing
- Clean error handling via `_raise_error()`.
- Improved docstrings.
- Safer `load()` with exception wrapping.
- Added `load_df()` convenience method.

---

## 📊 Test Script Created: `test_loader.py`
A standalone test file verifies loader behavior:

- Loads real CSV  
- Prints shape  
- Prints metadata  
- Shows first 5 rows  
- Catches and prints errors cleanly  

This file sits **outside** the `pipeline/` folder.

---

## 📎 Next Steps
The next pipeline module to implement:

### **➡ schema.py — Automatic schema detection**

This will include:
- identifying numeric / categorical / datetime columns  
- detecting ID columns  
- detecting target column  
- integrating with loader metadata  

---



