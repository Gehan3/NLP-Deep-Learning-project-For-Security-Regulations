# NLP Cybersecurity RAG Bot

Repository:

https://github.com/Boody700/Nlp_cybersecurity_Rag_bot

---

# Project Overview

This project aims to build a **Cybersecurity Retrieval-Augmented Generation (RAG) Assistant**.

The goal is to allow users to ask questions about cybersecurity standards and receive answers based on official documents such as:

* ISO/IEC standards
* PCI DSS
* Other cybersecurity frameworks like CBE

The project is being developed in multiple stages.

Currently, we completed the **document preprocessing pipeline**:

```
PDF Documents

        ↓

Text Extraction

        ↓

Parser

        ↓

JSON Export

        ↓

Cleaning

        ↓

(Next)
Chunking → Embeddings → Vector Database → RAG Chatbot
```

---

# Project Structure

```
Nlp_Proj/

│
├── data/
│
│   ├── raw/
│   │      Original PDF documents
│   │
│   ├── processed/
│   │      Parsed JSON files generated from PDFs
│   │
│   ├── cleaned/
│   │      Clean structured JSON files ready for chunking
│   │
│   └── vectordb/
│          Future vector database storage
│
│
├── src/
│
│   ├── extraction/
│   │
│   │      extract_pdf.py
│   │
│   │      Extracts text from PDF documents
│
│
│   ├── inspector/
│   │
│   │      inspect_pdf.py
│   │
│   │      Used to inspect PDF structure before parsing
│
│
│   ├── parser/
│   │
│   │      base_parser.py
│   │      Main parser logic used by standards
│   │
│   │
│   │      config/
│   │
│   │          parse_iso27001.py
│   │          parse_iso27002.py
│   │          parse_pci.py
│   │
│   │      Contains parsing rules for each standard
│
│
│   ├── experiments/
│   │
│   │      t.py
│   │      t2.py
│   │
│   │      Temporary testing files
│
│
│   ├── export_json.py
│   │
│   │      Runs parser and exports JSON output
│
│
│   └── cleaner.py
│
│          Cleans extracted data and separates sections
│
│
└── requirements.txt
```

---

# Installation

## 1. Clone the repository

Open terminal:

```bash
git clone https://github.com/Boody700/Nlp_cybersecurity_Rag_bot.git
```

Move into the project folder:

```bash
cd Nlp_cybersecurity_Rag_bot
```

---

## 2. Create Virtual Environment

Create:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Requirements

Run:

```bash
pip install -r requirements.txt
```

---

# Running The Project

## Step 1 - Add PDF Documents

Place your cybersecurity documents inside:

```
data/raw/
```

Example:

```
data/raw/

    ISO27002.pdf
    ISO27001.pdf
    PCI-DSS.pdf
```

---

# Step 2 - Parse Documents

Run:

```bash
python src/export_json.py
```

This will generate:

```
data/processed/
```

Example:

```
iso27002_controls.json
```

The file contains extracted controls:

Example:

```json
{
    "standard": "ISO27002",
    "control_id": "5.1",
    "title": "Policies for information security",
    "page": 19,
    "raw_text": "..."
}
```

---

# Step 3 - Clean Data

Run:

```bash
python src/cleaner.py
```

This removes:

* PDF artifacts
* Headers
* Footers
* Page numbers
* Unnecessary formatting

and separates each control into:

```
control

purpose

guidance

other_information
```

Output:

```
data/cleaned/
```

Example:

```json
{
    "control_id": "5.1",

    "control": "...",

    "purpose": "...",

    "guidance": "..."
}
```

---

# File Explanation

## base_parser.py

The main parsing engine.

Responsible for:

* Reading extracted PDF text
* Detecting controls
* Creating control objects
* Saving extracted information

---

## parse_iso27002.py

Contains ISO27002-specific parsing rules.

Example:

```
5.1 Policies for information security
```

is detected as a control.

---

## export_json.py

Runs the parser and exports the extracted controls into JSON format.

---

## cleaner.py

Transforms raw extracted text into structured information.

Example:

Before:

```
Control
Purpose
Guidance
Other information
```

After:

```
{
 control,
 purpose,
 guidance,
 other_information
}
```

---

# Supported Standards

Currently supported:

✅ ISO/IEC 27002:2022

Prepared for future support:

* ISO/IEC 27001
* PCI DSS

---

# Current Progress

Completed:

✅ PDF extraction
✅ PDF structure inspection
✅ Base parser
✅ ISO27002 parser
✅ JSON export
✅ Data cleaning

Next:

⬜ Chunking
⬜ Embeddings
⬜ ChromaDB Vector Database
⬜ Retrieval System
⬜ LLM Integration
⬜ Final Chatbot

---

# Git Workflow (IMPORTANT)

## Do NOT work directly on main

The `main` branch is the stable version.

Do not push changes directly to:

```
main
```

Every team member should create their own branch.

---

# Creating a New Branch

First update your local project:

```bash
git checkout main

git pull
```

Create your branch:

```bash
git checkout -b your-branch-name
```

Example:

```bash
git checkout -b ahmed-chunker
```

or:

```bash
git checkout -b sara-rag
```

---

# Before Working

Check your current branch:

```bash
git branch
```

Example:

```
  main
* ahmed-chunker
```

The `*` shows your current branch.

---

# Saving Your Work

After making changes:

Check files:

```bash
git status
```

Add changes:

```bash
git add .
```

Commit:

```bash
git commit -m "describe your changes"
```

Example:

```bash
git commit -m "Added chunking implementation"
```

Push:

```bash
git push -u origin your-branch-name
```

---

# Current Development Branches

Current stable branch:

```
main
```

Current development work:

```
B1
```

`B1` contains the completed preprocessing pipeline:

* PDF extraction
* Parsing
* JSON export
* Cleaning

Future features should be developed on separate branches.

---

# Generated Files

Do not manually edit generated files:

```
data/processed/

data/cleaned/

data/vectordb/
```

These files are created automatically by the pipeline.

If changes are needed:

1. Modify the source code.
2. Run the pipeline again.
3. Generate new files.

---

# Development Philosophy

Each stage has one responsibility:

```
Parser

↓

Exporter

↓

Cleaner

↓

Chunker

↓

Embedding Model

↓

Vector Database

↓

Retriever

↓

LLM
```

This makes the system easier to debug, improve, and maintain.

---

# Project Goal

Build an AI cybersecurity assistant capable of answering questions using trusted information extracted from official cybersecurity standards.
