# NLP-Deep-Learning-project-For-Security-Regualtions

---
#Try the app
https://nlp-deep-learning-project-for-security-regulation.streamlit.app/
# Project Overview

This project aims to build a **Cybersecurity Retrieval-Augmented Generation (RAG) Assistant**.

The goal is to allow users to ask questions about cybersecurity standards and receive answers based on official documents such as:

* ISO/IEC standards
* PCI DS
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
git clone https://github.com/Gehan3/NLP-Deep-Learning-project-For-Security-Regulations
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
## standard Chunking strategy
* 1)Fixed-size chunking
    A[Artificial intelligence is transforming technology and shaping the future.] --> B[Overlap]
    B --> C["Chunk 1: Artificial intelligence is transforming technology"]
    B --> D["Chunk 2: transforming technology and shaping the future."]
* 2) Semantic chunking
    A[Document] --> B[Segment document<br>sentences or paragraphs]
    B --> C[Initial first chunk]
    C --> D[Keep adding new segments until cosine similarity drops drastically]
    D --> E[Final first chunk]
    D --> F[Initial second chunk]
    E --> G[Final chunks]
    F --> G
* 3) Recursive chunking
    A[Document] --> B[Segment document<br>paragraphs or thematic sections]
    B --> C[Select a segment]
    C --> D{Size > chunk-size limit?}
    D -- NO --> B
    D -- YES --> E[Split further recursively]
    E --> D
* 4) Document structure-based chunking
    A["Document:<br>- Title<br>- Introduction<br>- Section #1<br>- Section #2<br>- Conclusion"] --> B[Chunk using the inherent structure]
    B --> C[Chunk 1: Title]
    B --> D[Chunk 2: Introduction]
    B --> E[Chunk 3: Section #1]
    B --> F[Chunk 4: Section #2]
    B --> G[Chunk 5: Conclusion]
    G --> H["*Merge with recursive chunking if needed"]
* 5) LLM-based chunking
    A[Document] --> B[Input to LLM]
    B --> C[LLM]
    C --> D[LLM generates chunks]
    D --> E[Final chunks]
## ✂️ Chunking Strategies (ISO 27002 RAG)
Since ISO 27002 has a highly structured layout (**Control ➡️ Purpose ➡️ Guidance ➡️ Other Info**), standard character-based chunking will cause a loss of context. We propose three structural chunking strategies:

### 1. Metadata Enrichment (Semantic Tagging)
* **Concept:** Structural splitting with context injection.
* **Mechanism:** Splits each control into its 4 raw sections and prepends the Control ID/Title as a prefix to every chunk.
  * *Example:* `[Control 8.32] [Section: Purpose] <text...>`
* **Pros:** Highly focused chunks, optimal for embedding models, preserves exact control mapping.
* **Cons:** Medium implementation overhead.

### 2. Parent-Child (Hierarchical Chunking)
* **Concept:** Decoupling retrieval (search) from generation (LLM context).
* **Mechanism:** 
  * **Child Chunks:** Small, granular sub-sections used *only* for semantic search.
  * **Parent Chunks:** The complete, undivided control document retrieved and sent to the LLM once a child is matched.
* **Pros:** Maximum retrieval accuracy + full context retention for generation.
* **Cons:** Requires a more complex vector DB architecture.

### 3. Sliding Window over Sections
* **Concept:** Regex-based structural splits with traditional linguistic overlap.
* **Mechanism:** Splits the document strictly by section headers, carrying over the last 1–2 sentences of each section into the next chunk.
* **Pros:** Simple to implement, maintains logical transitions between adjacent ideas.
* **Cons:** May lead to redundant context or larger chunk sizes.

### Main way used in parser Document structure-based chunking 
### I prefer Parent child (Recursive Chunking)
⚙️ Phase 1: Pipeline Initialization & Configuration

chunker.py
==========
Parent-Child (Parent Document Retrieval) chunking pipeline for the
local ISO 27002 RAG bot.

Pipeline position:
    base_parser.py -> cleaner.py -> exporter.py -> [chunker.py] -> (embedder/indexer next)

Reads:
    data/cleaned/iso27002_cleaned.json

Writes:
    data/chunks/iso27002_parents.json
    data/chunks/iso27002_children.json

Install:
    pip install tiktoken
    (script degrades gracefully to a word-based token estimator if
    tiktoken is not installed, so it never hard-crashes)
"""
chunker.py
The parent-child splitting engine.

Responsible for:

Reading the standardized JSON and retaining the full text sections as parent chunks

Tokenizing and slicing text into high-retrieval child chunks (target size: 60 tokens)

Managing sliding overlap (12 tokens) and absorbing small text crumbs (under 8 tokens)

Outputting structured JSON arrays for both parent and child layers

### 2. Smart Tokenizer Verification (Graceful Fallback)
`tiktoken` 
word-counting library

* **If Available:**
measure text sizes with absolute token-level accuracy with dictionary related to our model as GPT-4 (cl100K_base)
* **If Missing:
Fallback Heuristic
### 3. The Locked Blueprint (Safe Configuration)
*File Paths:
 Direct locations for the raw source data and the final storage directories for the processed parent and child pieces.
*Chunking Metrics:
The target token size for child fragments (60 units), the sliding overlap window to preserve context between adjacent pieces (12 units), and the minimum size limit to prevent tiny, meaningless text fragments (8 units).
*Target Sections:
Specific target keys from the standard (such as control statements, purposes, guidelines, and extra notes) slated for analysis and slicing.

### 4. Real-Time Activity Logging
Every single operation is printed with a precise timestamp (hour, minute, and second) along with its severity level.

### 5.TokenCounter with safe fallback
graph TD
    A[Start: Initialize TokenCounter] --> B{Is tiktoken installed?}
    
    %% Path A: Tiktoken is available
    B -- Yes --> C[Load 'cl100k_base' Encoder]
    C --> D[Log: Tokenizer loaded successfully]
    D --> E[Methods: Exact Token counting & splitting via regex/IDs]
    
    %% Path B: Fallback
    B -- No --> F[Trigger Exception / Catch ImportError]
    F --> G[Log Warning: Using fallback counter]
    G --> H[Methods: Approximate word-based counting & splitting]
    
    %% Splitting Logic
    E --> I[count(): len(encoder.encode)]
    H --> J[count(): round(words * 1.3)]
    
    I --> K[split_by_tokens(): Precise sub-arrays using token IDs]
    J --> L[split_by_tokens(): Sub-arrays using space-split words]
### Splitter 
1. **Mask Decimals & Abbreviations** 
   Swaps dots in control IDs (e.g., `6.1.2`) and abbr (`e.g.`) for protection  with a temporary placeholder to prevent early splitting.

2. **Regex Boundary Splitting** 
   Splits the protected text into raw sentences only when true punctuation is followed by a capital letter, quote, or list number.

3. **Restore Dots & Clean Text** 
   Replaces placeholders back with original periods and trims empty spaces to produce clean, isolated sentences.

4. **Map to Parent-Child Structures** 
   Groups the final sentences into structural objects (`ParentChunk` for full prompt context and `ChildChunk` for vector embeddings).

Parent-child Chunk
      ↓
metadata context
      ↓
Packing sentences into token-bounded, overlapping chunks
      ↓
per-section child generation
      ↓

### **change and try to see the results
check the code wz key word "change and try to see the results" and try in these parameters to find the best solution
============================================================
CHUNKING STATISTICS
============================================================
Total parents:            X   <- Verification of parsed blocks
Total children:           Y   <- Total embeddings generated
Avg tokens / child:       Z   <- Optimization tracking (Ideal: 100-150)
Min / Max tokens / child: A/B <- Validation against target context windows
Empty sections skipped:   N   <- Measures text extraction coverage
Oversized sentences:      M   <- Hard-split alerts (Keep as close to 0 as possible)
Children per section:
  - control               Count
  - guidance              Count
============================================================


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
