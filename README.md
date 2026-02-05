# Text-to-SQL Evaluation System

A comprehensive evaluation framework for Natural Language to SQL translation, focusing on improving query generation accuracy against the Spider dataset benchmark.


## 🎯 Project Overview

This project implements an advanced Text-to-SQL system with:
- Vector database integration with ChromaDB
- Semantic layer for business intelligence queries
- Multiple evaluation strategies and dataset splitting methods
- Production-ready modular architecture

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [Architecture](#-architecture)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Capabilities
- **SQL Generation**: Natural language to SQL translation with high accuracy
- **Spider Benchmark**: Comprehensive evaluation on industry-standard dataset
- **Multiple Strategies**: Support for various prompting and evaluation approaches
- **Execution Validation**: Verify generated SQL runs correctly against databases

### Advanced Features
- **ChromaDB Integration**: Retrieval-augmented generation with vector similarity search
- **Semantic Layer**: Business logic and domain knowledge integration
- **Dataset Splitting**: Random, database-based, difficulty-based, few-shot, and cross-domain splits
- **Preprocessing Pipeline**: Automatic query fixing and formatting compliance
- **Comprehensive Testing**: Unit and integration tests with high coverage

### Enhancement Options
- Multiple prompt templates (basic, few-shot, chain-of-thought, enhanced)
- Debugging mode for SQL correction
- Schema-aware query generation
- Intent recognition and query enhancement
- Automatic formatting compliance for Spider evaluation

## 📁 Project Structure

```
text2sql-evaluation/
├── src/                          # Source code
│   ├── data/                     # Data loading and processing
│   │   ├── spider_loader.py      # Spider dataset loader
│   │   ├── sql_parser.py         # SQL parsing utilities
│   │   └── dataset_splitter.py   # Dataset splitting strategies
│   ├── evaluation/               # Evaluation framework
│   │   ├── base_evaluator.py    # Base evaluation class
│   │   ├── chromadb_evaluator.py # ChromaDB-enhanced evaluator
│   │   ├── exec_evaluator.py    # Execution accuracy evaluation
│   │   └── evaluator.py          # Main evaluation coordinator
│   ├── retrieval/                # Retrieval systems
│   │   ├── chromadb_handler.py  # ChromaDB interface
│   │   └── spider_chromadb_integration.py
│   └── prompts/                  # Prompt templates
│       └── prompt_manager.py     # Prompt template management
├── semantic_layer/               # Semantic understanding
│   ├── core.py                   # Semantic layer core logic
│   ├── evaluator.py              # Semantic-enhanced evaluator
│   ├── config.json               # Business logic configuration
│   └── setup.py                  # Setup and validation
├── utils/                        # Utility functions
│   ├── sql_schema.py             # Schema parsing
│   ├── embedding_utils.py        # Embedding generation
│   ├── logging_utils.py          # Logging configuration
│   └── schema_utils.py           # Schema utilities
├── scripts/                      # Executable scripts
│   ├── evaluate_spider.py        # Main evaluation script
│   ├── build_chromadb.py         # Build vector database
│   ├── analyze_failures.py       # Failure analysis
│   └── test_components.py        # Component testing
├── tests/                        # Test suite
│   ├── test_evaluator.py
│   ├── test_chromadb.py
│   └── test_prompt_manager.py
├── data/                         # Data directory
│   ├── raw/spider/               # Spider dataset
│   ├── embeddings/               # Vector embeddings
│   └── processed/                # Processed data
├── configs/                      # Configuration files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- 2GB+ disk space for dataset and embeddings

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd text2sql-evaluation
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Spider Dataset
```bash
python scripts/download_spider.py --output data/raw/spider
```

### Step 5: Build ChromaDB (Optional)
```bash
python scripts/build_chromadb.py \
  --data-dir data/raw/spider \
  --persist-dir data/embeddings/chroma_db
```

## 🏃 Quick Start

### Basic Evaluation
Run evaluation on a sample of the Spider dataset:

```bash
python scripts/evaluate_spider.py \
  --gold data/raw/spider/dev_gold.sql \
  --pred data/raw/spider/dev_pred.sql \
  --db data/raw/spider/database \
  --table data/raw/spider/tables.json \
  --etype all
```

### Generate SQL with LangChain
```bash
python scripts/evaluate_spider.py \
  --use_langchain \
  --questions data/raw/spider/dev.json \
  --db data/raw/spider/database \
  --table data/raw/spider/tables.json \
  --prompt_type enhanced
```

### With ChromaDB Retrieval
```bash
python scripts/evaluate_spider.py \
  --use_langchain \
  --use_chromadb \
  --questions data/raw/spider/dev.json \
  --db data/raw/spider/database \
  --table data/raw/spider/tables.json \
  --chromadb_n_examples 5
```

### With Semantic Layer
```bash
python scripts/evaluate_spider.py \
  --use_langchain \
  --use_semantic \
  --questions data/raw/spider/dev.json \
  --db data/raw/spider/database \
  --table data/raw/spider/tables.json
```

## 📖 Usage

### Evaluation Types

**Exact Match (`match`)**: Compares SQL query structure
```bash
--etype match
```

**Execution (`exec`)**: Compares query results
```bash
--etype exec
```

**Both (`all`)**: Performs both evaluations
```bash
--etype all
```

### Prompt Templates

- `basic`: Simple template with schema and question
- `few_shot`: Includes example queries
- `chain_of_thought`: Step-by-step reasoning
- `enhanced`: Advanced template with rules and patterns
- `step_by_step`: Structured 5-step approach

```bash
--prompt_type enhanced
```

### Dataset Splitting

Create custom dataset splits for evaluation:

```bash
# Random split
python scripts/split_dataset.py --strategy random --train-size 0.8

# Database-based (cross-database generalization)
python scripts/split_dataset.py --strategy database

# Difficulty-based
python scripts/split_dataset.py --strategy difficulty

# Few-shot learning
python scripts/split_dataset.py --strategy few-shot --num-examples 5

# Cross-domain
python scripts/split_dataset.py --strategy cross-domain
```

### Advanced Configuration

**ChromaDB Parameters**:
```bash
--chromadb_n_examples 5           # Number of similar examples to retrieve
--chromadb_min_similarity 0.3     # Minimum similarity threshold
--chromadb_data_dir ./data/raw/spider
--chromadb_persist_dir ./data/embeddings/chroma_db
```

**Debug Mode**:
```bash
--enable_debugging    # Enable SQL debugging and correction
--verbose            # Detailed logging
```

## 📊 Evaluation Metrics

### Component-Level Metrics

- SELECT clause accuracy
- WHERE clause accuracy
- GROUP BY accuracy
- ORDER BY accuracy
- Keywords accuracy
- Overall F1 score

### Difficulty Breakdown

Results are reported across four difficulty levels:
- Easy
- Medium
- Hard
- Extra Hard

### Example Output

```
============================================================
                    FINAL RESULTS
============================================================

Exact Match Accuracy: 67.3%
Execution Accuracy: 96.8%

Component Scores:
  SELECT:    F1=0.89  Acc=0.91  Rec=0.87
  WHERE:     F1=0.84  Acc=0.86  Rec=0.82
  GROUP BY:  F1=0.76  Acc=0.78  Rec=0.74
  ORDER BY:  F1=0.71  Acc=0.73  Rec=0.69
  KEYWORDS:  F1=0.88  Acc=0.90  Rec=0.86

By Difficulty:
  Easy:       EM=82.4%  Exec=98.1%
  Medium:     EM=68.3%  Exec=96.9%
  Hard:       EM=51.2%  Exec=94.7%
  Extra Hard: EM=34.6%  Exec=91.3%
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  User Query                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Semantic Layer (Optional)               │
│  - Intent Recognition                                │
│  - Business Logic Enhancement                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         ChromaDB Retrieval (Optional)                │
│  - Similar Example Retrieval                         │
│  - Schema Context Enhancement                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              SQL Generation                          │
│  - Schema Parsing                                    │
│  - Prompt Construction                               │
│  - LLM Query Generation                              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            Preprocessing & Fixing                    │
│  - Backtick Removal                                  │
│  - Format Compliance                                 │
│  - Spider-specific Corrections                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 Evaluation                           │
│  - Exact Match (Structure)                           │
│  - Execution Accuracy (Results)                      │
│  - Component-level Analysis                          │
└─────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Composition over Inheritance**: Semantic layer and ChromaDB integration use composition
2. **Modular Architecture**: Clear separation of concerns across modules
3. **Strategy Pattern**: Multiple evaluation and splitting strategies
4. **Factory Pattern**: Evaluator creation based on configuration
5. **Template Method**: Base evaluator with extensible evaluation pipeline

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_evaluator.py -v

# Run specific test
python -m pytest tests/test_evaluator.py::test_sql_generation -v
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/my-feature`
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

### Project Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Maintain test coverage > 80%
- Update README for new features

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**ChromaDB Connection Issues**
```bash
# Reset ChromaDB
python scripts/reset_chromadb.py --confirm

# Rebuild
python scripts/build_chromadb.py --data-dir data/raw/spider
```

**Low Exact Match with High Execution**
This indicates formatting issues. Apply preprocessing:
```bash
python scripts/fix_query_formatting.py \
  --input results/evaluation.json \
  --apply-spider-compliance
```


**Missing Dependencies**
```bash
pip install -r requirements.txt --upgrade
```

## 🙏 Acknowledgments

- [Spider Dataset](https://yale-lily.github.io/spider) - Yale Semantic Parsing and Text-to-SQL Challenge
- [ChromaDB](https://www.trychroma.com/) - Vector database for embeddings
- [LangChain](https://www.langchain.com/) - LLM application framework
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
