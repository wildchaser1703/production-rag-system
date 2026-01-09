# Production-Grade RAG Pipeline 🚀

A state-of-the-art Retrieval-Augmented Generation (RAG) pipeline built with Python 3.10+, LangChain, and ChromaDB. Designed for scalability, observability, and evaluation.

## ✨ Key Features

- **Modern Python Core**: Leverages Python 3.10/3.11 type hinting and Pydantic V2 for robust validation.
- **Advanced Retrieval**: ChromaDB-backed vector storage with configurable chunking strategies.
- **Production Observability**: Structured logging with `loguru` and environment-based configuration.
- **Automated Evaluation**: Integrated with `RAGAS` to measure faithfulness, relevance, and precision.
- **CI/CD Ready**: Pre-configured GitHub Actions for linting (Ruff), type-checking (MyPy), and testing (Pytest).
- **Quality Ingestion**: Built-in support for web-based data ingestion with recursive chunking.

## 🛠️ Architecture

1. **Ingestion**: `DocumentProcessor` fetches and chunks content from web sources.
2. **Indexing**: `VectorManager` creates embeddings using OpenAI's `text-embedding-3-small` and stores them in ChromaDB.
3. **Retrieval**: Semantic search with configurable top-k retrieval.
4. **Generation**: `RAGGenerator` uses LCEL (LangChain Expression Language) for clean, composable RAG chains.
5. **Evaluation**: `RAGEvaluator` provides quantitative metrics on pipeline performance.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- OpenAI API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/production-rag-system.git
cd production-rag-system

# Install dependencies
pip install -e .
```

### Configuration
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
# Edit .env with your OpenAI API Key
```

### Running the Pipeline
```bash
python -m src.rag_pipeline.main
```

## 🧪 Testing and Quality Control

```bash
# Run tests
pytest

# Linting
ruff check .

# Type Checking
mypy src
```

## 📊 Evaluation Metrics
The pipeline uses the following RAGAS metrics:
- **Faithfulness**: Is the answer derived solely from the context?
- **Answer Relevance**: How well does the answer address the question?
- **Context Precision**: Is the retrieved context relevant to the question?
- **Context Recall**: Does the retrieved context contain all the necessary information?

## 📄 License
MIT