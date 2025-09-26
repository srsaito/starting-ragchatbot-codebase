# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system that enables users to query course materials and receive AI-powered responses. It combines ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Development Commands

### Environment Setup
- **Initial setup**: `uv sync` (installs dependencies via uv package manager)
- **Create .env**: Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`

### Running the Application
- **Start server**: `./run.sh` (recommended) or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Access**: Web interface at http://localhost:8000, API docs at http://localhost:8000/docs

### Code Quality Commands
- **Install dev dependencies**: `uv sync --extra dev`
- **Format code**: `./format_code.sh` - Automatically formats all Python code with black and isort
- **Check quality**: `./check_quality.sh` - Runs all quality checks (black, isort, ruff, mypy)
- **Run specific tools**:
  - Format: `uv run black backend/` and `uv run isort backend/`
  - Lint: `uv run ruff check backend/`
  - Type check: `uv run mypy backend/`
- **Pre-commit hooks**: `pre-commit install` (optional, for automatic checks before commits)

### Project Dependencies
- Python 3.13+ with uv package manager
- **IMPORTANT**: Always use `uv` for package management - never use pip directly
- Key packages: chromadb, anthropic, sentence-transformers, fastapi, uvicorn
- Dependencies managed in `pyproject.toml` and `uv.lock`

## Architecture Overview

### Core Components Flow
1. **Frontend** (`frontend/`): Vanilla JavaScript SPA with direct API calls
2. **API Gateway** (`backend/app.py`): FastAPI server handling HTTP requests  
3. **RAG System** (`backend/rag_system.py`): Main orchestrator coordinating all components
4. **AI Generator** (`backend/ai_generator.py`): Claude API integration with tool calling
5. **Search Tools** (`backend/search_tools.py`): Tool-based semantic search system
6. **Vector Store** (`backend/vector_store.py`): ChromaDB interface for embeddings
7. **Document Processor** (`backend/document_processor.py`): Parses course documents into chunks

### Key Architectural Patterns

**Tool-Based AI Architecture**: The system uses Anthropic's tool calling where Claude decides whether to search the knowledge base or answer from general knowledge. This is implemented via the `CourseSearchTool` class which provides semantic search capabilities to the AI.

**Session-Based Conversations**: Each user interaction maintains conversation history via `SessionManager` for context-aware responses across multiple queries.

**Modular RAG Pipeline**: 
- Documents → chunks via `DocumentProcessor`
- Chunks → embeddings via `VectorStore` 
- Query → semantic search via `SearchTools`
- Context + Query → AI response via `AIGenerator`

**Dual Storage Pattern**: ChromaDB stores both course metadata (for browsing) and content chunks (for semantic search) in separate collections, enabling both course discovery and content retrieval.

### Data Models

**Core Models** (defined in `backend/models.py`):
- `Course`: Contains title, lessons, instructor metadata
- `Lesson`: Individual lesson with number, title, optional link
- `CourseChunk`: Text chunk with course/lesson attribution for vector search

**Configuration**: Centralized config in `backend/config.py` with environment variable loading for API keys, chunk sizes, embedding models, etc.

## File Structure Significance

- `frontend/script.js:45`: `sendMessage()` function - entry point for all user queries
- `backend/app.py:56`: `/api/query` endpoint - main API entry point
- `backend/rag_system.py:102`: `query()` method - orchestrates the RAG pipeline  
- `backend/ai_generator.py:43`: `generate_response()` - handles Claude API with tools
- `backend/search_tools.py:52`: `execute()` - performs semantic search when AI decides to search
- `backend/vector_store.py`: Manages ChromaDB operations and embedding functions

## Document Loading

Documents are automatically loaded from `docs/` folder on server startup. The system supports PDF, DOCX, and TXT files. Course content is parsed, chunked, and embedded for semantic search. Existing courses are not reprocessed to avoid duplicates.

## Development Notes

- No traditional test suite exists - testing is primarily through the web interface
- The system uses sentence-transformers model "all-MiniLM-L6-v2" for embeddings
- Claude model configured as "claude-sonnet-4-20250514" in config
- ChromaDB persistence enabled with local storage in `backend/chroma_db/`
- CORS enabled for development with wildcard origins
- Static files served directly from FastAPI for simplicity
- make sure to use uv to manage all dependencies

## Code Quality Standards

### Formatting and Style
- **Black**: Line length 88, target Python 3.13+
- **isort**: Uses black-compatible profile
- **Ruff**: Fast linting with auto-fix capabilities
- **MyPy**: Type checking with lenient settings for gradual typing

### Quality Workflow
1. Before committing: Run `./format_code.sh` to auto-format code
2. Verify quality: Run `./check_quality.sh` to ensure all checks pass
3. Optional: Install pre-commit hooks with `pre-commit install` for automatic checks

### Tool Configuration
All tool settings are centralized in `pyproject.toml` for consistency:
- Black, Ruff, isort, and MyPy configurations
- Excludes `chroma_db/` and `docs/` from quality checks
- Development dependencies in `[project.optional-dependencies]` section