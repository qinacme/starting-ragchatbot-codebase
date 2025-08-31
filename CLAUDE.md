# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Copy environment template and add your Anthropic API key
cp .env.example .env

# Install dependencies using uv
uv sync
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials. The system uses a tool-based approach where Claude intelligently decides when to search versus answering directly.

### Core Components Flow

1. **Frontend (Vanilla JS)** → User interface with real-time chat
2. **FastAPI Backend** → API endpoints and request orchestration  
3. **RAG System** → Central orchestrator managing all components
4. **AI Generator** → Claude API integration with tool-calling capability
5. **Search Tool** → Semantic search orchestration
6. **Vector Store** → ChromaDB for embeddings and retrieval
7. **Document Processor** → Structured course content parsing

### Key Architectural Patterns

**Tool-Based RAG**: Claude autonomously decides whether to search based on query type. Uses Anthropic's tool-calling API with a single `search_course_content` tool.

**Structured Document Processing**: Course files follow a specific format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional url]
[content]
```

**Context-Enhanced Chunking**: Text chunks include course and lesson context:
- `"Course {title} Lesson {number} content: {chunk}"`
- Intelligent sentence-based chunking with configurable overlap

**Session Management**: Conversation history maintained with configurable limits (default: 2 messages).

## Configuration

### Key Settings (`backend/config.py`)
- `CHUNK_SIZE`: 800 characters (optimal for semantic search)
- `CHUNK_OVERLAP`: 100 characters (context preservation)
- `MAX_RESULTS`: 5 search results returned
- `MAX_HISTORY`: 2 conversation messages remembered
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (SentenceTransformer)
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"

### Data Models (`backend/models.py`)
- `Course`: Metadata with lessons list
- `Lesson`: Individual lesson with optional links
- `CourseChunk`: Vector storage unit with course/lesson context

## Search System

**Two-Stage Search**:
1. **Course Name Matching**: Fuzzy matching on course titles
2. **Content Search**: Semantic similarity on chunk embeddings

**Search Tool Parameters**:
- `query` (required): What to search for
- `course_name` (optional): Filter by course title
- `lesson_number` (optional): Filter by specific lesson

## Document Loading

Documents auto-load from `docs/` folder on startup. Supported formats: `.txt`, `.pdf`, `.docx`

**Deduplication**: Existing courses (by title) are skipped to prevent reprocessing.

**Storage Location**: ChromaDB persisted to `./chroma_db`

## Development Notes

- **No Test Suite**: This codebase currently has no automated tests
- **Dependency Management**: Uses `uv` for Python package management
- **Environment**: Requires Python 3.13+
- **API Key**: Anthropic API key required in `.env` file
- **Port**: Application runs on port 8000 by default

## Tool Integration

The search tool integrates with Claude's function calling via:
- Tool definitions registered with `ToolManager`
- Sources tracked and returned to frontend for attribution
- Results formatted with course/lesson context headers
- always use uv to run the server do not use pip directly
- use uv to run Python files