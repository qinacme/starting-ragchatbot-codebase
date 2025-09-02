#!/bin/bash
# Format code with Black and isort

set -e

echo "🔧 Formatting code with Black..."
uv run black backend/ main.py

echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✨ Code formatting complete!"