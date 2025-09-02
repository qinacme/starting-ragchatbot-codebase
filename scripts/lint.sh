#!/bin/bash
# Run all linting and quality checks

set -e

echo "🔍 Running flake8..."
uv run flake8 backend/ main.py

echo "📝 Checking formatting with Black..."
uv run black --check backend/ main.py

echo "📦 Checking import sorting with isort..."
uv run isort --check-only backend/ main.py

echo "✅ All quality checks passed!"

echo "ℹ️  Note: Run 'uv run mypy backend/ main.py' separately for optional type checking"