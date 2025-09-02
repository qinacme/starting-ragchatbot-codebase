#!/bin/bash
# Run all quality checks and formatting

set -e

echo "🚀 Running complete code quality pipeline..."
echo

# Format code first
echo "1. Formatting code..."
./scripts/format.sh
echo

# Then run all checks
echo "2. Running quality checks..."
./scripts/lint.sh
echo

echo "🎉 Code quality pipeline complete!"