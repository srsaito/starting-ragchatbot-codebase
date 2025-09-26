#!/bin/bash

# Code formatting script for RAG Chatbot
# Automatically formats Python code using black and isort

set -e  # Exit on error

echo "🎨 Formatting Code..."
echo "===================="

# Check if development dependencies are installed
echo "📦 Ensuring development dependencies are installed..."
uv sync --extra dev

echo ""
echo "1️⃣ Running isort to sort imports..."
echo "--------------------------------"
uv run isort backend/
echo "✅ Imports sorted"

echo ""
echo "2️⃣ Running Black formatter..."
echo "--------------------------"
uv run black backend/
echo "✅ Code formatted"

echo ""
echo "3️⃣ Running Ruff auto-fix for linting issues..."
echo "-------------------------------------------"
uv run ruff check --fix backend/ || true
echo "✅ Ruff fixes applied"

echo ""
echo "===================="
echo "✨ Code formatting complete!"
echo ""
echo "Next steps:"
echo "  • Review the changes with: git diff"
echo "  • Run quality checks with: ./check_quality.sh"
echo "  • Commit your formatted code"