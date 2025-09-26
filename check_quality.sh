#!/bin/bash

# Development quality check script for RAG Chatbot
# This script runs all code quality tools

set -e  # Exit on error

echo "🔍 Running Code Quality Checks..."
echo "================================="

# Check if development dependencies are installed
echo "📦 Ensuring development dependencies are installed..."
uv sync --extra dev

echo ""
echo "1️⃣ Running Black formatter check..."
echo "---------------------------------"
uv run black --check backend/ || {
    echo "❌ Black formatting issues found. Run './format_code.sh' to fix."
    exit 1
}
echo "✅ Black check passed"

echo ""
echo "2️⃣ Running isort import sorting check..."
echo "-------------------------------------"
uv run isort --check-only backend/ || {
    echo "❌ Import sorting issues found. Run './format_code.sh' to fix."
    exit 1
}
echo "✅ isort check passed"

echo ""
echo "3️⃣ Running Ruff linter..."
echo "----------------------"
uv run ruff check backend/
echo "✅ Ruff check passed"

echo ""
echo "4️⃣ Running MyPy type checker..."
echo "----------------------------"
uv run mypy backend/ || {
    echo "⚠️  MyPy type checking completed with warnings"
}

echo ""
echo "5️⃣ Running tests..."
echo "----------------"
if [ -f backend/test_rag_system.py ]; then
    uv run pytest backend/test_rag_system.py -v
    echo "✅ Tests passed"
else
    echo "⚠️  No tests found in backend/test_rag_system.py"
fi

echo ""
echo "================================="
echo "✨ All quality checks completed!"
echo ""
echo "Tips:"
echo "  • Run './format_code.sh' to automatically fix formatting issues"
echo "  • Run individual tools with: uv run <tool> <args>"
echo "  • Configure your IDE to use these tools for automatic formatting"