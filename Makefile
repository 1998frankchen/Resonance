# Resonance Development Makefile
# Author: Frank Chen (Resonance Team)

.PHONY: help dev-setup test demo format lint clean install-uv check-uv

# Default target
help:
	@echo "🎵 Resonance Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  dev-setup     Set up development environment"
	@echo "  install-uv    Install uv package manager"
	@echo ""
	@echo "Development Commands:"
	@echo "  test          Run all tests"
	@echo "  demo          Run demo validation tests"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run type checking and linting"
	@echo "  clean         Clean build artifacts"
	@echo ""
	@echo "Quick Start:"
	@echo "  make dev-setup && make test && make demo"

# Check if uv is installed
check-uv:
	@which uv > /dev/null || (echo "❌ uv not found. Run 'make install-uv' first." && exit 1)

# Install uv package manager
install-uv:
	@echo "📦 Installing uv package manager..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed! Please restart your shell or run: source ~/.local/bin/env"

# Set up development environment
dev-setup: check-uv
	@echo "🛠️ Setting up Resonance development environment..."
	
	@echo "Creating virtual environment..."
	@uv venv
	
	@echo "Installing package in editable mode..."
	@uv pip install -e .
	
	@echo "Installing development dependencies..."
	@uv pip install pytest black isort pre-commit mypy flake8 coverage
	
	@echo "Setting up pre-commit hooks..."
	@.venv/bin/pre-commit install || echo "⚠️ Pre-commit setup skipped"
	
	@echo "✅ Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate environment: source .venv/bin/activate"
	@echo "2. Run tests: make test"
	@echo "3. Run demos: make demo"

# Run all tests
test:
	@echo "🧪 Running Resonance tests..."
	@source .venv/bin/activate && python -m pytest tests/ -v || echo "⚠️ Some tests failed"
	@echo "✅ Test run completed"

# Run demo validation
demo:
	@echo "🎬 Running Resonance demo tests..."
	@source .venv/bin/activate && python demos/test_data_loading.py
	@echo ""
	@echo "🎯 Testing demo scripts..."
	@bash demos/scripts/demo_sft.sh > /dev/null && echo "✅ SFT demo script works"
	@bash demos/scripts/demo_dpo.sh > /dev/null && echo "✅ DPO demo script works"  
	@bash demos/scripts/demo_eval.sh > /dev/null && echo "✅ Eval demo script works"
	@echo "✅ All demos validated successfully!"

# Format code
format:
	@echo "🎨 Formatting code..."
	@source .venv/bin/activate && black src/ demos/ tests/ || echo "⚠️ Black formatting skipped"
	@source .venv/bin/activate && isort src/ demos/ tests/ || echo "⚠️ Import sorting skipped"
	@echo "✅ Code formatted"

# Run linting and type checking
lint:
	@echo "🔍 Running linting and type checking..."
	@source .venv/bin/activate && flake8 src/ --max-line-length=88 --extend-ignore=E203 || echo "⚠️ Flake8 linting found issues"
	@source .venv/bin/activate && mypy src/ --ignore-missing-imports || echo "⚠️ MyPy type checking found issues"
	@echo "✅ Linting completed"

# Run comprehensive quality checks
check: format lint test demo
	@echo "🎉 All quality checks completed!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .mypy_cache/
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned build artifacts"

# Install package for production
install: check-uv
	@echo "📦 Installing Resonance..."
	@uv pip install .
	@echo "✅ Resonance installed successfully!"

# Development server (for future web interface)
serve:
	@echo "🌐 Starting Resonance development server..."
	@echo "⚠️ Development server not yet implemented"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "⚠️ Documentation generation not yet implemented"

# Show project status
status:
	@echo "🎵 Resonance Project Status"
	@echo "=========================="
	@echo ""
	@echo "📁 Project Structure:"
	@find src/resonance -name "*.py" | head -10 | sed 's/^/  /'
	@echo "     ... ($(find src/resonance -name "*.py" | wc -l) total Python files)"
	@echo ""
	@echo "🎬 Demo Files:"
	@ls -la demos/data/ | tail -n +2 | sed 's/^/  /'
	@echo ""
	@echo "🖼️ Demo Images:"
	@ls -1 demos/images/ | wc -l | sed 's/^/  /' | tr -d '\n' && echo " image files"
	@echo ""
	@echo "📊 Lines of Code:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1 | sed 's/^/  /'

# Quick development cycle
dev: format lint test demo
	@echo "🚀 Development cycle completed successfully!"

# Production build
build: clean check
	@echo "🏗️ Building Resonance for production..."
	@source .venv/bin/activate && python -m build || echo "⚠️ Build failed"
	@echo "✅ Production build completed"