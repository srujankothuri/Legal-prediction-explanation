#!/bin/bash
# ============================================================================
# Commit #1: Initialize project structure
# Run this from inside your existing Legal-prediction-explanation repo
# ============================================================================

# Step 1: Clean the existing repo (backup first!)
# Your existing repo at: https://github.com/srujankothuri/Legal-prediction-explanation
# We'll restructure it completely

echo "Creating project structure..."

# ── Directory Structure ─────────────────────────────────────────────────────
mkdir -p configs/models
mkdir -p src/data
mkdir -p src/models
mkdir -p src/training
mkdir -p src/inference
mkdir -p src/evaluation
mkdir -p src/chatbot
mkdir -p src/utils
mkdir -p scripts
mkdir -p app/pages
mkdir -p app/components
mkdir -p notebooks
mkdir -p tests
mkdir -p trained_models
mkdir -p data/raw
mkdir -p data/embeddings/xlnet
mkdir -p data/embeddings/roberta
mkdir -p data/embeddings/bert
mkdir -p data/embeddings/distilbert
mkdir -p vector_db
mkdir -p assets
mkdir -p logs

# ── Create all __init__.py files ────────────────────────────────────────────
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/evaluation/__init__.py
touch src/chatbot/__init__.py
touch src/utils/__init__.py
touch app/__init__.py
touch app/pages/__init__.py
touch app/components/__init__.py
touch tests/__init__.py

# ── Create placeholder files ────────────────────────────────────────────────
touch logs/.gitkeep
touch vector_db/.gitkeep
touch trained_models/.gitkeep
touch data/embeddings/xlnet/.gitkeep
touch data/embeddings/roberta/.gitkeep
touch data/embeddings/bert/.gitkeep
touch data/embeddings/distilbert/.gitkeep
touch notebooks/.gitkeep

echo "✅ Project structure created!"
echo ""
echo "Next: Create the config/code files listed below, then run:"
echo "  git add -A"
echo "  git commit -m 'feat: initialize project structure with modular architecture'"
echo "  git push origin main"
