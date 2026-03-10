# ============================================================================
# Makefile — Legal Judgment Prediction & Explanation
# ============================================================================

.PHONY: help install setup train-encoder train-classifier evaluate app clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────

install: ## Install all dependencies
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

setup: install ## Full setup (install + create dirs + download nltk data)
	@echo "✅ Setup complete"

# ── Training ────────────────────────────────────────────────────────────────

train-encoder: ## Fine-tune an encoder (usage: make train-encoder MODEL=xlnet)
	python scripts/train_encoder.py --config configs/models/$(MODEL)_bigru.yaml

train-classifier: ## Train HAN classifier (usage: make train-classifier MODEL=xlnet)
	python scripts/train_classifier.py --config configs/models/$(MODEL)_bigru.yaml

generate-embeddings: ## Generate embeddings (usage: make generate-embeddings MODEL=xlnet)
	python scripts/generate_embeddings.py --config configs/models/$(MODEL)_bigru.yaml

train-all: ## Train all model variants end-to-end
	@for model in xlnet roberta bert distilbert; do \
		echo "══════ Training $$model ══════"; \
		python scripts/train_encoder.py --config configs/models/$${model}_bigru.yaml; \
		python scripts/generate_embeddings.py --config configs/models/$${model}_bigru.yaml; \
		python scripts/train_classifier.py --config configs/models/$${model}_bigru.yaml; \
	done

# ── Evaluation ──────────────────────────────────────────────────────────────

evaluate: ## Evaluate all trained models and generate comparison table
	python scripts/evaluate.py

# ── Application ─────────────────────────────────────────────────────────────

build-vectordb: ## Build FAISS vector database for chatbot
	python scripts/build_vector_db.py

app: ## Launch Streamlit application
	streamlit run app/streamlit_app.py

# ── Utilities ───────────────────────────────────────────────────────────────

test: ## Run unit tests
	python -m pytest tests/ -v

clean: ## Remove cached files and logs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/*.log
	@echo "✅ Cleaned"