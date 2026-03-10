# ============================================================================
# Makefile — Legal Judgment Prediction & Explanation
# ============================================================================

.PHONY: help install setup train-encoder generate-embeddings train-classifier \
        train-all evaluate build-vectordb app test clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────

install: ## Install all dependencies
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab', quiet=True)" || true

setup: install ## Full setup (install + nltk data)
	@echo "✅ Setup complete. Next: copy model files and run 'make app'"

# ── Training (requires GPU) ─────────────────────────────────────────────────

train-encoder: ## Fine-tune encoder (usage: make train-encoder MODEL=xlnet)
	python scripts/train_encoder.py --config configs/models/$(MODEL)_bigru.yaml

generate-embeddings: ## Generate embeddings (usage: make generate-embeddings MODEL=xlnet)
	python scripts/generate_embeddings.py --config configs/models/$(MODEL)_bigru.yaml

train-classifier: ## Train HAN classifier (usage: make train-classifier MODEL=xlnet)
	python scripts/train_classifier.py --config configs/models/$(MODEL)_bigru.yaml

train-pipeline: ## Run full pipeline for one model (usage: make train-pipeline MODEL=xlnet)
	@echo "══════ Step 1/3: Fine-tuning $(MODEL) encoder ══════"
	python scripts/train_encoder.py --config configs/models/$(MODEL)_bigru.yaml
	@echo ""
	@echo "══════ Step 2/3: Generating $(MODEL) embeddings ══════"
	python scripts/generate_embeddings.py --config configs/models/$(MODEL)_bigru.yaml
	@echo ""
	@echo "══════ Step 3/3: Training HAN classifier for $(MODEL) ══════"
	python scripts/train_classifier.py --config configs/models/$(MODEL)_bigru.yaml
	@echo ""
	@echo "✅ Full pipeline complete for $(MODEL)"

train-all: ## Train all model variants end-to-end
	@for model in xlnet roberta bert distilbert; do \
		echo ""; \
		echo "╔══════════════════════════════════════════╗"; \
		echo "║  Training: $$model                        ║"; \
		echo "╚══════════════════════════════════════════╝"; \
		echo ""; \
		$(MAKE) train-pipeline MODEL=$$model; \
	done
	@echo ""
	@echo "✅ All models trained. Run 'make evaluate' to compare results."

# ── Evaluation ──────────────────────────────────────────────────────────────

evaluate: ## Evaluate all trained models and generate comparison
	python scripts/evaluate.py

evaluate-model: ## Evaluate a single model (usage: make evaluate-model MODEL=xlnet)
	python scripts/evaluate.py --models $(MODEL)

# ── Chatbot ─────────────────────────────────────────────────────────────────

build-vectordb: ## Build FAISS vector database for chatbot
	python scripts/build_vector_db.py

# ── Application ─────────────────────────────────────────────────────────────

app: ## Launch Streamlit application
	streamlit run app/streamlit_app.py

# ── Testing ─────────────────────────────────────────────────────────────────

test: ## Run all unit tests
	python -m pytest tests/ -v

test-fast: ## Run tests excluding slow ones
	python -m pytest tests/ -v -m "not slow"

# ── Utilities ───────────────────────────────────────────────────────────────

clean: ## Remove cached files and logs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f logs/*.log
	@echo "✅ Cleaned"

clean-models: ## Remove all trained models (⚠️  destructive)
	@echo "⚠️  This will delete all trained models. Press Ctrl+C to cancel."
	@sleep 3
	rm -rf trained_models/*/
	rm -f trained_models/*.h5
	@echo "✅ Models removed"