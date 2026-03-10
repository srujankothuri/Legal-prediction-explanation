# ⚖️ Legal Judgment Prediction & Explanation

AI-powered Indian Supreme Court judgment prediction with sentence-level explainability and a RAG-based legal chatbot.

> **Published**: Springer ICDSA 2025 — *Court Judgment Prediction using Hierarchical Attention Networks*

## 🚧 Status: Under Active Development

This repository is being rebuilt from the ground up with proper software engineering practices.

## Architecture

```
PDF Upload → Text Extraction
           → Transformer Embeddings (Level 1)
           → BiGRU + Hierarchical Attention Prediction (Level 2)
           → Occlusion-based Explanation (Level 3)
           → InLegalBERT Summary
           → Streamlit UI + Legal Chatbot (RAG)
```

## Models

| Encoder | Classifier | Status |
|---------|-----------|--------|
| XLNet-base | 3x BiGRU + HAN Attention | 🔜 Pending |
| RoBERTa-base | 3x BiGRU + HAN Attention | 🔜 Pending |
| BERT-base | 3x BiGRU + HAN Attention | 🔜 Pending |
| DistilBERT | 3x BiGRU + HAN Attention | 🔜 Pending |

## Project Structure

```
├── configs/                    # YAML configuration files
│   ├── models/                 # Per-model hyperparameters
│   ├── training.yaml           # Training pipeline config
│   └── app.yaml                # Streamlit app config
├── src/
│   ├── data/                   # Dataset loading, preprocessing
│   ├── models/                 # Encoder + classifier architectures
│   ├── training/               # Training loops, embedding generation
│   ├── inference/              # Prediction, explanation, summarization
│   ├── evaluation/             # Metrics and model comparison
│   ├── chatbot/                # RAG pipeline for legal Q&A
│   └── utils/                  # Logging, config, device management
├── scripts/                    # CLI scripts for training & evaluation
├── app/                        # Streamlit application
├── trained_models/             # Saved model weights (Git LFS)
├── data/                       # Datasets and embeddings (Git LFS)
└── tests/                      # Unit tests
```

## Quick Start

```bash
# Clone
git clone https://github.com/srujankothuri/Legal-prediction-explanation.git
cd Legal-prediction-explanation

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run (after models are trained)
streamlit run app/streamlit_app.py
```

## Dataset

[ILDC — Indian Legal Documents Corpus](https://github.com/Exploration-Lab/ILDC)
- 34,816 Indian Supreme Court cases
- Binary classification: accepted / rejected

## License

MIT