# ⚖️ Legal Judgment Prediction & Explanation

AI-powered Indian Supreme Court judgment prediction with sentence-level explainability and a RAG-based legal chatbot.

> **Published**: Springer ICDSA 2025 — *Court Judgment Prediction using Hierarchical Attention Networks*

---

## Overview

This system predicts whether an Indian Supreme Court case will be **accepted** or **rejected**, explains which sentences influenced the prediction, and provides an interactive legal chatbot for Indian law queries.

### Architecture

```
PDF Upload → Text Extraction (PyPDF2)
  │
  ├─ Level 1: Transformer Encoder (fine-tuned on ILDC)
  │            Tokenize → chunk (510 tokens, 410 stride) → extract last-4-layer hidden states
  │            Output: (n_chunks, 3072) embeddings per document
  │
  ├─ Level 2: Hierarchical Attention Network
  │            3x BiGRU (200 units) → Attention (400 dim) → Dense → Sigmoid
  │            Output: binary prediction + attention coefficients
  │
  ├─ Level 3: Occlusion-based Explanation
  │            Mask each sentence → measure confidence drop → rank by importance
  │            Output: key sentences supporting the prediction
  │
  ├─ Summary: InLegalBERT Extractive Summarization
  │            Sentence embeddings → cosine similarity centrality → top-k selection
  │
  └─ Chatbot: RAG (FAISS + Mistral-7B via Together AI)
              Legal document retrieval → context-augmented answer generation
```

---

## Models

| Encoder | Classifier | Embedding Dim | Dataset |
|---------|-----------|---------------|---------|
| XLNet-base-cased | 3x BiGRU + HAN Attention | 3072 | ILDC Single |
| RoBERTa-base | 3x BiGRU + HAN Attention | 3072 | ILDC Single |
| BERT-base-uncased | 3x BiGRU + HAN Attention | 3072 | ILDC Single |
| DistilBERT-base | 3x BiGRU + HAN Attention | 3072 | ILDC Single |

Results are available on the **Model Comparison** page in the Streamlit app after training.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/srujankothuri/Legal-prediction-explanation.git
cd Legal-prediction-explanation
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your Together AI API key for the chatbot
```

### 3. Train Models (requires GPU)

```bash
# Train all four encoder variants end-to-end
make train-all

# Or train individually:
make train-encoder MODEL=xlnet
make generate-embeddings MODEL=xlnet
make train-classifier MODEL=xlnet
```

### 4. Build Vector Database (for chatbot)

```bash
make build-vectordb
```

### 5. Run the Application

```bash
make app
# Opens at http://localhost:8501
```

---

## Project Structure

```
├── configs/                          # YAML configuration (no hardcoded hyperparams)
│   ├── models/
│   │   ├── xlnet_bigru.yaml          # XLNet encoder + HAN classifier config
│   │   ├── roberta_bigru.yaml
│   │   ├── bert_bigru.yaml
│   │   └── distilbert_bigru.yaml
│   ├── training.yaml                 # Dataset paths, training settings
│   └── app.yaml                      # Streamlit + chatbot config
│
├── src/                              # Core source code
│   ├── data/
│   │   ├── dataset.py                # ILDC dataset loader with validation
│   │   ├── preprocessing.py          # Tokenization, chunking, padding
│   │   └── pdf_extractor.py          # PDF text extraction for legal docs
│   ├── models/
│   │   ├── encoders.py               # Unified encoder registry (XLNet, RoBERTa, BERT, DistilBERT)
│   │   ├── attention.py              # Hierarchical Attention Layer (Yang et al. 2016)
│   │   └── han_classifier.py         # BiGRU + Attention document classifier
│   ├── training/
│   │   ├── trainer.py                # Encoder fine-tuning loop (AdamW + warmup)
│   │   ├── embeddings.py             # Embedding generation from fine-tuned encoders
│   │   └── callbacks.py              # Batch generators, LR scheduling
│   ├── inference/
│   │   ├── predictor.py              # End-to-end prediction pipeline
│   │   ├── explainer.py              # Occlusion-based sentence explanation
│   │   └── summarizer.py             # InLegalBERT extractive summarization
│   ├── evaluation/
│   │   └── metrics.py                # Precision, recall, F1, confusion matrix
│   ├── chatbot/
│   │   ├── vector_store.py           # FAISS index builder + loader
│   │   └── rag_chain.py              # LangChain RAG with Mistral-7B
│   └── utils/
│       ├── logger.py                 # Centralized logging (console + file)
│       ├── config.py                 # YAML config loader with dot-access
│       └── device.py                 # Auto device detection (CUDA/MPS/CPU)
│
├── scripts/                          # CLI entry points
│   ├── train_encoder.py              # Fine-tune any encoder
│   ├── generate_embeddings.py        # Generate chunk embeddings
│   ├── train_classifier.py           # Train HAN on embeddings
│   ├── evaluate.py                   # Evaluate all models + comparison table
│   └── build_vector_db.py            # Build FAISS index for chatbot
│
├── app/                              # Streamlit application
│   ├── streamlit_app.py              # Main entry point
│   ├── pages/
│   │   ├── home.py                   # Project overview
│   │   ├── predict.py                # PDF upload → prediction → explanation
│   │   ├── compare_models.py         # Side-by-side model comparison
│   │   └── chatbot.py                # Legal Q&A chatbot
│   └── components/
│       ├── model_selector.py         # Model dropdown with availability status
│       └── result_display.py         # Prediction result rendering
│
├── tests/                            # Unit tests (pytest)
├── trained_models/                   # Saved weights (Git LFS)
├── data/
│   ├── raw/                          # ILDC datasets (Git LFS)
│   └── embeddings/                   # Pre-computed embeddings (Git LFS)
├── vector_db/                        # FAISS index
├── logs/                             # Training logs, evaluation results
├── assets/                           # UI images
├── Makefile                          # make train-all, make evaluate, make app
├── requirements.txt
├── .env.example
└── .gitattributes                    # Git LFS tracking rules
```

---

## Training Pipeline

The full training pipeline for one model variant:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Fine-tune Encoder                                       │
│     xlnet-base-cased + ILDC labels → AdamW (lr=2e-6, 5ep)  │
│     Output: trained_models/xlnet_finetuned/                 │
├─────────────────────────────────────────────────────────────┤
│  2. Generate Embeddings                                     │
│     Each doc → overlapping 512-token chunks → last 4 layers │
│     Output: data/embeddings/xlnet/{train,dev,test}.npy      │
├─────────────────────────────────────────────────────────────┤
│  3. Train HAN Classifier                                    │
│     Embeddings → 3x BiGRU → Attention → Dense → Sigmoid    │
│     Output: trained_models/han_xlnet.h5                     │
├─────────────────────────────────────────────────────────────┤
│  4. Evaluate                                                │
│     Test set → metrics + comparison table                   │
│     Output: logs/evaluation_results.json                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

[ILDC — Indian Legal Documents Corpus](https://github.com/Exploration-Lab/ILDC)

- **ILDC_single**: ~8,000 cases (single hearing per case)
- Binary labels: accepted (1) / rejected (0)
- Pre-split into train / dev / test

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make train-encoder MODEL=xlnet` | Fine-tune an encoder |
| `make generate-embeddings MODEL=xlnet` | Generate embeddings |
| `make train-classifier MODEL=xlnet` | Train HAN classifier |
| `make train-all` | Train all 4 model variants |
| `make evaluate` | Evaluate all models |
| `make build-vectordb` | Build FAISS index for chatbot |
| `make app` | Launch Streamlit application |
| `make test` | Run all unit tests |
| `make clean` | Remove cached files and logs |

---

## Citation

```bibtex
@inproceedings{kothuri2025court,
  title={Court Judgment Prediction using Hierarchical Attention Networks},
  author={Kothuri, Venkata Srujan and others},
  booktitle={International Conference on Data Science and Applications (ICDSA)},
  year={2025},
  publisher={Springer}
}
```

## License

MIT