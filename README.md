# Automating Court Judgement Prediction and Explanation for Indian Legal Cases

A research-backed project that predicts the likely outcome of an Indian legal case (**Accepted/Rejected**) and generates supporting explanation signals from the judgement text.

This repository also includes:
- legal case **summarization**,
- outcome **prediction**,
- outcome **explanation**, and
- a retrieval-augmented **LegalSmart chatbot** for Indian legal queries.

## Publication

This project is based on the published Springer conference paper:

- **Automating Court Judgement Prediction and Explanation for Indian Legal Cases**
  https://link.springer.com/chapter/10.1007/978-3-032-12827-0_7

A local copy of the paper is available in this repository:
- `Automating_Court_Judgement_Prediction_and_Explanation_for_Indian_Legal_Cases_springer.pdf`

## Project Highlights

- Hierarchical judgement prediction pipeline using:
  - **XLNet** for contextual sentence/document representations,
  - **BiGRU + Attention** for sequence-level decision modeling,
  - final **sigmoid/softmax-style classification** for outcome label generation.
- Reported model performance from the research work: **~74% accuracy**.
- Streamlit-based interface for end-to-end legal document interaction.
- RAG-enabled legal chatbot built with FAISS + LangChain + Together API.

## Repository Structure

```text
.
├── app.py                              # Streamlit app entrypoint
├── Comb_FINAL.py                       # Core utility functions (PDF parsing, embedding pipeline, explanation utilities)
├── README.md
├── Automating_Court_Judgement_...pdf   # Springer paper (local copy)
├── Prediction_full_xlnet/
│   └── XGA_concat_epoch1_3.h5          # Trained prediction model weights
├── L1_Output/
│   └── embeds.npy                      # Intermediate embeddings
├── law_vector_db/
│   ├── index.faiss
│   └── index.pkl                       # Chatbot retrieval index
└── chatbot_legal.jpeg                  # Chatbot banner image
```

## Core Components

### 1) Summarization
- Extracts text from uploaded case PDFs.
- Generates case summaries using **InLegalBERT** (`law-ai/InLegalBERT`).

### 2) Prediction
- Generates XLNet embeddings.
- Uses hierarchical BiGRU + attention network with stored weights (`.h5`) to predict case outcome.

### 3) Explanation
- Uses chunk/sentence-level contribution behavior to produce explanation signals supporting the prediction.

### 4) LegalSmart Chatbot
- Uses a FAISS vector database (`law_vector_db/`) and HuggingFace embeddings.
- Runs a conversational retrieval chain with a legal-domain prompt template.

## Quick Start (Developer Setup)

> The current codebase has research-prototype style dependencies; create an isolated environment before running.

### 1. Clone and enter the repository
```bash
git clone <your-repo-url>
cd Legal-prediction-explanation
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows PowerShell
```

### 3. Install dependencies
Install the project dependencies used by the Streamlit app, Transformers, TensorFlow/Keras, PyTorch, NLTK, and LangChain ecosystem packages.

If you maintain a `requirements.txt`, install with:
```bash
pip install -r requirements.txt
```

Otherwise, install the imports required by `app.py` and `Comb_FINAL.py` manually.

### 4. Run the app
```bash
streamlit run app.py
```

## Usage Flow

1. Open the Streamlit app.
2. Upload a legal case PDF in the **Summarization** tab.
3. Review generated summary.
4. Switch to **Prediction** to get the predicted judgement outcome.
5. Switch to **Explanation** for rationale-oriented explanation outputs.
6. Use **LegalSmart Chatbot** for legal Q&A grounded in the vector database.

## Notes and Limitations

- This repository is primarily a **research implementation** and may require environment tuning (CUDA, model versions, tokenizer/model artifacts).
- Some model artifacts are expected locally (e.g., XLNet directory and weight files).
- Chatbot functionality depends on a valid Together API key and available vector store files.
- Outputs are intended for research/assistance, not legal advice.

## Citation

If this project helps your work, please cite the Springer publication listed above.
