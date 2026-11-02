# Enterprise RAG Document Intelligence Assistant

A resume-aligned, runnable PDF question-answering demo built with Python, LangChain, FAISS, local sentence-transformer embeddings, and Streamlit. It chunks PDF text at 800 characters with 150-character overlap, retrieves the top 4 passages, and displays source and page citations.

## Features

- Upload one PDF per session and extract page-aware text with LangChain's PyPDFLoader.
- Split text with `chunk_size=800` and `chunk_overlap=150`.
- Embed locally with `sentence-transformers/all-MiniLM-L6-v2` and store vectors in FAISS.
- Retrieve the four most similar chunks for each question.
- Show source filename, page number, and retrieved text beside the answer.
- Grounding prompt instructs a configured LLM to use only retrieved context and admit when the evidence is insufficient.
- Optional OpenAI-compatible answer generation. Without an API key, the app uses a clearly labeled extractive fallback so retrieval and citations can still be demonstrated.

## Setup (Windows PowerShell)

Python 3.10–3.12 is recommended for the broadest compatibility with FAISS and PyTorch wheels.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

On macOS/Linux, create and activate a virtual environment with `python3 -m venv .venv`, then run `pip install -r requirements.txt` and `streamlit run app.py`.

The first upload downloads the embedding model from Hugging Face and may take a few minutes. Internet access is needed on first use unless the model is already cached.

## Enable generated answers (optional)

Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in `.env`. The answer model is used only after local retrieval; retrieved passages remain visible for verification. Without a key the app answers with a short extractive response from the highest-ranked passages. This fallback is not an LLM response.

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Never commit `.env` or share API keys. The default is no API key and no paid LLM call.

## Run

1. Start Streamlit with `streamlit run app.py`.
2. Upload a text-based PDF in the sidebar.
3. Wait for indexing to finish.
4. Ask a question about the document.
5. Check the source/page citations and expand retrieved passages to verify the answer.

Scanned/image-only PDFs need OCR before ingestion; OCR is not included in this demo.

## Project layout

```text
app.py                    Streamlit upload, chat, citations, and status UI
rag_assistant/             Ingestion, indexing, retrieval, and answer generation
  pipeline.py
requirements.txt
.env.example
.gitignore
README.md
tests/                     Lightweight pipeline behavior tests
```

## Architecture

`PDF upload → PyPDFLoader (per-page metadata) → RecursiveCharacterTextSplitter (800/150) → HuggingFace sentence-transformer embeddings → FAISS index → top-4 similarity retrieval → grounded answer → citations in Streamlit`

Page metadata is converted from PyPDFLoader's zero-based `page` to a human-facing one-based page number. Retrieval scores are not exposed as calibrated confidence probabilities.

## Interview explanation

“I load each PDF page with LangChain so page metadata travels with the text. I split the text into 800-character chunks with 150 characters of overlap to preserve context across boundaries. A local MiniLM sentence-transformer maps each chunk to an embedding, and FAISS indexes those vectors. For a question, I retrieve the four nearest chunks and pass only those chunks to a grounding-focused prompt. The UI shows the evidence and page citations. If no LLM key is configured, a labeled extractive fallback keeps the demo usable while making clear it is not a generated answer.”

Be precise in the interview: the LLM answer mode needs a configured API key; embeddings and FAISS retrieval are local. The demo does not claim production-grade access control, OCR, persistent indexing, or a retrieval evaluation benchmark.

## Troubleshooting

- **FAISS / torch install errors:** use Python 3.10, 3.11, or 3.12 and a fresh virtual environment. Windows wheel availability can differ by Python version.
- **Model download stalls:** check network access to Hugging Face or pre-cache `sentence-transformers/all-MiniLM-L6-v2`.
- **No text found:** the PDF may be scanned; OCR preprocessing is required.
- **LLM unavailable:** remove the API key to use the extractive fallback, or check key/model/provider configuration.
- **Wrong answer:** inspect the four retrieved chunks first. Retrieval quality depends on PDF text quality, chunk boundaries, embedding model, and whether the question uses document terminology.

## Verification

The source was syntax-checked in the creation environment. Full app execution requires installing `requirements.txt`, the embedding model download, and (for generated answers) an API key. See `tests/` for small behavior checks. Do not describe the app as fully end-to-end tested until it has been run locally with a real PDF.
