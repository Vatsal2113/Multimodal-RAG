# Multimodal RAG Pipeline

End-to-end multimodal RAG pipeline for PDF documents: extracts text, equations, tables and images; cleans OCR artifacts; auto-captions tables/images via Google Gemini; stores chunks in SQLite; embeds with HuggingFace + Chroma; offers an interactive REPL for retrieval and LLM-based Q&A.

## Features

- **Text & Equation Extraction** with hybrid chunking and OCR fallback  
- **Table Extraction** to Markdown plus one-sentence captions via Gemini  
- **Image Extraction** (PNG) with OCR and LLM-generated summaries  
- **SQLite Registry** of all “chunks” (text, equations, tables, images)  
- **Vector Indexing** using HuggingFace Embeddings + Chroma  
- **Interactive REPL** for:  
  - Listing sources  
  - Showing figures/tables by label  
  - “Find image by summary”  
  - Free-form Q&A over your documents  

## Requirements

- **Python** 3.8+  
- **Tesseract OCR** installed and on `$PATH`  
- **Google Gemini API key**  
- System dependencies for PDF parsing (e.g. `libgl1`, `poppler-utils`, etc.)  
- Python packages (see `requirements.txt` below)

## Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/your-username/multimodal-rag-pipeline.git
   cd multimodal-rag-pipeline
