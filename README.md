# Domain-Specific RAG Chatbot

A chatbot that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG). Built to answer only from the uploaded content, with source page references, and a clear refusal when the answer isn't available.

## Features
- Upload one or more PDF documents
- Automatic text extraction, chunking, and embedding
- Fast retrieval of the most relevant sections using FAISS
- Grounded answers generated via Groq LLM (llama-3.1-8b-instant)
- Displays source document and page number for every answer
- Refuses to answer when information isn't found in the documents

## Tech Stack
- **Language:** Python
- **PDF extraction:** pypdf
- **Text splitting:** LangChain text splitters
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector database:** FAISS
- **LLM:** Groq (llama-3.1-8b-instant)
- **Interface:** Streamlit

## How It Works
1. User uploads PDF file(s)
2. Text is extracted page by page, with page numbers preserved
3. Text is split into overlapping chunks (~800 characters)
4. Each chunk is converted into a numerical embedding
5. Embeddings are stored in a FAISS vector index
6. When a question is asked, it's embedded and compared against stored chunks
7. The top matching chunks are sent to the LLM along with the question
8. The LLM generates an answer using only the provided context, with source pages shown

## Setup

1. Clone this repository
```bash
git clone <your-repo-url>
cd domain-rag-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Groq API key
4. 4. Run the app
```bash
streamlit run app1.py
```

## Usage
1. Upload a PDF using the sidebar
2. Click "Process Documents"
3. Ask a question in the chat box
4. View the answer along with its source page number

## Limitations
- Answers are only as good as the text extracted from the PDF (scanned/image-based PDFs may not extract text without OCR)
- Currently supports single-session use (chat history resets on refresh)
- Free-tier LLM API may have rate limits

