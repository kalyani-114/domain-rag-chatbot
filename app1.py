import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Domain RAG Chatbot")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stChatMessage {
        background-color: #F0EEFB;
        border-radius: 12px;
        padding: 10px;
    }

    h1, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1 {
        color: #6C5CE7 !important;
    }

    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Domain-Specific RAG Chatbot")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages_data = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({"page_number": page_number, "text": text})
    return pages_data

def chunk_pages(pages_data, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for page in pages_data:
        for chunk_text in splitter.split_text(page["text"]):
            all_chunks.append({"page_number": page["page_number"], "text": chunk_text})
    return all_chunks

def build_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_chunks(question, index, chunks, top_k=3):
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    return [chunks[idx] for idx in indices[0]]

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""You are a document question-answering assistant.
Answer only from the supplied context. If the answer is not available, say:
If the context contains the answer, give only the answer with no refusal language. Only use the refusal sentence if the context truly does not contain the answer.

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents"):
        all_pages = []
        for file in uploaded_files:
            all_pages.extend(extract_text_from_pdf(file))
        chunks = chunk_pages(all_pages)
        st.session_state.index = build_faiss_index(chunks)
        st.session_state.chunks = chunks
        st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    if st.session_state.index is None:
        answer = "Please upload and process a PDF first."
    else:
        retrieved = retrieve_chunks(question, st.session_state.index, st.session_state.chunks)
        answer = generate_answer(question, retrieved)
        if "could not find this information" not in answer.lower():
            sources = ", ".join(set(f"Page {c['page_number']}" for c in retrieved))
            answer += f"\n\n**Source:** {sources}"

    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)
