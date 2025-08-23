import pdfplumber
import numpy as np
import faiss
import streamlit as st
import requests
from dotenv import load_dotenv
import os

# ---------------- Load environment variables ----------------
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")
EURI_CHAT_URL = os.getenv("EURI_CHAT_URL")
EURI_EMBEDDED_URL = os.getenv("EURI_EMBEDDED_URL")

# ---------------- Initialize conversation memory ----------------
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# ---------------- Text Chunking ----------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------- Generate Embeddings (Batch) ----------------
def generate_embeddings(texts):
    """
    Generates embeddings for a list of texts in a single API call.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {"input": texts, "model": "text-embedding-3-small"}
    response = requests.post(EURI_EMBEDDED_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    embeddings = [np.array(item['embedding'], dtype='float32') for item in data['data']]
    return embeddings

# ---------------- Build Vector Store with Caching ----------------
@st.cache_data(show_spinner=False)
def build_vector_store_cached(chunks):
    embeddings = generate_embeddings(chunks)  # batch embeddings
    embeddings = np.vstack(embeddings).astype('float32')  # 2D array
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# ---------------- Retrieve Context ----------------
def retrieve_context(question, chunks, index, embeddings, top_k=3):
    q_embed = generate_embeddings([question])[0]
    q_embed = np.array(q_embed, dtype='float32').reshape(1, -1)

    if q_embed.shape[1] != embeddings.shape[1]:
        raise ValueError(f"Query embedding dimension ({q_embed.shape[1]}) "
                         f"does not match index dimension ({embeddings.shape[1]})")

    D, I = index.search(q_embed, top_k)
    return "\n\n".join([chunks[i] for i in I[0]])

# ---------------- Ask EURI with Context ----------------
def ask_euri_with_context(question, context, memory):
    messages = [{"role": "system", "content": "You are a helpful assistant answering questions from a document."}]
    if memory:
        messages.extend(memory)

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    res.raise_for_status()
    reply = res.json()['choices'][0]['message']['content']

    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": reply})

    return reply

# ---------------- Streamlit UI ----------------
st.title("📄 PDF Knowledge Extraction RAG Bot (Fast)")

uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
user_question = st.text_input("Ask a question about the document")

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf("temp.pdf")
        chunks = chunk_text(full_text)

    with st.spinner("Generating embeddings and building vector store..."):
        index, embeddings = build_vector_store_cached(chunks)

    st.success("PDF loaded and indexed successfully!")

    if user_question:
        with st.spinner("Retrieving context and generating answer..."):
            context = retrieve_context(user_question, chunks, index, embeddings)
            response = ask_euri_with_context(user_question, context, st.session_state.conversation_memory)

        st.markdown("### ✅ Answer:")
        st.write(response)

        
        
