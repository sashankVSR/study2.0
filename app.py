# app.py

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Load Hugging Face Inference API client (StarCoder2 3B)
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")  # add in .streamlit/secrets.toml
client = InferenceClient(model="bigcode/starcoder2-3b", token=HUGGINGFACE_API_KEY)

# Prompt format
def build_prompt(context, question):
    return f"""### Context
{context}

### Question
{question}

### Answer
"""

# LLM Response
def generate_llm_response(context: str, question: str) -> str:
    try:
        prompt = build_prompt(context, question)
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.2,
            stop=["###"]
        )
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_file) -> List[str]:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text_chunks = []
    for page in doc:
        text = page.get_text("text")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        text_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip()])
    return text_chunks

# Clean text
def preprocess_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

# Create FAISS index
def create_faiss_index(chunks: List[str], model):
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Retrieve relevant chunks
def retrieve_relevant_chunks(question: str, chunks: List[str], model, index, k=3):
    q_embed = model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embed, k)
    return [chunks[i] for i in indices[0]]

# Main app
def main():
    st.set_page_config("StudyMate", page_icon="📘", layout="centered")
    st.title("📘 StudyMate")
    st.markdown("Upload a PDF and ask questions about it.")

    # Load embedding model
    embed_model = load_embedding_model()

    # Session state
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'index' not in st.session_state:
        st.session_state.index = None

    # Upload
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.session_state.chunks = []
        for file in uploaded_files:
            with st.spinner(f"Reading {file.name}..."):
                chunks = extract_text_from_pdf(file)
                chunks = [preprocess_text(c) for c in chunks]
                st.session_state.chunks.extend(chunks)
        if st.session_state.chunks:
            st.session_state.index, _ = create_faiss_index(st.session_state.chunks, embed_model)
            st.success("PDF processed and indexed!")

    # Question
    question = st.text_input("Ask a question about your documents")

    if st.button("Get Answer") and question and st.session_state.index:
        with st.spinner("Generating answer..."):
            context_chunks = retrieve_relevant_chunks(question, st.session_state.chunks, embed_model, st.session_state.index)
            context = " ".join(context_chunks)
            answer = generate_llm_response(context, question)
            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("View Retrieved Context"):
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**Excerpt {i}**: {chunk[:300]}...")

if __name__ == "__main__":
    main()
