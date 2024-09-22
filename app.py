# app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
import fitz  # PyMuPDF

# Load the pre-trained model and tokenizer for embedding
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Create FAISS index
dimension = 768  # DistilBERT embedding size
index = faiss.IndexFlatL2(dimension)
documents = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to embed texts
def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Streamlit interface
st.title("Interactive QA Bot")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Extract text and update documents
    pdf_text = extract_text_from_pdf(uploaded_file)
    documents.append(pdf_text)
    
    # Generate embeddings for the document
    embeddings = embed_texts([pdf_text])
    index.add(embeddings)
    
    st.success("Document uploaded and processed!")

# User query input
user_query = st.text_input("Ask a question about the uploaded document:")

if user_query:
    query_embedding = embed_texts([user_query])
    D, I = index.search(query_embedding, k=3)  # Retrieve top 3 results
    retrieved_texts = [documents[i] for i in I[0]]
    
    # Combine retrieved texts into a single context for QA
    context = " ".join(retrieved_texts)
    
    # Use the QA pipeline to extract the answer
    answer = qa_pipeline(question=user_query, context=context)
    
    st.write("### Answer:")
    st.write(answer['answer'])
