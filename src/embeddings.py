from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings