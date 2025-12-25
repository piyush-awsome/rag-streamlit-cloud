import streamlit as st
from rag import create_rag_chain
import os

st.set_page_config(
    page_title="RAG Document Q&A",
    layout="centered"
)

st.title("📄 RAG Document Q&A")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)

    pdf_path = "data/temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Building RAG pipeline..."):
        qa = create_rag_chain(pdf_path)

    question = st.text_input(
        "Ask a question",
        key="question_input"
    )

    if question:
        with st.spinner("Generating answer..."):
            answer = qa.run(question)

        st.subheader("Answer")
        st.write(answer)
