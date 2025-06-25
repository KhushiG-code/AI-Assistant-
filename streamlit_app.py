import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from constants import *

st.set_page_config(page_title="PDF Q&A Bot", layout="wide")

st.title("🤖 Ask Your PDF")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question from your PDF")

if st.button("Submit") and pdf_file and question:
    with st.spinner("Reading PDF and thinking..."):
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(pdf_file, tmp)
            tmp_path = Path(tmp.name)

        config = {
            "pdf_path": str(tmp_path)
        }

        pdfqa = PdfQA(config)
        pdfqa.init_embeddings()
        pdfqa.init_models()
        pdfqa.vector_db_pdf()
        pdfqa.retreival_qa_chain()

        answer = pdfqa.answer_query(question)
        st.success("Answer:")
        st.write(answer)
